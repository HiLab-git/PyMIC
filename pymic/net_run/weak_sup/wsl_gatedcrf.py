# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import numpy as np
import time
import torch
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.loss.seg.gatedcrf import GatedCRFLoss
from pymic.net_run.weak_sup import WSLSegAgent
from pymic.util.ramps import get_rampup_ratio

class WSLGatedCRF(WSLSegAgent):
    """
    Implementation of the Gated CRF loss for weakly supervised segmentation.
        
    * Reference: Anton Obukhov, Stamatios Georgoulis, Dengxin Dai, Luc Van Gool:
      Gated CRF Loss for Weakly Supervised Semantic Image Segmentation.
      `CoRR <http://arxiv.org/abs/1906.04651>`_, abs/1906.04651, 2019.
        
    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `weakly_supervised_learning` is needed. See :doc:`usage.wsl` for details.
    """
    def __init__(self, config, stage = 'train'):
        super(WSLGatedCRF, self).__init__(config, stage)
        # parameters for gated CRF 
        wsl_cfg = self.config['weakly_supervised_learning']
        w0 = wsl_cfg.get('GatedCRFLoss_W0'.lower(), 1.0)
        xy0= wsl_cfg.get('GatedCRFLoss_XY0'.lower(), 5)
        rgb= wsl_cfg.get('GatedCRFLoss_rgb'.lower(), 0.1)
        w1 = wsl_cfg.get('GatedCRFLoss_W1'.lower(), 1.0)
        xy1= wsl_cfg.get('GatedCRFLoss_XY1'.lower(), 3)
        kernel0 = {'weight': w0, 'xy': xy0, 'rgb': rgb}
        kernel1 = {'weight': w1, 'xy': xy1}
        self.kernels = [kernel0, kernel1]
        self.radius  = wsl_cfg.get('GatedCRFLoss_Radius'.lower(), 5.0)

    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        wsl_cfg     = self.config['weakly_supervised_learning']
        iter_max     = self.config['training']['iter_max']
        rampup_start = wsl_cfg.get('rampup_start', 0)
        rampup_end   = wsl_cfg.get('rampup_end', iter_max)
        train_loss, train_loss_sup, train_loss_reg = 0, 0, 0
        train_dice_list = []
        data_time, gpu_time, loss_time, back_time = 0, 0, 0, 0
        gatecrf_loss = GatedCRFLoss()
        self.net.train()
        for it in range(iter_valid):
            t0 = time.time()
            try:
                data = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data = next(self.trainIter)
            t1 = time.time()
            # get the inputs
            inputs = self.convert_tensor_type(data['image'])
            y      = self.convert_tensor_type(data['label_prob'])  
                         
            inputs, y = inputs.to(self.device), y.to(self.device)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            # forward + backward + optimize
            outputs = self.net(inputs)
            t2 = time.time()
            loss_sup = self.get_loss_value(data, outputs, y)

            # for gated CRF loss, the input should be like NCHW
            outputs_soft = torch.softmax(outputs, dim=1)
            input_shape  = list(inputs.shape)
            if(len(input_shape) == 5):
                [N, C, D, H, W] = input_shape
                new_shape  = [N*D, C, H, W]
                inputs = torch.transpose(inputs, 1, 2)
                inputs = torch.reshape(inputs, new_shape)
                [N, C, D, H, W] = list(outputs_soft.shape)
                new_shape    = [N*D, C, H, W]
                outputs_soft = torch.transpose(outputs_soft, 1, 2)
                outputs_soft = torch.reshape(outputs_soft, new_shape)
            batch_dict = {'rgb': inputs}
            loss_reg = gatecrf_loss(outputs_soft, self.kernels, self.radius,
                batch_dict,input_shape[-2], input_shape[-1])["loss"]
            
            rampup_ratio = get_rampup_ratio(self.glob_it, rampup_start, rampup_end, "sigmoid")
            regular_w = wsl_cfg.get('regularize_w', 0.1) * rampup_ratio
            loss = loss_sup + regular_w*loss_reg
            t3 = time.time()
            loss.backward()
            t4 = time.time()
            self.optimizer.step()

            train_loss = train_loss + loss.item()
            train_loss_sup = train_loss_sup + loss_sup.item()
            train_loss_reg = train_loss_reg + loss_reg.item() 
            # get dice evaluation for each class in annotated images
            if(isinstance(outputs, tuple) or isinstance(outputs, list)):
                outputs = outputs[0] 
            p_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
            p_soft   = get_soft_label(p_argmax, class_num, self.tensor_type)
            p_soft, y = reshape_prediction_and_ground_truth(p_soft, y) 
            dice_list   = get_classwise_dice(p_soft, y)
            train_dice_list.append(dice_list.cpu().numpy())

            data_time = data_time + t1 - t0 
            gpu_time  = gpu_time  + t2 - t1
            loss_time = loss_time + t3 - t2
            back_time = back_time + t4 - t3
        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup = train_loss_sup / iter_valid
        train_avg_loss_reg = train_loss_reg / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis = 0)
        train_avg_dice = train_cls_dice[1:].mean()

        train_scalers = {'loss': train_avg_loss, 'loss_sup':train_avg_loss_sup,
            'loss_reg':train_avg_loss_reg, 'regular_w':regular_w,
            'avg_fg_dice':train_avg_dice,     'class_dice': train_cls_dice,
            'data_time': data_time, 'forward_time':gpu_time, 
            'loss_time':loss_time, 'backward_time':back_time }
        return train_scalers
        