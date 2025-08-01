# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import numpy as np
import random
import time
import torch
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.loss.seg.dice import DiceLoss
from pymic.loss.seg.ce   import CrossEntropyLoss
from pymic.net_run.weak_sup import WSLSegAgent
from pymic.util.ramps import get_rampup_ratio

class WSLDMSPS(WSLSegAgent):
    """
    Weakly supervised segmentation based on Dynamically Mixed Pseudo Labels Supervision.

    * Reference: Meng Han, Xiangde Luo, Xiangjiang Xie, Wenjun Liao, Shichuan Zhang, Tao Song,
      Guotai Wang, Shaoting Zhang. DMSPS: Dynamically mixed soft pseudo-label supervision for 
      scribble-supervised medical image segmentation.
      `Medical Image Analysis 2024. <https://www.sciencedirect.com/science/article/pii/S1361841524001993>`_ 
    
    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `weakly_supervised_learning` is needed. See :doc:`usage.wsl` for details.
    """
    def __init__(self, config, stage = 'train'):
        net_type = config['network']['net_type']
        if net_type not in ['UNet2D_DualBranch', 'UNet3D_DualBranch']:
            raise ValueError("""For WSL_DMPLS, a dual branch network is expected. \
                It only supports UNet2D_DualBranch and UNet3D_DualBranch currently.""")
        super(WSLDMSPS, self).__init__(config, stage)

    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        wsl_cfg     = self.config['weakly_supervised_learning']
        iter_max     = self.config['training']['iter_max']
        rampup_start = wsl_cfg.get('rampup_start', 0)
        rampup_end   = wsl_cfg.get('rampup_end', iter_max)
        pseudo_loss_type = wsl_cfg.get('pseudo_sup_loss', 'ce_loss')
        if (pseudo_loss_type not in ('dice_loss', 'ce_loss')):
            raise ValueError("""For pseudo supervision loss, only dice_loss and ce_loss \
                are supported.""")
        train_loss, train_loss_sup, train_loss_reg = 0, 0, 0
        train_dice_list = []
        data_time, gpu_time, loss_time, back_time = 0, 0, 0, 0
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
            outputs1, outputs2 = self.net(inputs)
            t2 = time.time()

            loss_sup1 = self.get_loss_value(data, outputs1, y) 
            loss_sup2 = self.get_loss_value(data, outputs2, y) 
            loss_sup  = 0.5 * (loss_sup1 + loss_sup2)

            # get pseudo label with dynamical mix
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            beta = random.random()
            pseudo_lab = beta*outputs_soft1.detach() + (1.0-beta)*outputs_soft2.detach()
            # pseudo_lab = torch.argmax(pseudo_lab, dim = 1, keepdim = True)
            # pseudo_lab = get_soft_label(pseudo_lab, class_num, self.tensor_type)
            
            # calculate the pseudo label supervision loss
            loss_calculator = DiceLoss() if pseudo_loss_type == 'dice_loss' else CrossEntropyLoss()
            loss_dict1 = {"prediction":outputs1, 'ground_truth':pseudo_lab}
            loss_dict2 = {"prediction":outputs2, 'ground_truth':pseudo_lab}
            loss_reg   = 0.5 * (loss_calculator(loss_dict1) + loss_calculator(loss_dict2))
            
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
            if(isinstance(outputs1, tuple) or isinstance(outputs1, list)):
                outputs1 = outputs1[0] 
            p_argmax = torch.argmax(outputs1, dim = 1, keepdim = True)
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
            'loss_time':loss_time, 'backward_time':back_time}
        return train_scalers
        