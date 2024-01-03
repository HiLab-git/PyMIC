# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.net_run.semi_sup import SSLSegAgent
from pymic.util.ramps import get_rampup_ratio

def sharpening(P, T = 0.1):
    T = 1.0/T
    P_sharpen = P**T / (P**T + (1-P)**T)
    return P_sharpen

class SSLMCNet(SSLSegAgent):
    """
    Mutual Consistency Learning for semi-supervised segmentation. It requires a network 
    with multiple decoders for learning, such as `pymic.net.net2d.unet2d_mcnet.MCNet2D`.

    * Reference: Yicheng Wu, Zongyuan Ge et al. Mutual consistency learning for 
    semi-supervised medical image segmentation.
      `MIA 2022. <https://doi.org/10.1016/j.media.2022.102530>`_ 

    The original code is at: https://github.com/ycwu1997/MC-Net

    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::
        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `semi_supervised_learning` is needed. See :doc:`usage.ssl` for details.
    """
    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        ssl_cfg     = self.config['semi_supervised_learning']
        iter_max     = self.config['training']['iter_max']
        rampup_start = ssl_cfg.get('rampup_start', 0)
        rampup_end   = ssl_cfg.get('rampup_end', iter_max)
        temperature  = ssl_cfg.get('temperature', 0.1)
        unsup_loss_name = ssl_cfg.get('unsupervised_loss', "MSE") 
        train_loss  = 0
        train_loss_sup = 0
        train_loss_reg = 0
        train_dice_list = []
        self.net.train()
        
        for it in range(iter_valid):
            try:
                data_lab = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data_lab = next(self.trainIter)
            try:
                data_unlab = next(self.trainIter_unlab)
            except StopIteration:
                self.trainIter_unlab = iter(self.train_loader_unlab)
                data_unlab = next(self.trainIter_unlab)

            # get the inputs
            x0   = self.convert_tensor_type(data_lab['image'])
            y0   = self.convert_tensor_type(data_lab['label_prob'])  
            x1   = self.convert_tensor_type(data_unlab['image'])
            inputs = torch.cat([x0, x1], dim = 0)               
            inputs, y0 = inputs.to(self.device), y0.to(self.device)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward pass to obtain multiple predictions
            outputs     = self.net(inputs)
            num_outputs = len(outputs)
            n0 = list(x0.shape)[0] 
            p0 = F.softmax(outputs[0], dim=1)[:n0]
            # for probability prediction and pseudo respectively
            p_ori = torch.zeros((num_outputs,) + outputs[0].shape) 
            y_psu = torch.zeros((num_outputs,) + outputs[0].shape)

             # get supervised loss
            loss_sup = 0
            for idx in range(num_outputs):
                p0i = outputs[idx][:n0]
                loss_sup += self.get_loss_value(data_lab, p0i, y0)

                # get pseudo labels
                p_i = F.softmax(outputs[idx], dim=1)
                p_ori[idx] = p_i
                y_psu[idx] = sharpening(p_i, temperature) 
            
            # get regularization loss
            loss_reg = 0.0
            for i in range(num_outputs):
                for j in range(num_outputs):
                    if (i!=j):
                        loss_reg += F.mse_loss(p_ori[i], y_psu[j], reduction='mean') 
                
            rampup_ratio = get_rampup_ratio(self.glob_it, rampup_start, rampup_end, "sigmoid")
            regular_w = ssl_cfg.get('regularize_w', 0.1) * rampup_ratio
            loss = loss_sup + regular_w*loss_reg

            loss.backward()
            self.optimizer.step()

            train_loss = train_loss + loss.item()
            train_loss_sup = train_loss_sup + loss_sup.item()
            train_loss_reg = train_loss_reg + loss_reg.item() 
            # get dice evaluation for each class in annotated images
            if(isinstance(p0, tuple) or isinstance(p0, list)):
                p0 = p0[0] 
            p0_argmax = torch.argmax(p0, dim = 1, keepdim = True)
            p0_soft   = get_soft_label(p0_argmax, class_num, self.tensor_type)
            p0_soft, y0 = reshape_prediction_and_ground_truth(p0_soft, y0) 
            dice_list   = get_classwise_dice(p0_soft, y0)
            train_dice_list.append(dice_list.cpu().numpy())
        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup = train_loss_sup / iter_valid
        train_avg_loss_reg = train_loss_reg / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis = 0)
        train_avg_dice = train_cls_dice[1:].mean()

        train_scalers = {'loss': train_avg_loss, 'loss_sup':train_avg_loss_sup,
            'loss_reg':train_avg_loss_reg, 'regular_w':regular_w,
            'avg_fg_dice':train_avg_dice,     'class_dice': train_cls_dice}
        return train_scalers
