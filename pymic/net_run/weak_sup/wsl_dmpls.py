# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import numpy as np
import random
import torch
from torch.optim import lr_scheduler
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.loss.seg.dice import DiceLoss
from pymic.net_run.weak_sup import WSLSegAgent
from pymic.util.ramps import get_rampup_ratio

class WSLDMPLS(WSLSegAgent):
    """
    Weakly supervised segmentation based on Dynamically Mixed Pseudo Labels Supervision.

    * Reference: Xiangde Luo, Minhao Hu, Wenjun Liao, Shuwei Zhai, Tao Song, Guotai Wang,
      Shaoting Zhang. ScribblScribble-Supervised Medical Image Segmentation via 
      Dual-Branch Network and Dynamically Mixed Pseudo Labels Supervision.
      `MICCAI 2022. <https://arxiv.org/abs/2203.02106>`_ 
    
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
        super(WSLDMPLS, self).__init__(config, stage)

    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        wsl_cfg     = self.config['weakly_supervised_learning']
        iter_max     = self.config['training']['iter_max']
        rampup_start = wsl_cfg.get('rampup_start', 0)
        rampup_end   = wsl_cfg.get('rampup_end', iter_max)
        train_loss  = 0
        train_loss_sup = 0
        train_loss_reg = 0
        train_dice_list = []
        self.net.train()
        for it in range(iter_valid):
            try:
                data = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data = next(self.trainIter)
            
            # get the inputs
            inputs = self.convert_tensor_type(data['image'])
            y      = self.convert_tensor_type(data['label_prob'])  
                         
            inputs, y = inputs.to(self.device), y.to(self.device)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            # forward + backward + optimize
            outputs1, outputs2 = self.net(inputs)
            loss_sup1 = self.get_loss_value(data, outputs1, y) 
            loss_sup2 = self.get_loss_value(data, outputs2, y) 
            loss_sup  = 0.5 * (loss_sup1 + loss_sup2)

            # get pseudo label with dynamical mix
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            beta = random.random()
            pseudo_lab = beta*outputs_soft1.detach() + (1.0-beta)*outputs_soft2.detach()
            pseudo_lab = torch.argmax(pseudo_lab, dim = 1, keepdim = True)
            pseudo_lab = get_soft_label(pseudo_lab, class_num, self.tensor_type)
            
            # calculate the pseudo label supervision loss
            loss_calculator = DiceLoss()
            loss_dict1 = {"prediction":outputs1, 'ground_truth':pseudo_lab}
            loss_dict2 = {"prediction":outputs2, 'ground_truth':pseudo_lab}
            loss_reg   = 0.5 * (loss_calculator(loss_dict1) + loss_calculator(loss_dict2))
            
            rampup_ratio = get_rampup_ratio(self.glob_it, rampup_start, rampup_end, "sigmoid")
            regular_w = wsl_cfg.get('regularize_w', 0.1) * rampup_ratio
            loss = loss_sup + regular_w*loss_reg

            loss.backward()
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
        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup = train_loss_sup / iter_valid
        train_avg_loss_reg = train_loss_reg / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis = 0)
        train_avg_dice = train_cls_dice[1:].mean()

        train_scalers = {'loss': train_avg_loss, 'loss_sup':train_avg_loss_sup,
            'loss_reg':train_avg_loss_reg, 'regular_w':regular_w,
            'avg_fg_dice':train_avg_dice,     'class_dice': train_cls_dice}

        return train_scalers
        