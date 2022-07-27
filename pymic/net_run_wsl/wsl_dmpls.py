# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from pymic.io.nifty_dataset import NiftyDataset
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.loss.seg.dice import DiceLoss
from pymic.loss.seg.ssl import TotalVariationLoss
from pymic.net_run.agent_seg import SegmentationAgent
from pymic.net_run_wsl.wsl_em import WSL_EntropyMinimization
from pymic.util.ramps import sigmoid_rampup

class WSL_DMPLS(WSL_EntropyMinimization):
    """
    Implementation of the following paper:
        Xiangde Luo, Minhao Hu, Wenjun Liao, Shuwei Zhai, Tao Song, Guotai Wang,
        Shaoting Zhang. ScribblScribble-Supervised Medical Image Segmentation via 
        Dual-Branch Network and Dynamically Mixed Pseudo Labels Supervision.
        MICCAI 2022. 
    """
    def __init__(self, config, stage = 'train'):
        net_type = config['network']['net_type']
        if net_type not in ['DualBranchUNet2D', 'DualBranchUNet3D']:
            raise ValueError("""For WSL_DMPLS, a dual branch network is expected. \
                It only supports DualBranchUNet2D and DualBranchUNet3D currently.""")
        super(WSL_DMPLS, self).__init__(config, stage)

    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        wsl_cfg     = self.config['weakly_supervised_learning']
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
            
            iter_max = self.config['training']['iter_max']
            ramp_up_length = wsl_cfg.get('ramp_up_length', iter_max)
            regular_w = 0.0
            if(self.glob_it > wsl_cfg.get('iter_sup', 0)):
                regular_w = wsl_cfg.get('regularize_w', 0.1)
                if(ramp_up_length is not None and self.glob_it < ramp_up_length):
                    regular_w = regular_w * sigmoid_rampup(self.glob_it, ramp_up_length)
            loss = loss_sup + regular_w*loss_reg

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

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
        train_avg_dice = train_cls_dice.mean()

        train_scalers = {'loss': train_avg_loss, 'loss_sup':train_avg_loss_sup,
            'loss_reg':train_avg_loss_reg, 'regular_w':regular_w,
            'avg_dice':train_avg_dice,     'class_dice': train_cls_dice}
        return train_scalers
        