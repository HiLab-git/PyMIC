# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import torch
import torch.nn as nn
import numpy as np
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.util.ramps import sigmoid_rampup
from pymic.net_run_ssl.ssl_em import SSLSegAgent

class SSLURPC(SSLSegAgent):
    """
    Uncertainty-Rectified Pyramid Consistency according to the following paper:
    Xiangde Luo, Wenjun Liao, Jieneng Chen, Tao Song, Yinan Chen,
    Shichuan Zhang, Nianyong Chen, Guotai Wang, Shaoting Zhang. 
    Efficient Semi-supervised Gross Target Volume of Nasopharyngeal Carcinoma 
    Segmentation via Uncertainty Rectified Pyramid Consistency.
    MICCAI 2021, pp. 318-329.
    https://arxiv.org/abs/2012.07042 
    """
    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        ssl_cfg     = self.config['semi_supervised_learning']
        train_loss  = 0
        train_loss_sup = 0
        train_loss_unsup = 0
        train_dice_list = []
        self.net.train()
        kl_distance = nn.KLDivLoss(reduction='none')
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

            # forward pass
            outputs_list = self.net(inputs)
            n0 = list(x0.shape)[0] 

            # get supervised loss
            p0 = [output_i[:n0] for output_i in outputs_list]
            loss_sup = self.get_loss_value(data_lab, p0, y0)

            # get average probability across scales
            outputs_soft_list = [torch.softmax(item, dim=1) for item in outputs_list]
            outputs_soft_avg  = torch.mean(torch.stack(outputs_soft_list),dim = 0)
            p1_avg = outputs_soft_avg[n0:] * 0.99 + 0.005 # for unannotated images

            # unsupervised loss
            loss_unsup = 0.0
            for soft_i in outputs_soft_list:
                p1_i = soft_i[n0:] * 0.99 + 0.005
                var  = torch.sum(kl_distance(
                        torch.log(p1_i), p1_avg), dim=1, keepdim=True)
                exp_var = torch.exp(-var)            
                square_e= torch.square(p1_avg - p1_i)
                loss_i  = torch.mean(square_e * exp_var)  / \
                            (torch.mean(exp_var) + 1e-8) + torch.mean(var)
                loss_unsup += loss_i
            loss_unsup = loss_unsup / len(outputs_list)
                        
            iter_max = self.config['training']['iter_max']
            ramp_up_length = ssl_cfg.get('ramp_up_length', iter_max)
            consis_w = 0.0
            if(self.glob_it > ssl_cfg.get('iter_sup', 0)):
                consis_w = ssl_cfg.get('consis_w', 0.1)
                if(ramp_up_length is not None and self.glob_it < ramp_up_length):
                    consis_w = consis_w * sigmoid_rampup(self.glob_it, ramp_up_length)

            loss = loss_sup + consis_w*loss_unsup

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_loss = train_loss + loss.item()
            train_loss_sup   = train_loss_sup + loss_sup.item()
            train_loss_unsup = train_loss_unsup + loss_unsup.item() 
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
        train_avg_loss_unsup = train_loss_unsup / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis = 0)
        train_avg_dice = train_cls_dice.mean()

        train_scalers = {'loss': train_avg_loss, 'loss_sup':train_avg_loss_sup,
            'loss_unsup':train_avg_loss_unsup, 'consis_w':consis_w,
            'avg_dice':train_avg_dice,     'class_dice': train_cls_dice}
        return train_scalers
