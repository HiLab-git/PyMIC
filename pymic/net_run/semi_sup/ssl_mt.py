# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import torch
import numpy as np
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.net_run.semi_sup import SSLSegAgent
from pymic.net.net_dict_seg import SegNetDict
from pymic.util.ramps import get_rampup_ratio

class SSLMeanTeacher(SSLSegAgent):
    """
    Mean Teacher for semi-supervised segmentation.

    * Reference: Antti Tarvainen, Harri Valpola: Mean teachers are better role models: 
      Weight-averaged consistency targets improve semi-supervised deep learning results.
      `NeurIPS 2017. <https://arxiv.org/abs/1703.01780>`_
    
    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `semi_supervised_learning` is needed. See :doc:`usage.ssl` for details.
    """
    def __init__(self, config, stage = 'train'):
        super(SSLMeanTeacher, self).__init__(config, stage)
        self.net_ema = None 

    def create_network(self):
        super(SSLMeanTeacher, self).create_network()
        if(self.net_ema is None):
            net_name = self.config['network']['net_type']
            if(net_name not in SegNetDict):
                raise ValueError("Undefined network {0:}".format(net_name))
            self.net_ema = SegNetDict[net_name](self.config['network'])
        if(self.tensor_type == 'float'):
            self.net_ema.float()
        else:
            self.net_ema.double()

    def training(self):
        class_num   = self.config['network']['class_num']
        iter_valid  = self.config['training']['iter_valid']
        ssl_cfg     = self.config['semi_supervised_learning']
        iter_max     = self.config['training']['iter_max']
        rampup_start = ssl_cfg.get('rampup_start', 0)
        rampup_end   = ssl_cfg.get('rampup_end', iter_max)
        train_loss  = 0
        train_loss_sup = 0
        train_loss_reg = 0
        train_dice_list = []
        self.net.train()
        self.net_ema.to(self.device)
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
            noise = torch.clamp(torch.randn_like(x1) * 0.1, -0.2, 0.2)
            inputs_ema = x1 + torch.clamp(torch.randn_like(x1) * 0.1, -0.2, 0.2)
            inputs_ema = inputs_ema.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            outputs = self.net(inputs)
            n0 = list(x0.shape)[0] 
            p0 = outputs[:n0]
            loss_sup = self.get_loss_value(data_lab, p0, y0)

            outputs_soft = torch.softmax(outputs, dim=1)
            p1_soft = outputs_soft[n0:]
            
            with torch.no_grad():
                outputs_ema = self.net_ema(inputs_ema)
                p1_ema_soft = torch.softmax(outputs_ema, dim=1)
            
            rampup_ratio = get_rampup_ratio(self.glob_it, rampup_start, rampup_end, "sigmoid")
            regular_w = ssl_cfg.get('regularize_w', 0.1) * rampup_ratio

            loss_reg = torch.nn.MSELoss()(p1_soft, p1_ema_soft)
            loss = loss_sup + regular_w*loss_reg

            loss.backward()
            self.optimizer.step()

            # update EMA
            alpha = ssl_cfg.get('ema_decay', 0.99)
            alpha = min(1 - 1 / (self.glob_it / iter_valid + 1), alpha)
            for ema_param, param in zip(self.net_ema.parameters(), self.net.parameters()):
                ema_param.data.mul_(alpha).add(param.data, alpha = 1.0 - alpha)

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