# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import numpy as np
import random
import torch
import torch.nn.functional as F
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.net.net_dict_seg import SegNetDict
from pymic.net_run.weak_sup import WSLSegAgent
from pymic.util.ramps import get_rampup_ratio
from pymic.util.general import keyword_match

class WSLUSTM(WSLSegAgent):
    """
    USTM for scribble-supervised segmentation.

    * Reference: Xiaoming Liu, Quan Yuan, Yaozong Gao, Helei He, Shuo Wang, 
      Xiao Tang, Jinshan Tang, Dinggang Shen: Weakly Supervised Segmentation 
      of COVID19 Infection with Scribble Annotation on CT Images.
      `Patter Recognition <https://doi.org/10.1016/j.patcog.2021.108341>`_, 2022.
        
    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `weakly_supervised_learning` is needed. See :doc:`usage.wsl` for details.
    """
    def __init__(self, config, stage = 'train'):
        super(WSLUSTM, self).__init__(config, stage)
        self.net_ema = None 
    
    def create_network(self):
        super(WSLUSTM, self).create_network()
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
        wsl_cfg     = self.config['weakly_supervised_learning']
        iter_max     = self.config['training']['iter_max']
        rampup_start = wsl_cfg.get('rampup_start', 0)
        rampup_end   = wsl_cfg.get('rampup_end', iter_max)
        train_loss  = 0
        train_loss_sup = 0
        train_loss_reg = 0
        train_dice_list = []
        self.net.train()
        self.net_ema.to(self.device)
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
            noise   = torch.clamp(torch.randn_like(inputs) * 0.1, -0.2, 0.2)
            outputs = self.net(inputs + noise)
            out_prob= F.softmax(outputs, dim=1)
            loss_sup = self.get_loss_value(data, outputs, y)

            rot_times = random.randrange(0, 4)
            inputs_rot= torch.rot90(inputs, rot_times, [-2, -1])
            noise = torch.clamp(torch.randn_like(inputs_rot) * 0.1, -0.2, 0.2)
            with torch.no_grad():
                ema_inputs  = inputs_rot + noise
                ema_outputs = self.net_ema(ema_inputs)
                ema_out_prob= F.softmax(ema_outputs, dim=1)
            out_prob_rot = torch.rot90(out_prob, rot_times, [-2, -1])
            square_error = torch.square(out_prob_rot - ema_out_prob)

            # the forward pass number for uncertainty estimation
            T = wsl_cfg.get("ustm_mcdroput_n", 8)
            preds = torch.zeros([T] + list(y.shape)).to(self.device)
            for i in range(T//2):
                ema_inputs_r = torch.cat([inputs_rot, inputs_rot], dim = 0)
                ema_inputs_r = ema_inputs_r + \
                    torch.clamp(torch.randn_like(ema_inputs_r) * 0.1, -0.2, 0.2)
                ema_inputs_r = ema_inputs_r.to(self.device)
                with torch.no_grad():
                    ema_outputs_r = self.net_ema(ema_inputs_r)
                # reshape from [2B, C, D, H, W] to [2, B, C, D, H, W]
                preds[2*i:2*(i+1)] = ema_outputs_r.reshape([2]+list(y.shape))
            preds = torch.softmax(preds, dim = 2)
            preds = torch.mean(preds, dim = 0)
            uncertainty = -1.0 * torch.sum(preds*torch.log(preds + 1e-6),
                 dim=1, keepdim=True)
            
            rampup_ratio = get_rampup_ratio(self.glob_it, rampup_start, rampup_end, "sigmoid")
            class_num = list(y.shape)[1]
            threshold = (0.75+0.25*rampup_ratio)*np.log(class_num)
            mask      = (uncertainty < threshold).float()
            loss_reg  = torch.sum(mask*square_error)/(2*torch.sum(mask)+1e-16)

            regular_w = wsl_cfg.get('regularize_w', 0.1) * rampup_ratio
            loss = loss_sup + regular_w*loss_reg

            loss.backward()
            self.optimizer.step()

            # update EMA
            alpha = wsl_cfg.get('ema_decay', 0.99)
            alpha = min(1 - 1 / (self.glob_it / iter_valid + 1), alpha)
            for ema_param, param in zip(self.net_ema.parameters(), self.net.parameters()):
                ema_param.data.mul_(alpha).add(param.data, alpha = 1.0 - alpha)

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
        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup = train_loss_sup / iter_valid
        train_avg_loss_reg = train_loss_reg / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis = 0)
        train_avg_dice = train_cls_dice[1:].mean()

        train_scalers = {'loss': train_avg_loss, 'loss_sup':train_avg_loss_sup,
            'loss_reg':train_avg_loss_reg, 'regular_w':regular_w,
            'avg_fg_dice':train_avg_dice,     'class_dice': train_cls_dice}
        return train_scalers