# -*- coding: utf-8 -*-
from __future__ import print_function, division

import copy
import os
import sys
import time
import random
import scipy
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from scipy import special
from datetime import datetime
from tensorboardX import SummaryWriter
from pymic.io.image_read_write import save_nd_array_as_image
from pymic.io.nifty_dataset import NiftyDataset
from pymic.transform.trans_dict import TransformDict
from pymic.net.net_dict_seg import SegNetDict
from pymic.net_run.agent_abstract import NetRunAgent
from pymic.net_run.agent_seg import SegmentationAgent
from pymic.net_run.infer_func import Inferer
from pymic.loss.loss_dict_seg import SegLossDict
from pymic.loss.seg.combined import CombinedLoss
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from pymic.util.image_process import convert_label
from pymic.util.parse_config import parse_config
from pymic.loss.seg.ssl import EntropyLoss
from pymic.net_run_ssl.ssl_em import SSLSegAgent


class SSLMeanTeacher(SSLSegAgent):
    """
    Training and testing agent for semi-supervised segmentation
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
        train_loss  = 0
        train_loss_sup = 0
        train_loss_unsup = 0
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
            
            # # for debug
            # for i in range(inputs.shape[0]):
            #     image_i = inputs[i][0]
            #     label_i = labels_prob[i][1]
            #     pixw_i  = pix_w[i][0]
            #     print(image_i.shape, label_i.shape, pixw_i.shape)
            #     image_name = "temp/image_{0:}_{1:}.nii.gz".format(it, i)
            #     label_name = "temp/label_{0:}_{1:}.nii.gz".format(it, i)
            #     weight_name= "temp/weight_{0:}_{1:}.nii.gz".format(it, i)
            #     save_nd_array_as_image(image_i, image_name, reference_name = None)
            #     save_nd_array_as_image(label_i, label_name, reference_name = None)
            #     save_nd_array_as_image(pixw_i, weight_name, reference_name = None)
            # continue

            inputs, y0 = inputs.to(self.device), y0.to(self.device)
            noise = torch.clamp(torch.randn_like(x1) * 0.1, -0.2, 0.2)
            inputs_ema = x1 + noise
            inputs_ema = inputs_ema.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            outputs = self.net(inputs)
            n0 = list(x0.shape)[0] 
            p0, p1  = torch.tensor_split(outputs, [n0,], dim = 0)
            outputs_soft = torch.softmax(outputs, dim=1)
            p0_soft, p1_soft = torch.tensor_split(outputs_soft, [n0,], dim = 0)
            with torch.no_grad():
                outputs_ema = self.net_ema(inputs_ema)
                p1_ema_soft = torch.softmax(outputs_ema, dim=1)
                
            loss_sup = self.get_loss_value(data_lab, x0, p0, y0)
            consis_w0= ssl_cfg.get('consis_w', 0.1)
            iter_sup = ssl_cfg.get('iter_sup', 0)
            iter_max = self.config['training']['iter_max']
            ramp_up_length = ssl_cfg.get('ramp_up_length', iter_max)
            consis_w = self.get_consistency_weight_with_rampup(
                consis_w0, self.glob_it, ramp_up_length)
            consis_w = 0.0 if self.glob_it < iter_sup else consis_w
            loss_unsup = torch.nn.MSELoss()(p1_soft, p1_ema_soft)
            loss = loss_sup + consis_w*loss_unsup

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # update EMA
            alpha = ssl_cfg.get('ema_decay', 0.99)
            alpha = min(1 - 1 / (iter_max + 1), alpha)
            for ema_param, param in zip(self.net_ema.parameters(), self.net.parameters()):
                ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

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
           
def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('   pymic_net_run train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)
    agent = SSLMeanTeacher(config, stage)
    agent.run()

if __name__ == "__main__":
    main()