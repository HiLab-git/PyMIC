# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.loss.cls.util import get_soft_label

class L1Loss(nn.Module):
    def __init__(self, params):
        super(L1Loss, self).__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        labels  = loss_input_dict['ground_truth'][:, None] # reshape to N, 1
        softmax = nn.Softmax(dim = 1)
        predict = softmax(predict)
        num_class  = list(predict.size())[1]
        data_type = 'float' if(predict.dtype is torch.float32) else 'double'
        soft_y = get_soft_label(labels, num_class, data_type)
        loss = self.l1_loss(predict, soft_y)
        return loss

class RectifiedLoss(nn.Module):
    def __init__(self, params):
        super(RectifiedLoss, self).__init__()
        # self.l1_loss = nn.L1Loss()
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        labels  = loss_input_dict['ground_truth'][:, None] # reshape to N, 1
        
        # softmax = nn.Softmax(dim = 1)
        # predict = softmax(predict)
        num_class  = list(predict.size())[1]
        data_type = 'float' if(predict.dtype is torch.float32) else 'double'
        soft_y = get_soft_label(labels, num_class, data_type)
        g = 2* soft_y - 1
        loss = torch.exp((g*1.5- predict) * g)
        mask = predict < g 
        if (data_type  == 'float'):
            mask = mask.float() 
        else:
            mask = mask.double() 
        w = (mask - 0.5) * g + 0.5    
        loss = w * loss + 0.1*(g - predict) * (g - predict)
        loss = loss.mean()
        # loss = self.l1_loss(predict, soft_y)
        return loss