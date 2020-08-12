# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.loss.util import reshape_tensor_to_2D

class CrossEntropyLoss(nn.Module):
    def __init__(self, params):
        super(CrossEntropyLoss, self).__init__()
        self.enable_pix_weight = params['CrossEntropyLoss_Enable_Pixel_Weight'.lower()]
        self.enable_cls_weight = params['CrossEntropyLoss_Enable_Class_Weight'.lower()]
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        pix_w   = loss_input_dict['pixel_weight']
        cls_w   = loss_input_dict['class_weight']
        softmax = loss_input_dict['softmax']

        if(softmax):
            predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y)

        ce = - soft_y* torch.log(predict)
        if(self.enable_cls_weight):
            if(cls_w is None):
                raise ValueError("Class weight is enabled but not defined")
            ce = torch.sum(ce * cls_w, dim = 1)
        else:
            ce = torch.sum(ce, dim = 1)
        if(self.enable_pix_weight):
            if(pix_w is None):
                raise ValueError("Pixel weight is enabled but not defined")
            pix_w = reshape_tensor_to_2D(pix_w)
            ce    = torch.sum(ce * pix_w) / torch.sum(pix_w)
        else:
            ce = torch.mean(ce)  
        return ce