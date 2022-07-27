# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.loss.seg.util import reshape_tensor_to_2D

class CrossEntropyLoss(nn.Module):
    def __init__(self, params):
        super(CrossEntropyLoss, self).__init__()
        if(params is None):
            self.softmax = True
        else:
            self.softmax = params.get('loss_softmax', True)
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        pix_w   = loss_input_dict.get('pixel_weight', None)

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y)

        # for numeric stability
        predict = predict * 0.999 + 5e-4
        ce = - soft_y* torch.log(predict)
        ce = torch.sum(ce, dim = 1) # shape is [N]
        if(pix_w is None):
            ce = torch.mean(ce)  
        else:
            pix_w = torch.squeeze(reshape_tensor_to_2D(pix_w))
            ce = torch.sum(pix_w * ce) / (pix_w.sum() + 1e-5) 
        return ce

class PartialCrossEntropyLoss(nn.Module):
    def __init__(self, params):
        super(CrossEntropyLoss, self).__init__()
        self.softmax = params.get('loss_softmax', True)
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y)

        # for numeric stability
        predict = predict * 0.999 + 5e-4
        ce = - soft_y* torch.log(predict)
        ce = torch.sum(ce, dim = 1) # shape is [N]
        ce = torch.mean(ce)  
        return ce

class GeneralizedCrossEntropyLoss(nn.Module):
    """
    Generalized cross entropy loss to deal with noisy labels. 
        Z. Zhang et al. Generalized Cross Entropy Loss for Training Deep Neural Networks 
        with Noisy Labels, NeurIPS 2018.
    """
    def __init__(self, params):
        super(GeneralizedCrossEntropyLoss, self).__init__()
        self.enable_pix_weight = params['GeneralizedCrossEntropyLoss_Enable_Pixel_Weight'.lower()]
        self.enable_cls_weight = params['GeneralizedCrossEntropyLoss_Enable_Class_Weight'.lower()]
        self.q = params['GeneralizedCrossEntropyLoss_q'.lower()]

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        pix_w   = loss_input_dict['pixel_weight']
        cls_w   = loss_input_dict['class_weight']
        softmax = loss_input_dict['softmax']

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(softmax):
            predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y)
        gce     = (1.0 - torch.pow(predict, self.q)) / self.q * soft_y
        
        if(self.enable_cls_weight):
            if(cls_w is None):
                raise ValueError("Class weight is enabled but not defined")
            gce = torch.sum(gce * cls_w, dim = 1)
        else:
            gce = torch.sum(gce, dim = 1)
        
        if(self.enable_pix_weight):
            if(pix_w is None):
                raise ValueError("Pixel weight is enabled but not defined")
            pix_w = reshape_tensor_to_2D(pix_w)
            gce    = torch.sum(gce * pix_w) / torch.sum(pix_w)
        else:
            gce = torch.mean(gce)
        return gce 
