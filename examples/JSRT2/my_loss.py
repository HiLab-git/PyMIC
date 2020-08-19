# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np 
from pymic.loss.util import reshape_tensor_to_2D, get_classwise_dice

class MyFocalDiceLoss(nn.Module):
    """
    Focal Dice loss proposed in the following paper:
       P. Wang et al. Focal dice loss and image dilatin for brain tumor segmentation.
       in Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical
       Decision Support, 2018.
    """
    def __init__(self, params):
        super(MyFocalDiceLoss, self).__init__()
        self.enable_pix_weight  = params['MyFocalDiceLoss_Enable_Pixel_Weight'.lower()]
        self.enable_cls_weight  = params['MyFocalDiceLoss_Enable_Class_Weight'.lower()]
        self.beta = params['MyFocalDiceLoss_beta'.lower()]
        assert(self.beta >= 1.0)

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

        if(self.enable_pix_weight):
            if(pix_w is None):
                raise ValueError("Pixel weight is enabled but not defined")
            pix_w = reshape_tensor_to_2D(pix_w)
        dice_score = get_classwise_dice(predict, soft_y, pix_w)
        dice_score = 0.01 + dice_score * 0.98
        dice_loss  = 1.0 - torch.pow(dice_score, 1.0 / self.beta)

        if(self.enable_cls_weight):
            if(cls_w is None):
                raise ValueError("Class weight is enabled but not defined")
            weighted_loss = dice_loss * cls_w
            avg_loss =  weighted_loss.sum() / cls_w.sum()
        else:
            avg_loss = torch.mean(dice_loss)   
        return avg_loss
