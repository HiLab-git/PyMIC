# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.loss.util import reshape_tensor_to_2D, get_classwise_dice

class DiceLoss(nn.Module):
    def __init__(self, params):
        super(DiceLoss, self).__init__()
        self.enable_pix_weight = params['DiceLoss_Enable_Pixel_Weight'.lower()]
        self.enable_cls_weight = params['DiceLoss_Enable_Class_Weight'.lower()]

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
        if(self.enable_cls_weight):
            if(cls_w is None):
                raise ValueError("Class weight is enabled but not defined")
            weighted_dice = dice_score * cls_w
            avg_dice =  weighted_dice.sum() / cls_w.sum()
        else:
            avg_dice = torch.mean(dice_score)   
        dice_loss  = 1.0 - avg_dice
        return dice_loss

class MultiScaleDiceLoss(nn.Module):
    def __init__(self, params):
        super(MultiScaleDiceLoss, self).__init__()
        self.enable_pix_weight  = params['MultiScaleDiceLoss_Enable_Pixel_Weight'.lower()]
        self.enable_cls_weight  = params['MultiScaleDiceLoss_Enable_Class_Weight'.lower()]
        self.multi_scale_weight = params['MultiScaleDiceLoss_Scale_Weight'.lower()]
        dice_params = {'DiceLoss_Enable_Pixel_Weight'.lower(): self.enable_pix_weight,
            'DiceLoss_Enable_Class_Weight'.lower(): self.enable_cls_weight}
        self.base_loss = DiceLoss(dice_params)

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        pix_w   = loss_input_dict['pixel_weight']
        cls_w   = loss_input_dict['class_weight']
        softmax = loss_input_dict['softmax']
        if(isinstance(predict, tuple) or isinstance(predict, list)):
            predict_num = len(predict)
            assert(predict_num == len(self.multi_scale_weight))
            loss   = 0.0
            weight = 0.0
            interp_mode = 'trilinear' if(len(predict[0].shape) == 5) else 'bilinear'
            for i in range(predict_num):
                soft_y_temp = nn.functional.interpolate(soft_y, 
                    size = list(predict[i].shape)[2:], mode = interp_mode)
                if(pix_w is not None):
                    pix_w_temp  = nn.functional.interpolate(pix_w, 
                        size = list(predict[i].shape)[2:], mode = interp_mode)
                else:
                    pix_w_temp = None
                temp_loss_dict = {}
                temp_loss_dict['prediction'] = predict[i]
                temp_loss_dict['ground_truth'] = soft_y_temp
                temp_loss_dict['pixel_weight'] = pix_w_temp
                temp_loss_dict['class_weight'] = cls_w
                temp_loss_dict['softmax'] = softmax
                temp_loss = self.base_loss(temp_loss_dict)
                loss      = loss + temp_loss * self.multi_scale_weight[i]
                weight    = weight + self.multi_scale_weight[i]
            loss = loss/weight
        else:
            loss = self.base_loss(loss_input_dict)
        return loss