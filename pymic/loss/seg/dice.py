# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.loss.seg.abstract import AbstractSegLoss
from pymic.loss.seg.util import reshape_tensor_to_2D, get_classwise_dice

class DiceLoss(AbstractSegLoss):
    '''
    Dice loss for segmentation tasks.
    The parameters should be written in the `params` dictionary, and it has the
    following fields:

    :param `loss_softmax`: (bool) Apply softmax to the prediction of network or not. 
    '''
    def __init__(self, params = None):
        super(DiceLoss, self).__init__(params)

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        pix_w   = loss_input_dict.get('pixel_weight', None)
        cls_w   = loss_input_dict.get('class_weight', None)
        
        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.acti_func is not None):
            predict = self.get_activated_prediction(predict, self.acti_func)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y) 
        if(pix_w is not None):
            pix_w = reshape_tensor_to_2D(pix_w) 
        dice_loss = 1.0 - get_classwise_dice(predict, soft_y, pix_w)
        if(cls_w is not None):
            weighted_loss = dice_loss * cls_w
            avg_loss = weighted_loss.sum() / cls_w.sum()
        else:
            avg_loss = dice_loss.mean()
        return avg_loss

class BinaryDiceLoss(AbstractSegLoss):
    '''
    Fuse all the foreground classes together and calculate the Dice value. 
    '''
    def __init__(self, params = None):
        super(BinaryDiceLoss, self).__init__(params)

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        
        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.acti_func is not None):
            predict = self.get_activated_prediction(predict, self.acti_func)
        predict = 1.0 - predict[:, :1, :, :, :]
        soft_y  = 1.0 -  soft_y[:, :1, :, :, :]
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y) 
        dice_score = get_classwise_dice(predict, soft_y)
        dice_loss  = 1.0 - dice_score.mean()
        return dice_loss

class GroupDiceLoss(AbstractSegLoss):
    '''
    Fuse all the foreground classes together and calculate the Dice value. 
    '''
    def __init__(self, params = None):
        super(GroupDiceLoss, self).__init__(params)
        self.group = 2

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        
        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.acti_func is not None):
            predict = self.get_activated_prediction(predict, self.acti_func)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y) 
        num_class  = list(predict.size())[1]
        cls_per_group = (num_class - 1) // self.group
        loss_all = 0.0
        for g in range(self.group):
            c0 = 1 + g*cls_per_group
            c1 = min(num_class, c0 + cls_per_group)
            pred_g = torch.sum(predict[:, c0:c1], dim = 1, keepdim = True)
            y_g    = torch.sum( soft_y[:, c0:c1], dim = 1, keepdim = True)
            loss_all += 1.0 - get_classwise_dice(pred_g, y_g)[0]
        avg_loss = loss_all / self.group
        return avg_loss

class FocalDiceLoss(AbstractSegLoss):
    """
    Focal Dice according to the following paper:

    * Pei Wang and Albert C. S. Chung, Focal Dice Loss and Image Dilation for 
      Brain Tumor Segmentation, 2018.

    The parameters should be written in the `params` dictionary, and it has the
    following fields:

    :param `loss_softmax`: (bool) Apply softmax to the prediction of network or not. 
    :param `FocalDiceLoss_beta`: (float) The hyper-parameter to set (>=1.0).
    """
    def __init__(self, params = None):
        super(FocalDiceLoss, self).__init__(params)
        self.beta = params['FocalDiceLoss_beta'.lower()] #beta should be >=1.0

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.acti_func is not None):
            predict = self.get_activated_prediction(predict, self.acti_func)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y) 

        dice_score = get_classwise_dice(predict, soft_y)
        dice_score = torch.pow(dice_score, 1.0 / self.beta)
        dice_loss  = 1.0 - dice_score.mean()
        return dice_loss

class NoiseRobustDiceLoss(AbstractSegLoss):
    """
    Noise-robust Dice loss according to the following paper. 
        
    * G. Wang et al. A Noise-Robust Framework for Automatic Segmentation of COVID-19 
      Pneumonia Lesions From CT Images, 
      `IEEE TMI <https://doi.org/10.1109/TMI.2020.3000314>`_, 2020.

    The parameters should be written in the `params` dictionary, and it has the
    following fields:

    :param `loss_softmax`: (bool) Apply softmax to the prediction of network or not. 
    :param `NoiseRobustDiceLoss_gamma`:  (float) The hyper-parameter gammar to set (1, 2).
    """
    def __init__(self, params):
        super(NoiseRobustDiceLoss, self).__init__(params)
        self.gamma = params['NoiseRobustDiceLoss_gamma'.lower()]

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.acti_func is not None):
            predict = self.get_activated_prediction(predict, self.acti_func)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y) 

        numerator = torch.abs(predict - soft_y)
        numerator = torch.pow(numerator, self.gamma)
        denominator = predict + soft_y 
        numer_sum = torch.sum(numerator,  dim = 0)
        denom_sum = torch.sum(denominator,  dim = 0)
        loss_vector = numer_sum / (denom_sum + 1e-5)
        loss = torch.mean(loss_vector)   
        return loss
