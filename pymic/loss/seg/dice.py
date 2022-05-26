# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.loss.seg.util import reshape_tensor_to_2D, get_classwise_dice

class DiceLoss(nn.Module):
    def __init__(self, params = None):
        super(DiceLoss, self).__init__()
        if(params is None):
            self.softmax = True
        else:
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
        dice_score = get_classwise_dice(predict, soft_y)
        dice_loss  = 1.0 - dice_score.mean()
        return dice_loss

class FocalDiceLoss(nn.Module):
    """
    focal Dice according to the following paper:
    Pei Wang and Albert C. S. Chung, Focal Dice Loss and Image Dilation for 
    Brain Tumor Segmentation, 2018
    """
    def __init__(self, params = None):
        super(FocalDiceLoss, self).__init__()
        self.softmax = params.get('loss_softmax', True)
        self.beta = params['FocalDiceLoss_beta'.lower()] #beta should be >=1.0

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y) 

        dice_score = get_classwise_dice(predict, soft_y)
        dice_score = torch.pow(dice_score, 1.0 / self.beta)
        dice_loss  = 1.0 - dice_score.mean()
        return dice_loss

class NoiseRobustDiceLoss(nn.Module):
    """
    Noise-robust Dice loss according to the following paper. 
        G. Wang et al. A Noise-Robust Framework for Automatic Segmentation of COVID-19 
        Pneumonia Lesions From CT Images, IEEE TMI, 2020. 
        https://doi.org/10.1109/TMI.2020.3000314
    """
    def __init__(self, params):
        super(NoiseRobustDiceLoss, self).__init__()
        self.softmax = params.get('loss_softmax', True)
        self.gamma = params['NoiseRobustDiceLoss_gamma'.lower()]

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
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
