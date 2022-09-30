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
        
        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y) 
        dice_score = get_classwise_dice(predict, soft_y)
        dice_loss  = 1.0 - dice_score.mean()
        return dice_loss

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
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
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
