
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.loss.seg.abstract import AbstractSegLoss
from pymic.loss.seg.util import reshape_tensor_to_2D, get_classwise_dice

class ExpLogLoss(AbstractSegLoss):
    """
    The exponential logarithmic loss in this paper: 
        
    * K. Wong et al.: 3D Segmentation with Exponential Logarithmic Loss for Highly 
      Unbalanced Object Sizes. `MICCAI 2018. <https://arxiv.org/abs/1809.00076>`_

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `loss_softmax`: (bool) Apply softmax to the prediction of network or not.
    :param `ExpLogLoss_w_dice`:  (float) Weight of ExpLog Dice loss in the range of [0, 1].
    :param `ExpLogLoss_gamma`: (float) Hyper-parameter gamma. 
    """
    def __init__(self, params):
        super(ExpLogLoss, self).__init__(params)
        self.w_dice = params['ExpLogLoss_w_dice'.lower()]
        self.gamma  = params['ExpLogLoss_gamma'.lower()]

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
        dice_score = 0.005 + dice_score * 0.99
        exp_dice   = -torch.log(dice_score)
        exp_dice   = torch.pow(exp_dice, self.gamma)
        exp_dice   = torch.mean(exp_dice)

        predict= 0.005 + predict * 0.99
        wc     = torch.mean(soft_y, dim = 0)
        wc     = 1.0 / (wc + 0.1)
        wc     = torch.pow(wc, 0.5)
        ce     = - torch.log(predict)
        exp_ce = wc * torch.pow(ce, self.gamma)
        exp_ce = torch.sum(soft_y * exp_ce, dim = 1)
        exp_ce = torch.mean(exp_ce)

        loss = exp_dice * self.w_dice + exp_ce * (1.0 - self.w_dice)
        return loss