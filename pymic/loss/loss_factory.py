# -*- coding: utf-8 -*-
from __future__ import print_function, division
from pymic.loss.ce import CrossEntropyLoss, GeneralizedCrossEntropyLoss
from pymic.loss.dice import DiceLoss, MultiScaleDiceLoss
from pymic.loss.dice import DiceWithCrossEntropyLoss, NoiseRobustDiceLoss
from pymic.loss.exp_log import ExpLogLoss

loss_dict = {'CrossEntropyLoss': CrossEntropyLoss,
    'GeneralizedCrossEntropyLoss': GeneralizedCrossEntropyLoss,
    'DiceLoss': DiceLoss,
    'MultiScaleDiceLoss': MultiScaleDiceLoss,
    'DiceWithCrossEntropyLoss': DiceWithCrossEntropyLoss,
    'NoiseRobustDiceLoss': NoiseRobustDiceLoss,
    'ExpLogLoss': ExpLogLoss}

def get_loss(params):
    loss_type = params['loss_type']
    if(loss_type in loss_dict):
        loss_obj = loss_dict[loss_type](params)
    else:
        raise ValueError("Undefined loss type {0:}".format(loss_type))
    return loss_obj