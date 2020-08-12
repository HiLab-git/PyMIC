# -*- coding: utf-8 -*-
from __future__ import print_function, division
from pymic.loss.ce import CrossEntropyLoss
from pymic.loss.dice import DiceLoss, MultiScaleDiceLoss

loss_dict = {'CrossEntropyLoss': CrossEntropyLoss,
    'DiceLoss': DiceLoss,
    'MultiScaleDiceLoss': MultiScaleDiceLoss}

def get_loss(params):
    loss_type = params['loss_type']
    if(loss_type in loss_dict):
        loss_obj = loss_dict[loss_type](params)
    else:
        raise ValueError("Undefined loss type {0:}".format(loss_type))
    return loss_obj