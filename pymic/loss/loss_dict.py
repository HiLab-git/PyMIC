# -*- coding: utf-8 -*-
from __future__ import print_function, division
from pymic.loss.ce import CrossEntropyLoss, GeneralizedCrossEntropyLoss
from pymic.loss.dice import DiceLoss, MultiScaleDiceLoss
from pymic.loss.dice import DiceWithCrossEntropyLoss, NoiseRobustDiceLoss
from pymic.loss.exp_log import ExpLogLoss
from pymic.loss.mse import MSELoss, MAELoss

LossDict = {'CrossEntropyLoss': CrossEntropyLoss,
    'GeneralizedCrossEntropyLoss': GeneralizedCrossEntropyLoss,
    'DiceLoss': DiceLoss,
    'MultiScaleDiceLoss': MultiScaleDiceLoss,
    'DiceWithCrossEntropyLoss': DiceWithCrossEntropyLoss,
    'NoiseRobustDiceLoss': NoiseRobustDiceLoss,
    'ExpLogLoss': ExpLogLoss,
    'MSELoss': MSELoss,
    'MAELoss': MAELoss}
