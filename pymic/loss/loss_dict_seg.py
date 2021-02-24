# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch.nn as nn 
from pymic.loss.seg.ce import CrossEntropyLoss, GeneralizedCrossEntropyLoss
from pymic.loss.seg.dice import DiceLoss, MultiScaleDiceLoss
from pymic.loss.seg.dice import DiceWithCrossEntropyLoss, NoiseRobustDiceLoss
from pymic.loss.seg.exp_log import ExpLogLoss
from pymic.loss.seg.mse import MSELoss, MAELoss

SegLossDict = {'CrossEntropyLoss': CrossEntropyLoss,
    'GeneralizedCrossEntropyLoss': GeneralizedCrossEntropyLoss,
    'DiceLoss': DiceLoss,
    'MultiScaleDiceLoss': MultiScaleDiceLoss,
    'DiceWithCrossEntropyLoss': DiceWithCrossEntropyLoss,
    'NoiseRobustDiceLoss': NoiseRobustDiceLoss,
    'ExpLogLoss': ExpLogLoss,
    'MSELoss': MSELoss,
    'MAELoss': MAELoss}

