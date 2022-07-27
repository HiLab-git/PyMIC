# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch.nn as nn 
from pymic.loss.seg.ce import CrossEntropyLoss, GeneralizedCrossEntropyLoss
from pymic.loss.seg.dice import DiceLoss, FocalDiceLoss, NoiseRobustDiceLoss
from pymic.loss.seg.slsr import SLSRLoss
from pymic.loss.seg.exp_log import ExpLogLoss
from pymic.loss.seg.mse import MSELoss, MAELoss

SegLossDict = {'CrossEntropyLoss': CrossEntropyLoss,
    'GeneralizedCrossEntropyLoss': GeneralizedCrossEntropyLoss,
    'SLSRLoss': SLSRLoss,
    'DiceLoss': DiceLoss,
    'FocalDiceLoss': FocalDiceLoss,
    'NoiseRobustDiceLoss': NoiseRobustDiceLoss,
    'ExpLogLoss': ExpLogLoss,
    'MSELoss': MSELoss,
    'MAELoss': MAELoss}

