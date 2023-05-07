# -*- coding: utf-8 -*-
"""
Built-in loss functions for segmentation. 
The following are for fully supervised learning, or learnig from noisy labels:

* CrossEntropyLoss :mod:`pymic.loss.seg.ce.CrossEntropyLoss`
* GeneralizedCELoss :mod:`pymic.loss.seg.ce.GeneralizedCELoss`
* DiceLoss :mod:`pymic.loss.seg.dice.DiceLoss`
* FocalDiceLoss :mod:`pymic.loss.seg.dice.FocalDiceLoss`
* NoiseRobustDiceLoss :mod:`pymic.loss.seg.dice.NoiseRobustDiceLoss`
* ExpLogLoss :mod:`pymic.loss.seg.exp_log.ExpLogLoss`
* MAELoss  :mod:`pymic.loss.seg.mse.MAELoss`
* MSELoss  :mod:`pymic.loss.seg.mse.MSELoss`
* SLSRLoss :mod:`pymic.loss.seg.slsr.SLSRLoss`

The following are for semi-supervised or weakly supervised learning:

* EntropyLoss :mod:`pymic.loss.seg.ssl.EntropyLoss`
* GatedCRFLoss: :mod:`pymic.loss.seg.gatedcrf.GatedCRFLoss`
* MumfordShahLoss  :mod:`pymic.loss.seg.mumford_shah.MumfordShahLoss`
* TotalVariationLoss :mod:`pymic.loss.seg.ssl.TotalVariationLoss`
"""
from __future__ import print_function, division
import torch.nn as nn 
from pymic.loss.seg.ce import CrossEntropyLoss, GeneralizedCELoss
from pymic.loss.seg.dice import DiceLoss, FocalDiceLoss, \
     NoiseRobustDiceLoss, BinaryDiceLoss, GroupDiceLoss
from pymic.loss.seg.exp_log import ExpLogLoss
from pymic.loss.seg.mse import MSELoss, MAELoss
from pymic.loss.seg.slsr import SLSRLoss

SegLossDict = {
    'CrossEntropyLoss': CrossEntropyLoss,
    'GeneralizedCELoss': GeneralizedCELoss,
    'DiceLoss': DiceLoss,
    'BinaryDiceLoss': BinaryDiceLoss,
    'FocalDiceLoss': FocalDiceLoss,
    'NoiseRobustDiceLoss': NoiseRobustDiceLoss,
    'GroupDiceLoss': GroupDiceLoss,
    'ExpLogLoss': ExpLogLoss,
    'MAELoss': MAELoss,
    'MSELoss': MSELoss,
    'SLSRLoss': SLSRLoss
    }

