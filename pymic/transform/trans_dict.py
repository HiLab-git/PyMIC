# -*- coding: utf-8 -*-
from __future__ import print_function, division
from pymic.transform.gamma_correction import  ChannelWiseGammaCorrection
from pymic.transform.flip import RandomFlip
from pymic.transform.pad import Pad
from pymic.transform.rotate import RandomRotate
from pymic.transform.rescale import Rescale  
from pymic.transform.threshold import * 
from pymic.transform.normalize import *
from pymic.transform.crop import *
from pymic.transform.label_convert import * 

TransformDict = {
    'ChannelWiseGammaCorrection': ChannelWiseGammaCorrection,
    'ChannelWiseNormalize': ChannelWiseNormalize,
    'ChannelWiseThreshold': ChannelWiseThreshold,
    'ChannelWiseThresholdWithNormalize': ChannelWiseThresholdWithNormalize,
    'CropWithBoundingBox': CropWithBoundingBox,
    'LabelConvert': LabelConvert,
    'LabelConvertNonzero': LabelConvertNonzero,
    'LabelToProbability': LabelToProbability,
    'RandomCrop': RandomCrop,
    'RandomFlip': RandomFlip,
    'RandomRotate': RandomRotate,
    'ReduceLabelDim': ReduceLabelDim,
    'Rescale': Rescale,
    'Pad': Pad,
}
