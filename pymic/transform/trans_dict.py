# -*- coding: utf-8 -*-
from __future__ import print_function, division
from pymic.transform.gamma_correction import  ChannelWiseGammaCorrection
from pymic.transform.gray2rgb import GrayscaleToRGB
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
    'ChannelWiseThreshold': ChannelWiseThreshold,
    'ChannelWiseThresholdWithNormalize': ChannelWiseThresholdWithNormalize,
    'CropWithBoundingBox': CropWithBoundingBox,
    'CenterCrop': CenterCrop,
    'GrayscaleToRGB': GrayscaleToRGB,
    'LabelConvert': LabelConvert,
    'LabelConvertNonzero': LabelConvertNonzero,
    'LabelToProbability': LabelToProbability,
    'NormalizeWithMeanStd': NormalizeWithMeanStd,
    'NormalizeWithMinMax': NormalizeWithMinMax,
    'NormalizeWithPercentiles': NormalizeWithPercentiles,
    'RandomCrop': RandomCrop,
    'RandomResizedCrop':RandomResizedCrop,
    'RandomFlip': RandomFlip,
    'RandomRotate': RandomRotate,
    'ReduceLabelDim': ReduceLabelDim,
    'Rescale': Rescale,
    'Pad': Pad,
}
