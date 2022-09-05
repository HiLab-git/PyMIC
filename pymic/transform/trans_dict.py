# -*- coding: utf-8 -*-
from __future__ import print_function, division
from pymic.transform.intensity import  *
from pymic.transform.flip import RandomFlip
from pymic.transform.pad import Pad
from pymic.transform.rotate import RandomRotate
from pymic.transform.rescale import Rescale, RandomRescale  
from pymic.transform.threshold import * 
from pymic.transform.normalize import *
from pymic.transform.crop import *
from pymic.transform.label_convert import * 

TransformDict = {
    'ChannelWiseThreshold': ChannelWiseThreshold,
    'ChannelWiseThresholdWithNormalize': ChannelWiseThresholdWithNormalize,
    'CropWithBoundingBox': CropWithBoundingBox,
    'CenterCrop': CenterCrop,
    'GrayscaleToRGB': GrayscaleToRGB,
    'GammaCorrection': GammaCorrection,
    'GaussianNoise': GaussianNoise,
    'LabelConvert': LabelConvert,
    'LabelConvertNonzero': LabelConvertNonzero,
    'LabelToProbability': LabelToProbability,
    'NormalizeWithMeanStd': NormalizeWithMeanStd,
    'NormalizeWithMinMax': NormalizeWithMinMax,
    'NormalizeWithPercentiles': NormalizeWithPercentiles,
    'PartialLabelToProbability':PartialLabelToProbability,
    'RandomCrop': RandomCrop,
    'RandomResizedCrop': RandomResizedCrop,
    'RandomRescale': RandomRescale,
    'RandomFlip': RandomFlip,
    'RandomRotate': RandomRotate,
    'ReduceLabelDim': ReduceLabelDim,
    'Rescale': Rescale,
    'Pad': Pad,
}
