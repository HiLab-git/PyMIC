# -*- coding: utf-8 -*-
from __future__ import print_function, division
from pymic.transform.transform3d import * 

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
