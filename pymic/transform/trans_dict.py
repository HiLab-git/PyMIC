# -*- coding: utf-8 -*-
"""
The built-in transforms in PyMIC are:

.. code-block:: none

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
    'Pad': Pad.

"""
from __future__ import print_function, division
from pymic.transform.intensity import  *
from pymic.transform.flip import *
from pymic.transform.pad import *
from pymic.transform.rotate import *
from pymic.transform.rescale import *
from pymic.transform.transpose import *
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
    'InPainting': InPainting,
    'InOutPainting': InOutPainting,
    'LabelConvert': LabelConvert,
    'LabelConvertNonzero': LabelConvertNonzero,
    'LabelToProbability': LabelToProbability,
    'LocalShuffling': LocalShuffling,
    'NonLinearTransform': NonLinearTransform,
    'NormalizeWithMeanStd': NormalizeWithMeanStd,
    'NormalizeWithMinMax': NormalizeWithMinMax,
    'NormalizeWithPercentiles': NormalizeWithPercentiles,
    'PartialLabelToProbability':PartialLabelToProbability,
    'RandomCrop': RandomCrop,
    'RandomResizedCrop': RandomResizedCrop,
    'RandomRescale': RandomRescale,
    'RandomTranspose': RandomTranspose,
    'RandomFlip': RandomFlip,
    'RandomRotate': RandomRotate,
    'ReduceLabelDim': ReduceLabelDim,
    'Rescale': Rescale,
    'OutPainting': OutPainting,
    'Pad': Pad,
}
