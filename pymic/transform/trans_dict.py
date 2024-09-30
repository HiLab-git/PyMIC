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
    'IntensityClip': IntensityClip,
    'NonLinearTransform': NonLinearTransform,
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
    'SelfSuperviseLabel': SelfSuperviseLabel,
    'Pad': Pad.

"""
from __future__ import print_function, division
from pymic.transform.affine import *
from pymic.transform.intensity import  *
from pymic.transform.flip import *
from pymic.transform.pad import *
from pymic.transform.rotate import *
from pymic.transform.rescale import *
from pymic.transform.transpose import *
from pymic.transform.threshold import * 
from pymic.transform.normalize import *
from pymic.transform.crop import *
from pymic.transform.crop4dino import Crop4Dino
from pymic.transform.crop4vox2vec import Crop4Vox2Vec
from pymic.transform.crop4vf import Crop4VolumeFusion, VolumeFusion, VolumeFusionShuffle
from pymic.transform.volume_fusion import *
from pymic.transform.label_convert import *  

TransformDict = {
    'Affine': Affine,
    'AdaptiveContrastAdjust': AdaptiveContrastAdjust,
    'ChannelWiseThreshold': ChannelWiseThreshold,
    'ChannelWiseThresholdWithNormalize': ChannelWiseThresholdWithNormalize,
    'CropWithBoundingBox': CropWithBoundingBox,
    'CropWithForeground': CropWithForeground,
    'CropHumanRegion': CropHumanRegion,
    'CenterCrop': CenterCrop,
    'Crop4Dino': Crop4Dino,
    'Crop4Vox2Vec': Crop4Vox2Vec,
    'Crop4VolumeFusion': Crop4VolumeFusion,
    'GrayscaleToRGB': GrayscaleToRGB,
    'GammaCorrection': GammaCorrection,
    'GaussianNoise': GaussianNoise,
    'HistEqual': HistEqual, 
    'InPainting': InPainting,
    'InOutPainting': InOutPainting,
    'LabelConvert': LabelConvert,
    'LabelConvertNonzero': LabelConvertNonzero,
    'LabelToProbability': LabelToProbability,
    'LocalShuffling': LocalShuffling,
    'IntensityClip': IntensityClip,
    'MaskedImageModeling': MaskedImageModeling,
    'NonLinearTransform': NonLinearTransform,
    'NormalizeWithMeanStd': NormalizeWithMeanStd,
    'NormalizeWithMinMax': NormalizeWithMinMax,
    'NormalizeWithPercentiles': NormalizeWithPercentiles,
    'PartialLabelToProbability':PartialLabelToProbability,
    'RandomCrop': RandomCrop,
    'RandomSlice': RandomSlice,
    'RandomResizedCrop': RandomResizedCrop,
    'RandomRescale': RandomRescale,
    'RandomTranspose': RandomTranspose,
    'RandomFlip': RandomFlip,
    'RandomRotate': RandomRotate,
    'ReduceLabelDim': ReduceLabelDim,
    'Rescale': Rescale,
    'Resample': Resample,
    'SelfReconstructionLabel': SelfReconstructionLabel,
    'MaskedImageModelingLabel': MaskedImageModelingLabel,
    'OutPainting': OutPainting,
    'Pad': Pad,
    'PatchSwaping':PatchSwaping,
    'VolumeFusion': VolumeFusion,
    'VolumeFusionShuffle': VolumeFusionShuffle
}
