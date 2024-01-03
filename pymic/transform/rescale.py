# -*- coding: utf-8 -*-
from __future__ import print_function, division

import json
import random
import numpy as np
from scipy import ndimage
from pymic import TaskType
from pymic.transform.abstract_transform import AbstractTransform
from pymic.util.image_process import *


class Rescale(AbstractTransform):
    """Rescale the image to a given size.
    
    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `Rescale_output_size`: (list/tuple or int) The output size along each spatial axis, 
        such as [D, H, W] or [H, W].  If D is None, the input image is only reslcaled in 2D.
        If int, the smallest axis is matched to output_size keeping aspect ratio the same
        as the input.
    :param `Rescale_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `True`.
    """
    def __init__(self, params):
        super(Rescale, self).__init__(params)
        self.output_size = params["Rescale_output_size".lower()]
        self.inverse     = params.get("Rescale_inverse".lower(), True)
        assert isinstance(self.output_size, (int, list, tuple))

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim   = len(input_shape) - 1
        
        if isinstance(self.output_size, (list, tuple)):
            output_size = self.output_size
            if(output_size[0] is None):
                output_size[0] = input_shape[1]
            assert(len(output_size) == input_dim)
        else:
            min_edge = min(input_shape[1:])
            output_size = [self.output_size * input_shape[i+1] / min_edge for \
                            i in range(input_dim)]
        scale = [(output_size[i] + 0.0)/input_shape[1:][i] for i in range(input_dim)]
        scale = [1.0] + scale
        image_t = ndimage.interpolation.zoom(image, scale, order = 1)

        sample['image'] = image_t
        sample['Rescale_origin_shape'] = json.dumps(input_shape)
        if('label' in sample and \
        self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
            label = sample['label']
            label = ndimage.interpolation.zoom(label, scale, order = 0)
            sample['label'] = label
        if('pixel_weight' in sample and \
        self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
            weight = sample['pixel_weight']
            weight = ndimage.interpolation.zoom(weight, scale, order = 1)
            sample['pixel_weight'] = weight
        
        return sample

    def inverse_transform_for_prediction(self, sample):
        if(isinstance(sample['Rescale_origin_shape'], list) or \
            isinstance(sample['Rescale_origin_shape'], tuple)):
            origin_shape = json.loads(sample['Rescale_origin_shape'][0])
        else:
            origin_shape = json.loads(sample['Rescale_origin_shape'])
        origin_dim   = len(origin_shape) - 1
        predict = sample['predict']
        input_shape = predict.shape
        scale = [(origin_shape[1:][i] + 0.0)/input_shape[2:][i] for \
                i in range(origin_dim)]
        scale = [1.0, 1.0] + scale

        output_predict = ndimage.interpolation.zoom(predict, scale, order = 1)
        sample['predict'] = output_predict
        return sample

class RandomRescale(AbstractTransform):
    """
    Rescale the input image randomly along each spatial axis. 

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `RandomRescale_lower_bound`: (list/tuple or float) 
        Desired minimal rescale ratio. If tuple/list, the length should be 3 or 2.
    :param `RandomRescale_upper_bound`: (list/tuple or float) 
        Desired maximal rescale ratio. If tuple/list, the length should be 3 or 2.
    :param `RandomRescale_probability`: (optional, float) 
        The probability of applying RandomRescale. Default is 0.5.
    :param `RandomRescale_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `True`.
    """
    def __init__(self, params):
        super(RandomRescale, self).__init__(params)
        self.ratio0 = params["RandomRescale_lower_bound".lower()]
        self.ratio1 = params["RandomRescale_upper_bound".lower()]
        self.prob   = params.get('RandomRescale_probability'.lower(), 0.5)
        self.inverse     = params.get("RandomRescale_inverse".lower(), False)
        assert isinstance(self.ratio0, (float, list, tuple))
        assert isinstance(self.ratio1, (float, list, tuple))

    def __call__(self, sample):

        image = sample['image']
        input_shape = image.shape
        input_dim   = len(input_shape) - 1
        assert(input_dim == len(self.ratio0) and input_dim == len(self.ratio1))
        
        if isinstance(self.ratio0, (list, tuple)):
            for i in range(input_dim):
                if(self.ratio0[i] is None):
                    self.ratio0[i] = 1.0 
                if(self.ratio1[i] is None):
                    self.ratio1[i] = 1.0
                assert(self.ratio0[i] <= self.ratio1[i])
            scale = [self.ratio0[i] + random.random()*(self.ratio1[i] - self.ratio0[i]) \
                for i in range(input_dim)]
        else:
            scale = self.ratio0 + random.random()*(self.ratio1 - self.ratio0)
            scale = [scale] * input_dim
        scale = [1.0] + scale
        image_t = ndimage.interpolation.zoom(image, scale, order = 1)

        sample['image'] = image_t
        sample['RandomRescale_Param'] = json.dumps(input_shape)
        if('label' in sample and \
          self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
            label = sample['label']
            label = ndimage.interpolation.zoom(label, scale, order = 0)
            sample['label'] = label
        if('pixel_weight' in sample and \
          self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
            weight = sample['pixel_weight']
            weight = ndimage.interpolation.zoom(weight, scale, order = 1)
            sample['pixel_weight'] = weight
        
        return sample

    def inverse_transform_for_prediction(self, sample):
        if(isinstance(sample['RandomRescale_Param'], list) or \
            isinstance(sample['RandomRescale_Param'], tuple)):
            origin_shape = json.loads(sample['RandomRescale_Param'][0])
        else:
            origin_shape = json.loads(sample['RandomRescale_Param'])
        origin_dim   = len(origin_shape) - 1
        predict = sample['predict']
        input_shape = predict.shape
        scale = [(origin_shape[1:][i] + 0.0)/input_shape[2:][i] for \
                i in range(origin_dim)]
        scale = [1.0, 1.0] + scale

        output_predict = ndimage.interpolation.zoom(predict, scale, order = 1)
        sample['predict'] = output_predict
        return sample


class Resample(Rescale):
    """Resample the image to a given spatial resolution.
    
    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `Rescale_output_size`: (list/tuple or int) The output size along each spatial axis, 
        such as [D, H, W] or [H, W].  If D is None, the input image is only reslcaled in 2D.
        If int, the smallest axis is matched to output_size keeping aspect ratio the same
        as the input.
    :param `Rescale_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `True`.
    """
    def __init__(self, params):
        super(Rescale, self).__init__(params)
        self.output_spacing = params["Resample_output_spacing".lower()]
        self.ignore_zspacing= params.get("Resample_ignore_zspacing_range".lower(), None)
        self.inverse        = params.get("Resample_inverse".lower(), True)
        # assert isinstance(self.output_size, (int, list, tuple))

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim   = len(input_shape) - 1
        spacing     = sample['spacing']
        out_spacing = [item for item in self.output_spacing] 
        for i in range(input_dim):
            out_spacing[i] = spacing[i] if out_spacing[i] is None else out_spacing[i]
        if(self.ignore_zspacing is not None):
            if(spacing[0] > self.ignore_zspacing[0] and spacing[0] < self.ignore_zspacing[1]):
                out_spacing[0] = spacing[0]
        scale = [spacing[i]  / out_spacing[i] for i in range(input_dim)]
        scale = [1.0] + scale

        image_t = ndimage.interpolation.zoom(image, scale, order = 1)

        sample['image']   = image_t
        sample['spacing'] = out_spacing
        sample['Resample_origin_shape'] = json.dumps(input_shape)
        if('label' in sample and \
        self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
            label = sample['label']
            label = ndimage.interpolation.zoom(label, scale, order = 0)
            sample['label'] = label
        if('pixel_weight' in sample and \
        self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
            weight = sample['pixel_weight']
            weight = ndimage.interpolation.zoom(weight, scale, order = 1)
            sample['pixel_weight'] = weight
        
        return sample

    def inverse_transform_for_prediction(self, sample):
        if(isinstance(sample['Resample_origin_shape'], list) or \
            isinstance(sample['Resample_origin_shape'], tuple)):
            origin_shape = json.loads(sample['Resample_origin_shape'][0])
        else:
            origin_shape = json.loads(sample['Resample_origin_shape'])
        
        origin_dim   = len(origin_shape) - 1
        predict = sample['predict']
        input_shape = predict.shape
        scale = [(origin_shape[1:][i] + 0.0)/input_shape[2:][i] for \
                i in range(origin_dim)]
        scale = [1.0, 1.0] + scale

        output_predict = ndimage.interpolation.zoom(predict, scale, order = 1)
        sample['predict'] = output_predict
        return sample