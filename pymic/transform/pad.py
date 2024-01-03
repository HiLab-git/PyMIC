# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import json
import math
import random
import numpy as np
from pymic import TaskType
from pymic.transform.abstract_transform import AbstractTransform
from pymic.util.image_process import *


class Pad(AbstractTransform):
    """
    Pad an image to an new spatial shape.
    The image has a shape of [C, D, H, W] or [C, H, W]. 
    The real output size will be max(image_size, output_size).

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `Pad_output_size`: (list/tuple) The output size along each spatial axis. 
    :param `Pad_ceil_mode`: (optional, bool) If true, the real output size will
        be the minimal integer multiples of output_size higher than the input size.
        For example, the input image has a shape of [3, 100, 100], `Pad_output_size` 
        = [32, 32], and the real output size will be [3, 128, 128] if `Pad_ceil_mode` = True.
    :param `Pad_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `True`.
    """
    def __init__(self, params):
        super(Pad, self).__init__(params)
        self.output_size = params['Pad_output_size'.lower()]
        self.ceil_mode   = params.get('Pad_ceil_mode'.lower(), False)
        self.inverse = params.get('Pad_inverse'.lower(), True)

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        assert(len(self.output_size) == input_dim)
        if(self.ceil_mode):
            multiple = [int(math.ceil(float(input_shape[1+i])/self.output_size[i]))\
                for i in range(input_dim)]
            output_size = [multiple[i] * self.output_size[i] \
                for i in range(input_dim)]
        else:
            output_size = self.output_size
        margin = [max(0, output_size[i] - input_shape[1+i]) \
            for i in range(input_dim)]

        margin_lower = [int(margin[i] / 2) for i in range(input_dim)]
        margin_upper = [margin[i] - margin_lower[i] for i in range(input_dim)]
        sample['Pad_Param'] = json.dumps((margin_lower, margin_upper))

        pad = [(margin_lower[i], margin_upper[i]) for  i in range(input_dim)]
        pad = tuple([(0, 0)] + pad)
        image_t = np.pad(image, pad, 'reflect') if(max(margin) > 0) else image

        sample['image'] = image_t
        
        if('label' in sample and \
        self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
            label = sample['label']
            label = np.pad(label, pad, 'reflect') if(max(margin) > 0) else label
            sample['label'] = label
        if('pixel_weight' in sample and \
        self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
            weight = sample['pixel_weight']
            weight = np.pad(weight, pad, 'reflect') if(max(margin) > 0) else weight
            sample['pixel_weight'] = weight
        return sample
    
    def inverse_transform_for_prediction(self, sample):
        if(isinstance(sample['Pad_Param'], list) or isinstance(sample['Pad_Param'], tuple)):
            params = json.loads(sample['Pad_Param'][0]) 
        else:
            params = json.loads(sample['Pad_Param']) 
        margin_lower = params[0]
        margin_upper = params[1]
        predict = sample['predict']
        if(isinstance(predict, tuple) or isinstance(predict, list)):
            output_predict = []
            for predict_i in predict:
                predict_shape = predict_i.shape
                crop_min = [0, 0] + margin_lower
                crop_max = [predict_shape[2:][i] - margin_upper[i] \
                    for i in range(len(margin_lower))]
                crop_max = list(predict_shape[:2]) + crop_max
                crop_predict = crop_ND_volume_with_bounding_box(predict_i, crop_min, crop_max)
                output_predict.append(crop_predict)
        else:
            predict_shape = predict.shape
            crop_min = [0, 0] + margin_lower
            crop_max = [predict_shape[2:][i] - margin_upper[i] \
                for i in range(len(margin_lower))]
            crop_max = list(predict_shape[:2]) + crop_max

            output_predict = crop_ND_volume_with_bounding_box(predict, crop_min, crop_max)
        sample['predict'] = output_predict
        return sample