# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import json
import math
import random
import numpy as np
from scipy import ndimage
from pymic.transform.abstract_transform import AbstractTransform
from pymic.util.image_process import *


class Pad(AbstractTransform):
    """
    Pad the image (shape [C, D, H, W] or [C, H, W]) to an new spatial shape, 
    the real output size will be max(image_size, output_size)
    """
    def __init__(self, params):
        """
        :param output_size: (tuple/list) the size along each spatial axis. 
        :param ceil_mode: (bool) if true, the real output size is integer multiples of output_size.
        """
        super(Pad, self).__init__(params)
        self.output_size = params['Pad_output_size'.lower()]
        self.ceil_mode   = params['Pad_ceil_mode'.lower()]
        self.inverse = params['Pad_inverse'.lower()]

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
        
        if('label' in sample and self.task == 'segmentation'):
            label = sample['label']
            label = np.pad(label, pad, 'reflect') if(max(margin) > 0) else label
            sample['label'] = label
        if('pixel_weight' in sample and self.task == 'segmentation'):
            weight = sample['pixel_weight']
            weight = np.pad(weight, pad, 'reflect') if(max(margin) > 0) else weight
            sample['pixel_weight'] = weight
        return sample
    
    def inverse_transform_for_prediction(self, sample):
        ''' crop sample['predict'] (5D or 4D) to the original spatial shape.
         assume batch size is 1, otherwise scale may be different for 
         different elemenets in the batch.

        origin_shape is a 4D or 3D vector as saved in __call__().'''
        # raise ValueError("not implemented")
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