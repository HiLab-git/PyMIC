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


class Rescale(AbstractTransform):
    """Rescale the image in a sample to a given size."""
    def __init__(self, params):
        """
        output_size (tuple/list or int): Desired output size. 
            If tuple/list, output_size should in the format of [D, H, W] or [H, W].
            Channel number is kept the same as the input. If D is None, the input image
            is only reslcaled in 2D.
            If int, the smallest axis is matched to output_size keeping 
            aspect ratio the same.
        """
        super(Rescale, self).__init__(params)
        self.output_size = params["Rescale_output_size".lower()]
        self.inverse     = params["Rescale_inverse".lower()]
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
        if('label' in sample and self.task == 'segmentation'):
            label = sample['label']
            label = ndimage.interpolation.zoom(label, scale, order = 0)
            sample['label'] = label
        if('pixel_weight' in sample and self.task == 'segmentation'):
            weight = sample['pixel_weight']
            weight = ndimage.interpolation.zoom(weight, scale, order = 1)
            sample['pixel_weight'] = weight
        
        return sample

    def inverse_transform_for_prediction(self, sample):
        ''' rescale sample['predict'] (5D or 4D) to the original spatial shape.
         assume batch size is 1, otherwise scale may be different for 
         different elemenets in the batch.

        origin_shape is a 4D or 3D vector as saved in __call__().'''
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