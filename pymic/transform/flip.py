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


class RandomFlip(AbstractTransform):
    """ random flip the image (shape [C, D, H, W] or [C, H, W]) """
    def __init__(self, params):
        """
            flip_depth (bool) : random flip along depth axis or not, only used for 3D images
            flip_height (bool): random flip along height axis or not
            flip_width (bool) : random flip along width axis or not
        """
        super(RandomFlip, self).__init__(params)
        self.flip_depth  = params['RandomFlip_flip_depth'.lower()]
        self.flip_height = params['RandomFlip_flip_height'.lower()]
        self.flip_width  = params['RandomFlip_flip_width'.lower()]
        self.inverse = params['RandomFlip_inverse'.lower()]

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        flip_axis = []
        if(self.flip_width):
            if(random.random() > 0.5):
                flip_axis.append(-1)
        if(self.flip_height):
            if(random.random() > 0.5):
                flip_axis.append(-2)
        if(input_dim == 3 and self.flip_depth):
            if(random.random() > 0.5):
                flip_axis.append(-3)

        sample['RandomFlip_Param'] = json.dumps(flip_axis)
        if(len(flip_axis) > 0):
            # use .copy() to avoid negative strides of numpy array
            # current pytorch does not support negative strides
            image_t = np.flip(image, flip_axis).copy()
            sample['image'] = image_t
            if('label' in sample and self.task == 'segmentation'):
                sample['label'] = np.flip(sample['label'] , flip_axis).copy()
            if('pixel_weight' in sample and self.task == 'segmentation'):
                sample['pixel_weight'] = np.flip(sample['pixel_weight'] , flip_axis).copy()
            
        return sample

    def  inverse_transform_for_prediction(self, sample):
        ''' flip sample['predict'] (5D or 4D) to the original direction.
         assume batch size is 1, otherwise flip parameter may be different for 
         different elemenets in the batch.

        flip_axis is a list as saved in __call__().'''
        if(isinstance(sample['RandomFlip_Param'], list) or \
            isinstance(sample['RandomFlip_Param'], tuple)):
            flip_axis = json.loads(sample['RandomFlip_Param'][0]) 
        else:
            flip_axis = json.loads(sample['RandomFlip_Param']) 
        if(len(flip_axis) > 0):
            sample['predict']  = np.flip(sample['predict'] , flip_axis).copy()
        return sample