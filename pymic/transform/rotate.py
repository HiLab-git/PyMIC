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


class RandomRotate(AbstractTransform):
    """
    random rotate the image (shape [C, D, H, W] or [C, H, W]) 
    """
    def __init__(self, params): 
        """
        angle_range_d (tuple/list/None) : rorate angle range along depth axis (degree),
               only used for 3D images
        angle_range_h (tuple/list/None) : rorate angle range along height axis (degree)
        angle_range_w (tuple/list/None) : rorate angle range along width axis (degree)
        """
        super(RandomRotate, self).__init__(params)
        self.angle_range_d  = params['RandomRotate_angle_range_d'.lower()]
        self.angle_range_h  = params['RandomRotate_angle_range_h'.lower()]
        self.angle_range_w  = params['RandomRotate_angle_range_w'.lower()]
        self.inverse = params['RandomRotate_inverse'.lower()]

    def __apply_transformation(self, image, transform_param_list, order = 1):
        """
        apply rotation transformation to an ND image
        Args:
            image (nd array): the input nd image
            transform_param_list (list): a list of roration angle and axes
            order (int): interpolation order
        """
        for angle, axes in transform_param_list:
            image = ndimage.rotate(image, angle, axes, reshape = False, order = order)
        return image

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        
        transform_param_list = []
        if(self.angle_range_d is not None):
            angle_d = np.random.uniform(self.angle_range_d[0], self.angle_range_d[1])
            transform_param_list.append([angle_d, (-1, -2)])
        if(input_dim == 3):
            if(self.angle_range_h is not None):
                angle_h = np.random.uniform(self.angle_range_h[0], self.angle_range_h[1])
                transform_param_list.append([angle_h, (-1, -3)])
            if(self.angle_range_w is not None):
                angle_w = np.random.uniform(self.angle_range_w[0], self.angle_range_w[1])
                transform_param_list.append([angle_w, (-2, -3)])
        assert(len(transform_param_list) > 0)

        sample['RandomRotate_Param'] = json.dumps(transform_param_list)
        image_t = self.__apply_transformation(image, transform_param_list, 1)
        sample['image'] = image_t
        if('label' in sample and self.task == 'segmentation'):
            sample['label'] = self.__apply_transformation(sample['label'] , 
                                transform_param_list, 0)
        if('pixel_weight' in sample and self.task == 'segmentation'):
            sample['pixel_weight'] = self.__apply_transformation(sample['pixel_weight'] , 
                                transform_param_list, 1)
        return sample

    def  inverse_transform_for_prediction(self, sample):
        ''' rorate sample['predict'] (5D or 4D) to the original direction.
        assume batch size is 1, otherwise rotate parameter may be different for 
        different elemenets in the batch.

        transform_param_list is a list as saved in __call__().'''
        # get the paramters for invers transformation
        if(isinstance(sample['RandomRotate_Param'], list) or \
            isinstance(sample['RandomRotate_Param'], tuple)):
            transform_param_list = json.loads(sample['RandomRotate_Param'][0]) 
        else:
            transform_param_list = json.loads(sample['RandomRotate_Param']) 
        transform_param_list.reverse()
        for i in range(len(transform_param_list)):
            transform_param_list[i][0] = - transform_param_list[i][0]
        sample['predict'] = self.__apply_transformation(sample['predict'] , 
                                transform_param_list, 1)
        return sample