# -*- coding: utf-8 -*-
from __future__ import print_function, division

import json
import random
import numpy as np
from scipy import ndimage
from pymic import TaskType
from pymic.transform.abstract_transform import AbstractTransform
from pymic.util.image_process import *


class RandomRotate(AbstractTransform):
    """
    Random rotate an image, wiht a shape of [C, D, H, W] or [C, H, W].

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `RandomRotate_angle_range_d`: (list/tuple or None) 
        Rotation angle (degree) range along depth axis (x-y plane), e.g., (-90, 90).
        If None, no rotation along this axis. 
    :param `RandomRotate_angle_range_h`: (list/tuple or None) 
        Rotation angle (degree) range along height axis (x-z plane), e.g., (-90, 90).
        If None, no rotation along this axis. Only used for 3D images. 
    :param `RandomRotate_angle_range_w`: (list/tuple or None) 
        Rotation angle (degree) range along width axis (y-z plane), e.g., (-90, 90).
        If None, no rotation along this axis. Only used for 3D images. 
    :param `RandomRotate_probability`: (optional, float) 
        The probability of applying RandomRotate. Default is 0.5.
    :param `RandomRotate_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `True`.
    """
    def __init__(self, params): 
        super(RandomRotate, self).__init__(params)
        self.angle_range_d  = params['RandomRotate_angle_range_d'.lower()]
        self.angle_range_h  = params.get('RandomRotate_angle_range_h'.lower(), None)
        self.angle_range_w  = params.get('RandomRotate_angle_range_w'.lower(), None)
        self.prob = params.get('RandomRotate_probability'.lower(), 0.5)
        self.inverse = params.get('RandomRotate_inverse'.lower(), True)

    def __apply_transformation(self, image, transform_param_list, order = 1):
        """
        Apply rotation transformation to an ND image.
    
        :param image: The input ND image.
        :param transform_param_list:  (list) A list of roration angle and axes.
        :param order: (int) Interpolation order.
        """
        for angle, axes in transform_param_list:
            image = ndimage.rotate(image, angle, axes, reshape = False, order = order)
        return image

    def __call__(self, sample):
        # if(random.random() > self.prob):
        #     sample['RandomRotate_triggered'] = False
        #     return sample
        # else:
        #     sample['RandomRotate_triggered'] = True
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
        # select a random transform from the possible list rather than 
        # use a combination for higher efficiency
        transform_param_list = [random.choice(transform_param_list)]
        sample['RandomRotate_Param'] = json.dumps(transform_param_list)
        image_t = self.__apply_transformation(image, transform_param_list, 1)
        sample['image'] = image_t
        if('label' in sample and \
        self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
            sample['label'] = self.__apply_transformation(sample['label'] , 
                                transform_param_list, 0)
        if('pixel_weight' in sample and \
        self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
            sample['pixel_weight'] = self.__apply_transformation(sample['pixel_weight'] , 
                                transform_param_list, 1)
        return sample

    def  inverse_transform_for_prediction(self, sample):
        if(not sample['RandomRotate_triggered']):
            return sample
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