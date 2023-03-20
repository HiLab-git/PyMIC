# -*- coding: utf-8 -*-
from __future__ import print_function, division

import json
import random
import numpy as np
from pymic import TaskType
from pymic.transform.abstract_transform import AbstractTransform


class RandomTranspose(AbstractTransform):
    """
    Random transpose for 3D volumes. Assume the input has a shape of [C, D, H, W], the
    output shape will be of [C, D, H, W], [C, W, H, D] or [C, H, D, W]

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `RandomTranspose_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `True`.
    """
    def __init__(self, params): 
        super(RandomTranspose, self).__init__(params)
        self.inverse = params.get('RandomTranspose_inverse'.lower(), True)

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        assert(input_dim == 3)

        rand_num = random.random()
        if(rand_num < 0.4):
            transpose_axis = None 
        elif(rand_num < 0.7):
            transpose_axis = [0, 3, 2, 1]
        else:
            transpose_axis = [0, 2, 1, 3]
        sample['RandomTranspose_Param'] = json.dumps(transpose_axis) 
        if(transpose_axis is not None):
            image_t = np.transpose(image, transpose_axis)
            sample['image'] = image_t
            if('label' in sample and \
            self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
                sample['label'] = np.transpose(sample['label'] , transpose_axis)
            if('pixel_weight' in sample and \
            self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
                sample['pixel_weight'] = np.transpose(sample['pixel_weight'] , transpose_axis)           
        return sample

    def  inverse_transform_for_prediction(self, sample):
        if(isinstance(sample['RandomTranspose_Param'], list) or \
            isinstance(sample['RandomTranspose_Param'], tuple)):
            transpose_axis = json.loads(sample['RandomTranspose_Param'][0]) 
        else:
            transpose_axis = json.loads(sample['RandomTranspose_Param'])
        if(transpose_axis is not None):
            sample['predict']  = np.transpose(sample['predict'] , transpose_axis)
        return sample