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

class ReduceLabelDim(AbstractTransform):
    """
    remove the first dimension of label tensor
    """
    def __init__(self, params):
        super(ReduceLabelDim, self).__init__(params)
        self.inverse = params['ReduceLabelDim_inverse'.lower()]
    
    def __call__(self, sample):
        label = sample['label']
        label_converted = label[0]
        sample['label'] = label_converted
        return sample

class LabelConvert(AbstractTransform):
    """ Convert a list of labels to another list"""
    def __init__(self, params):
        """
        source_list (tuple/list): A list of labels to be converted
        target_list (tuple/list): The target label list
        """
        super(LabelConvert, self).__init__(params)
        self.source_list = params['LabelConvert_source_list'.lower()]
        self.target_list = params['LabelConvert_target_list'.lower()]
        self.inverse = params['LabelConvert_inverse'.lower()]
        assert(len(self.source_list) == len(self.target_list))
    
    def __call__(self, sample):
        label = sample['label']
        label_converted = convert_label(label, self.source_list, self.target_list)
        sample['label'] = label_converted
        return sample

class LabelConvertNonzero(AbstractTransform):
    """ Convert label into binary (nonzero as 1)"""
    def __init__(self, params):
        super(LabelConvertNonzero, self).__init__(params)
        self.inverse = params['LabelConvertNonzero_inverse'.lower()]
    
    def __call__(self, sample):
        label = sample['label']
        label_converted = np.asarray(label > 0, np.uint8)
        sample['label'] = label_converted
        return sample

class LabelToProbability(AbstractTransform):
    """Convert one-channel label map to one-hot multi-channel probability map"""
    def __init__(self, params): 
        """
        class_num (int): the class number in the label map
        """
        super(LabelToProbability, self).__init__(params)
        self.class_num = params['LabelToProbability_class_num'.lower()]
        self.inverse   = params['LabelToProbability_inverse'.lower()]
    
    def __call__(self, sample):
        if(self.task == 'segmentation'):
            label = sample['label'][0]
            label_prob = []
            for i in range(self.class_num):
                temp_prob = label == i*np.ones_like(label)
                label_prob.append(temp_prob)
            label_prob = np.asarray(label_prob, np.float32)
    
            sample['label_prob'] = label_prob
        elif(self.task == 'classification'):
            label_idx = sample['label']
            label_prob = np.zeros((self.class_num,), np.float32)
            label_prob[label_idx] = 1.0
            sample['label_prob'] = label_prob 
        return sample



