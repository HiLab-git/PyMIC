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
    Remove the first dimension of label tensor.
    """
    def __init__(self, params):
        super(ReduceLabelDim, self).__init__(params)
        self.inverse = params.get('ReduceLabelDim_inverse'.lower(), False)
    
    def __call__(self, sample):
        label = sample['label']
        label_converted = label[0]
        sample['label'] = label_converted
        return sample

class LabelConvert(AbstractTransform):
    """ 
    Convert the label based on a source list and target list. 
    
    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `LabelConvert_source_list`: (list) A list of labels to be converted.
    :param `LabelConvert_target_list`: (list) The target label list.
    :param `LabelConvert_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `False`.
    """
    def __init__(self, params):
        super(LabelConvert, self).__init__(params)
        self.source_list = params['LabelConvert_source_list'.lower()]
        self.target_list = params['LabelConvert_target_list'.lower()]
        self.inverse = params.get('LabelConvert_inverse'.lower(), False)
        assert(len(self.source_list) == len(self.target_list))
    
    def __call__(self, sample):
        label = sample['label']
        label_converted = convert_label(label, self.source_list, self.target_list)
        sample['label'] = label_converted
        return sample

class LabelConvertNonzero(AbstractTransform):
    """ 
    Convert label into binary, i.e., setting nonzero labels as 1.
    """
    def __init__(self, params):
        super(LabelConvertNonzero, self).__init__(params)
        self.inverse = params.get('LabelConvertNonzero_inverse'.lower(), False)
    
    def __call__(self, sample):
        label = sample['label']
        label_converted = np.asarray(label > 0, np.uint8)
        sample['label'] = label_converted
        return sample

class LabelToProbability(AbstractTransform):
    """
    Convert one-channel label map to one-hot multi-channel probability map.
    
    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `LabelToProbability_class_num`: (int) The class number in the label map.
    :param `LabelToProbability_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `False`.
    """
    def __init__(self, params): 
        super(LabelToProbability, self).__init__(params)
        self.class_num = params['LabelToProbability_class_num'.lower()]
        self.inverse   = params.get('LabelToProbability_inverse'.lower(), False)

    def __call__(self, sample):
        if(self.task == 'segmentation'):
            label = sample['label'][0] # sample['label'] is (1, h, w)
            label_prob = np.zeros((self.class_num, *label.shape), dtype = np.float32)
            for i in range(self.class_num):
                label_prob[i] = label == i*np.ones_like(label)
            sample['label_prob'] = label_prob
        elif(self.task == 'classification'):
            label_idx = sample['label']
            label_prob = np.zeros((self.class_num,), np.float32)
            label_prob[label_idx] = 1.0
            sample['label_prob'] = label_prob 
        return sample


class PartialLabelToProbability(AbstractTransform):
    """
    Convert one-channel partial label map to one-hot multi-channel probability map.
    This is used for segmentation tasks only. In the input label map, 0 represents the
    background class, 1 to C-1 represent the foreground classes, and C represents 
    unlabeled pixels. In the output dictionary, `label_prob` is the one-hot probability 
    map, and `pixel_weight` represents a weighting map, where the weight for a pixel
    is 0 if the label is unkown. 

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `PartialLabelToProbability_class_num`: (int) The class number for the 
        segmentation task.  
    :param `PartialLabelToProbability_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `False`.
    """
    def __init__(self, params): 
        """
        class_num (int): the class number in the label map
        """
        super(PartialLabelToProbability, self).__init__(params)
        self.class_num = params['PartialLabelToProbability_class_num'.lower()]
        self.inverse   = params.get('PartialLabelToProbability_inverse'.lower(), False)
    
    def __call__(self, sample):
        label = sample['label'][0]
        assert(label.max() <= self.class_num)
        label_prob = np.zeros((self.class_num, *label.shape), dtype = np.float32)
        for i in range(self.class_num):
            label_prob[i] = label == i*np.ones_like(label)
        sample['label_prob'] = label_prob
        sample['pixel_weight'] = 1.0 - np.asarray([label == self.class_num], np.float32)
        return sample




