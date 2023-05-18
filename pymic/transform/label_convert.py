# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import json
import math
import random
import numpy as np
from scipy import ndimage
from pymic import TaskType
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
        if(self.task == TaskType.SEGMENTATION):
            label = sample['label'][0] # sample['label'] is (1, h, w)
            label_prob = np.zeros((self.class_num, *label.shape), dtype = np.float32)
            for i in range(self.class_num):
                label_prob[i] = label == i*np.ones_like(label)
            sample['label_prob'] = label_prob
        elif(self.task == TaskType.CLASSIFICATION_ONE_HOT):
            label_idx = sample['label']
            label_prob = np.zeros((self.class_num,), np.float32)
            label_prob[label_idx] = 1.0
            sample['label_prob'] = label_prob 
        return sample

class LabelSmooth(AbstractTransform):
    """
    Apply label smoothing to one-hot labels.
    
    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `LabelSmooth_alpha`: (float) Alpha value for label smoothing.
    :param `LabelSmooth_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `False`.
    """
    def __init__(self, params): 
        super(LabelSmooth, self).__init__(params)
        self.alpha = params['LabelSmooth_alpha'.lower()]
        self.inverse   = params.get('LabelSmooth_inverse'.lower(), False)

    def __call__(self, sample):
        label_prob = sample['label_prob']
        K = list(label_prob.shape)[1]
        sample['label_prob'] = label_prob * (1.0 - self.alpha) + self.alpha / K
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


class SelfReconstructionLabel(AbstractTransform):
    """
    Used for self-supervised learning with image reconstruction tasks. 
    """
    def __init__(self, params): 
        """
        class_num (int): the class number in the label map
        """
        super(SelfReconstructionLabel, self).__init__(params)
        self.inverse   = params.get('SelfReconstructionLabel_inverse'.lower(), False)
    
    def __call__(self, sample):
        image = sample['image'] 
        label = image * 1.0
        sample['label'] = label
        return sample


class MaskedImageModelingLabel(AbstractTransform):
    """
    Used for self-supervised learning with image reconstruction tasks.
    Only reconstruct the masked region in the input. 
    The input images is masked in local patches.  
    """
    def __init__(self, params): 
        """
        class_num (int): the class number in the label map
        """
        super(MaskedImageModelingLabel, self).__init__(params)
        self.patch_size = params.get('MaskedImageModelingLabel_patch_size'.lower(), [16, 16, 16])
        self.masking_ratio = params.get('MaskedImageModelingLabel_ratio'.lower(), 0.15)
        self.inverse    = params.get('MaskedImageModelingLabel_inverse'.lower(), False)
    
    def __call__(self, sample):
        image = sample['image'] 
        C, D, H, W = image.shape 
        patch_size = self.patch_size
        mask = np.ones([D, H, W], np.float32)
        grid_size = [math.ceil((image.shape[i+1] + 0.0) / patch_size[i]) for i in range(3)]
        for d in range(grid_size[0]):
            d0 = d*patch_size[0]
            for h in range(grid_size[1]):
                h0 = h*patch_size[1]
                for w in range(grid_size[2]):
                    w0 = w*patch_size[2] 
                    if(random.random() > self.masking_ratio):
                        continue 
                    d1 = min(d0 + patch_size[0], D)
                    h1 = min(h0 + patch_size[1], H)
                    w1 = min(w0 + patch_size[2], W)
                    mask[d0:d1, h0:h1, w0:w1] = np.zeros([d1 - d0, h1 - h0,  w1 - w0])
        sample['pixel_weight'] = 1 - mask
        sample['image'] = image * mask
        sample['label'] = image
        return sample

