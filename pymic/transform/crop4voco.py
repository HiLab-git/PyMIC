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
from pymic.transform.crop import CenterCrop
from pymic.transform.intensity import *
from pymic.util.image_process import *

def get_position_label(roi=96, num_crops=4):
    half    = roi // 2
    max_roi = roi * num_crops
    center_x, center_y = np.random.randint(low=half, high=max_roi - half), \
        np.random.randint(low=half, high=max_roi - half)

    x_min, x_max = center_x - half, center_x + half
    y_min, y_max = center_y - half, center_y + half

    total_area = roi * roi
    labels = []
    for j in range(num_crops):
        for i in range(num_crops):
            crop_x_min, crop_x_max = i * roi, (i + 1) * roi
            crop_y_min, crop_y_max = j * roi, (j + 1) * roi

            dx = min(crop_x_max, x_max) - max(crop_x_min, x_min)
            dy = min(crop_y_max, y_max) - max(crop_y_min, y_min)
            if dx <= 0 or dy <= 0:
                area = 0
            else:
                area = (dx * dy) / total_area
            labels.append(area)

    labels = np.asarray(labels).reshape(1, num_crops * num_crops)
    return x_min, y_min, labels

class Crop4VoCo(CenterCrop):
    """
    Randomly crop an volume into two views with augmentation. This is used for
    self-supervised pretraining such as DeSD.

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `DualViewCrop_output_size`: (list/tuple) Desired output size [D, H, W].
        The output channel is the same as the input channel. 
    :param `DualViewCrop_scale_lower_bound`: (list/tuple) Lower bound of the range of scale
        for each dimension. e.g. (1.0, 0.5, 0.5).
    param `DualViewCrop_scale_upper_bound`: (list/tuple) Upper bound of the range of scale
        for each dimension. e.g. (1.0, 2.0, 2.0).
    :param `DualViewCrop_inverse`: (optional, bool) Is inverse transform needed for inference.
        Default is `False`. Currently, the inverse transform is not supported, and 
        this transform is assumed to be used only during training stage. 
    """
    def __init__(self, params):
        roi_size = params.get('Crop4VoCo_roi_size'.lower(), 64)
        if isinstance(roi_size, int):
            self.roi_size = [roi_size] * 3 
        else:
            self.roi_size = roi_size
        self.roi_num  = params.get('Crop4VoCo_roi_num'.lower(), 2)
        self.base_num = params.get('Crop4VoCo_base_num'.lower(), 4)
        
        self.inverse     = params.get('Crop4VoCo_inverse'.lower(), False)
        self.task        = params['Task'.lower()]
         
    def __call__(self, sample):
        image = sample['image']
        channel, input_size = image.shape[0], image.shape[1:]
        input_dim   = len(input_size)
        # print(input_size, self.roi_size)
        assert(input_size[0] == self.roi_size[0])
        assert(input_size[1] == self.roi_size[1] * self.base_num)
        assert(input_size[2] == self.roi_size[2] * self.base_num)

        base_num, roi_num, roi_size  = self.base_num, self.roi_num, self.roi_size
        base_crops, roi_crops, roi_labels = [], [], []
        crop_size = [channel] + list(roi_size)
        for j in range(base_num):
            for i in range(base_num):
                crop_min = [0, 0, roi_size[1]*j, roi_size[2]*i]
                crop_max = [crop_min[d] + crop_size[d] for d in range(4)]
                crop_out = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
                base_crops.append(crop_out)

        for i in range(roi_num):
            x_min, y_min, label = get_position_label(self.roi_size[2], base_num)
            # print('label', label)
            crop_min = [0, 0, y_min, x_min]
            crop_max = [crop_min[d] + crop_size[d] for d in range(4)]
            crop_out = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
            roi_crops.append(crop_out)
            roi_labels.append(label)
        roi_labels = np.concatenate(roi_labels, 0).reshape(roi_num, base_num * base_num)

        base_crops = np.stack(base_crops, 0)
        roi_crops  = np.stack(roi_crops, 0)
        sample['image'] = base_crops, roi_crops, roi_labels
        return sample

   