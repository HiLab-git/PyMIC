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


class ChannelWiseThreshold(AbstractTransform):
    """Threshold the image (shape [C, D, H, W] or [C, H, W]) for each channel
    """
    def __init__(self, params):
        """
        threshold (tuple/list): The threshold value along each channel.
        """
        super(ChannelWiseThreshold, self).__init__(params)
        self.threshold = params['ChannelWiseThreshold_threshold'.lower()]
        self.inverse = params['ChannelWiseThreshold_inverse'.lower()]

    def __call__(self, sample):
        image= sample['image']
        for chn in range(image.shape[0]):
            mask = np.asarray(image[chn] > self.threshold[chn], image.dtype)
            image[chn] = mask * (image[chn] - self.threshold[chn])

        sample['image'] = image
        return sample

class ChannelWiseThresholdWithNormalize(AbstractTransform):
    """Threshold the image (shape [C, D, H, W] or [C, H, W]) for each channel
       and then normalize the image based on remaining pixels
    """
    def __init__(self, params):
        """
        :param threshold_lower: (tuple/list/None) The lower threshold value along each channel.
        :param threshold_upper: (typle/list/None) The upper threshold value along each channel.
        :param mean_std_mode: (bool) If true, nomalize the image based on mean and std values,
            and pixels values outside the threshold value are replaced random number.
            If false, use the min and max values for normalization.
        """
        super(ChannelWiseThresholdWithNormalize, self).__init__(params)
        self.threshold_lower = params['ChannelWiseThresholdWithNormalize_threshold_lower'.lower()]
        self.threshold_upper = params['ChannelWiseThresholdWithNormalize_threshold_upper'.lower()]
        self.mean_std_mode   = params['ChannelWiseThresholdWithNormalize_mean_std_mode'.lower()]
        self.inverse = params['ChannelWiseThresholdWithNormalize_inverse'.lower()]

    def __call__(self, sample):
        image= sample['image']
        for chn in range(image.shape[0]):
            v0 = self.threshold_lower[chn]
            v1 = self.threshold_upper[chn]
            if(self.mean_std_mode == True):
                mask = np.ones_like(image[chn])
                if(v0 is not None):
                    mask = mask * np.asarray(image[chn] > v0)
                if(v1 is not None):
                    mask = mask * np.asarray(image[chn] < v1)
                pixels   = image[chn][mask > 0]
                chn_mean = pixels.mean()
                chn_std  = pixels.std()
                chn_norm = (image[chn] - chn_mean)/chn_std
                chn_random = np.random.normal(0, 1, size = chn_norm.shape)
                chn_norm[mask == 0] = chn_random[mask == 0]
                image[chn] = chn_norm
            else:
                img_chn = image[chn]
                if(v0 is not None):
                    img_chn[img_chn < v0] = v0
                    min_value = v0 
                else:
                    min_value = img_chn.min()
                if(v1 is not None):
                    img_chn[img_chn > v1] = v1 
                    max_value = img_chn.max() 
                else:
                    max_value = img_chn.max() 
                img_chn = (img_chn - min_value) / (max_value - min_value)
                image[chn] = img_chn
        sample['image'] = image
        return sample