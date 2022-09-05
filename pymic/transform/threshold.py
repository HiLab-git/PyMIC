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
    """
    Thresholding the image for given channels.

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `ChannelWiseThreshold_channels`: (list/tuple or None) 
        A list of specified channels for thresholding. If None (by default), 
        all the channels will be thresholded.
    :param `ChannelWiseThreshold_threshold_lower`: (list/tuple or None) 
        The lower threshold for the given channels.
    :param `ChannelWiseThreshold_threshold_upper`: (list/tuple or None) 
        The upper threshold for the given channels.  
    :param `ChannelWiseThreshold_replace_lower`: (list/tuple or None) 
        The output value for pixels with an input value lower than the threshold_lower.
    :param `ChannelWiseThreshold_replace_upper`: (list/tuple or None) 
        The output value for pixels with an input value higher than the threshold_upper.      
    :param `ChannelWiseThreshold_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `False`.
    """
    def __init__(self, params):
        super(ChannelWiseThreshold, self).__init__(params)
        self.channels = params['ChannelWiseThreshold_channels'.lower()]
        self.threshold_lower = params['ChannelWiseThreshold_threshold_lower'.lower()]
        self.threshold_upper = params['ChannelWiseThreshold_threshold_upper'.lower()]
        self.replace_lower   = params['ChannelWiseThreshold_replace_lower'.lower()]
        self.replace_upper   = params['ChannelWiseThreshold_replace_upper'.lower()]
        self.inverse = params.get('ChannelWiseThreshold_inverse'.lower(), False)

    def __call__(self, sample):
        image= sample['image']
        channels = range(image.shape[0]) if self.channels is None else self.channels
        for i in range(len(channels)):
            chn = channels[i]
            if((self.threshold_lower is not None) and (self.threshold_lower[i] is not None)):
                t_lower = self.threshold_lower[i]
                r_lower = self.threshold_lower[i]
                if((self.replace_lower is not None) and (self.replace_lower[i] is not None)):
                    r_lower = self.replace_lower[i]
                image[chn][image[chn] < t_lower] = r_lower
            
            if((self.threshold_upper is not None) and (self.threshold_upper[i] is not None)):
                t_upper = self.threshold_upper[i]
                r_upper = self.threshold_upper[i]
                if((self.replace_upper is not None) and (self.replace_upper[i] is not None)):
                    r_upper= self.replace_upper[i]
                image[chn][image[chn] > t_upper] = r_upper
        sample['image'] = image
        return sample

class ChannelWiseThresholdWithNormalize(AbstractTransform):
    """
    Apply thresholding and normalization for given channels. 
    Pixel intensity will be truncated to the range of (lower, upper) and then 
    normalized. If mean_std_mode is True, the mean and std values for pixel
    in the target range is calculated for normalization, and input intensity 
    outside that range will be replaced by random values. Otherwise, the intensity
    will be normalized to [0, 1].
    
    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `ChannelWiseThresholdWithNormalize_channels`: (list/tuple or None) 
        A list of specified channels for thresholding. If None (by default), 
        all the channels will be affected by this transform.
    :param `ChannelWiseThresholdWithNormalize_threshold_lower`: (list/tuple or None) 
        The lower threshold for the given channels.
    :param `ChannelWiseThresholdWithNormalize_threshold_upper`: (list/tuple or None) 
        The upper threshold for the given channels.  
    :param `ChannelWiseThresholdWithNormalize_mean_std_mode`: (bool) 
        If True, using mean and std for normalization. If False, using min and max 
        values for normalization.      
    :param `ChannelWiseThresholdWithNormalize_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `False`.
    """
    def __init__(self, params):
        super(ChannelWiseThresholdWithNormalize, self).__init__(params)
        self.channels = params['ChannelWiseThresholdWithNormalize_channels'.lower()]
        self.threshold_lower = params['ChannelWiseThresholdWithNormalize_threshold_lower'.lower()]
        self.threshold_upper = params['ChannelWiseThresholdWithNormalize_threshold_upper'.lower()]
        self.mean_std_mode   = params['ChannelWiseThresholdWithNormalize_mean_std_mode'.lower()]
        self.inverse = params.get('ChannelWiseThresholdWithNormalize_inverse'.lower(), False)

    def __call__(self, sample):
        image= sample['image']
        channels = range(image.shape[0]) if self.channels is None else self.channels
        for chn in channels:
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