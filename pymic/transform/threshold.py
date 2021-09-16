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
        channels (tuple/list/None): the list of specified channels for thresholding. Default value 
            is all the channels.
        threshold_lower (tuple/list/None): The lower threshold values for specified channels.
        threshold_upper (tuple/list/None): The uppoer threshold values for specified channels.
        replace_lower (tuple/list/None): new values for pixels with intensity smaller than 
            threshold_lower. Default value is 
        replace_upper (tuple/list/None): new values for pixels with intensity larger than threshold_upper.
        """
        super(ChannelWiseThreshold, self).__init__(params)
        self.channlels = params['ChannelWiseThreshold_channels'.lower()]
        self.threshold_lower = params['ChannelWiseThreshold_threshold_lower'.lower()]
        self.threshold_upper = params['ChannelWiseThreshold_threshold_upper'.lower()]
        self.replace_lower   = params['ChannelWiseThreshold_replace_lower'.lower()]
        self.replace_upper   = params['ChannelWiseThreshold_replace_upper'.lower()]
        self.inverse = params.get('ChannelWiseThreshold_inverse'.lower(), False)

    def __call__(self, sample):
        image= sample['image']
        channels = range(image.shape[0]) if self.channlels is None else self.channlels
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
    Note that this can be replaced by ChannelWiseThreshold + NormalizeWithMinMax
    
    Threshold the image (shape [C, D, H, W] or [C, H, W]) for each channel
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
        self.inverse = params.get('ChannelWiseThresholdWithNormalize_inverse'.lower(), False)

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