import json
import math
import random
import numpy as np
from scipy import ndimage
from pymic.transform.abstract_transform import AbstractTransform
from pymic.util.image_process import *


class NormalizeWithMeanStd(AbstractTransform):
    """
    Normalize the image based on mean and std. The image should have a shape
    of [C, D, H, W] or [C, H, W].  

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `NormalizeWithMeanStd_channels`: (list/tuple or None) 
        A list or tuple of int for specifying the channels. 
        If None, the transform operates on all the channels.
    :param `NormalizeWithMeanStd_mean`: (list/tuple or None) 
        The mean values along each specified channel.
        If None, the mean values are calculated automatically.
    :param `NormalizeWithMeanStd_std`: (list/tuple or None) 
        The std values along each specified channel.
        If None, the std values are calculated automatically.
    :param `NormalizeWithMeanStd_ignore_non_positive`: (optional, bool) 
        Only used when mean and std are not given. Default is False.
        If True, calculate mean and std in the positive region for normalization,
        and set non-positive region to random. If False, calculate
        the mean and std values in the entire image region. 
    :param `NormalizeWithMeanStd_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `False`.
    """
    def __init__(self, params):
        super(NormalizeWithMeanStd, self).__init__(params)
        self.chns = params['NormalizeWithMeanStd_channels'.lower()]
        self.mean = params.get('NormalizeWithMeanStd_mean'.lower(), None)
        self.std  = params.get('NormalizeWithMeanStd_std'.lower(), None)
        self.ingore_np = params.get('NormalizeWithMeanStd_ignore_non_positive'.lower(), False)
        self.inverse = params.get('NormalizeWithMeanStd_inverse'.lower(), False)

    def __call__(self, sample):
        image= sample['image']
        chns = self.chns if self.chns is not None else range(image.shape[0])
        if(self.mean is None):
            self.mean = [None] * len(chns)
            self.std  = [None] * len(chns)

        for i in range(len(chns)):
            chn = chns[i]
            chn_mean, chn_std = self.mean[i], self.std[i]
            if(chn_mean is None):
                if(self.ingore_np):
                    pixels = image[chn][image[chn] > 0]
                    chn_mean, chn_std = pixels.mean(), pixels.std() 
                else:
                    chn_mean, chn_std = image[chn].mean(), image[chn].std()
    
            chn_norm = (image[chn] - chn_mean)/chn_std

            if(self.ingore_np):
                chn_random = np.random.normal(0, 1, size = chn_norm.shape)
                chn_norm[image[chn] <= 0] = chn_random[image[chn] <= 0]
            image[chn] = chn_norm
        sample['image'] = image
        return sample


class NormalizeWithMinMax(AbstractTransform):
    """Nomralize the image to [0, 1]. The shape should be [C, D, H, W] or [C, H, W].

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `NormalizeWithMinMax_channels`: (list/tuple or None) 
        A list or tuple of int for specifying the channels. 
        If None, the transform operates on all the channels.
    :param `NormalizeWithMinMax_threshold_lower`: (list/tuple or None) 
        The min values along each specified channel.
        If None, the min values are calculated automatically.
    :param `NormalizeWithMinMax_threshold_upper`: (list/tuple or None) 
        The max values along each specified channel.
        If None, the max values are calculated automatically.
    :param `NormalizeWithMinMax_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `False`.
    """
    def __init__(self, params):
        super(NormalizeWithMinMax, self).__init__(params)
        self.chns = params['NormalizeWithMinMax_channels'.lower()]
        self.thred_lower = params['NormalizeWithMinMax_threshold_lower'.lower()]
        self.thred_upper = params['NormalizeWithMinMax_threshold_upper'.lower()]
        self.inverse = params.get('NormalizeWithMinMax_inverse'.lower(), False)

    def __call__(self, sample):
        image= sample['image']
        chns = self.chns if self.chns is not None else range(image.shape[0])
        for i in range(len(chns)):
            chn = chns[i]
            img_chn = image[chn]
            v0, v1 = img_chn.min(), img_chn.max()
            if(self.thred_lower is not None) and  (self.thred_lower[i] is not None):
                v0 = self.thred_lower[i]
            if(self.thred_upper is not None) and  (self.thred_upper[i] is not None):
                v1 = self.thred_upper[i]

            img_chn[img_chn < v0] = v0
            img_chn[img_chn > v1] = v1
            img_chn = (img_chn - v0) / (v1 - v0)
            image[chn] = img_chn
        sample['image'] = image
        return sample

class NormalizeWithPercentiles(AbstractTransform):
    """Nomralize the image to [0, 1] with percentiles for given channels.
    The shape should be [C, D, H, W] or [C, H, W].
    
    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `NormalizeWithPercentiles_channels`: (list/tuple or None) 
        A list or tuple of int for specifying the channels. 
        If None, the transform operates on all the channels.
    :param `NormalizeWithPercentiles_percentile_lower`: (float) 
        The min percentile, which must be between 0 and 100 inclusive.
    :param `NormalizeWithPercentiles_percentile_upper`: (float) 
        The max percentile, which must be between 0 and 100 inclusive.
    :param `NormalizeWithMinMax_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `False`.
    """
    def __init__(self, params):
        super(NormalizeWithPercentiles, self).__init__(params)
        self.chns = params['NormalizeWithPercentiles_channels'.lower()]
        self.percent_lower = params['NormalizeWithPercentiles_percentile_lower'.lower()]
        self.percent_upper = params['NormalizeWithPercentiles_percentile_upper'.lower()]
        self.inverse = params.get('NormalizeWithPercentiles_inverse'.lower(), False)

    def __call__(self, sample):
        image= sample['image']
        chns = self.chns if self.chns is not None else range(image.shape[0])
        for i in range(len(chns)):
            chn = chns[i]
            img_chn = image[chn]
            v0 = np.percentile(img_chn, self.percent_lower)
            v1 = np.percentile(img_chn, self.percent_upper)

            img_chn[img_chn < v0] = v0
            img_chn[img_chn > v1] = v1
            img_chn = (img_chn - v0) / (v1 - v0)
            image[chn] = img_chn
        sample['image'] = image
        return sample