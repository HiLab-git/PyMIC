import json
import math
import random
import numpy as np
from scipy import ndimage
from pymic.transform.abstract_transform import AbstractTransform
from pymic.util.image_process import *


class NormalizeWithMeanStd(AbstractTransform):
    """Nomralize the image (shape [C, D, H, W] or [C, H, W]) with mean and std for given channels
    """
    def __init__(self, params):
        """
        :param chanels: (None or tuple/list) the indices of channels to be noramlized.
        :param mean: (None or tuple/list): The mean values along each channel.
        :param  std : (None or tuple/list): The std values along each channel.
            When mean and std are not provided, calculate them from the entire image
            region or the non-positive region.
        :param ignore_non_positive: (bool) Only used when mean and std are not given. 
            Use positive region to calculate mean and std, and set non-positive region to random.  
        :param inverse: (bool) Whether inverse transform is needed or not.
        """
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
    """Nomralize the image (shape [C, D, H, W] or [C, H, W]) with min and max for given channels
    """
    def __init__(self, params):
        """
        :param chanels: (None or tuple/list) the indices of channels to be noramlized.
        :param threshold_lower: (tuple/list/None) The lower threshold value along each channel.
        :param threshold_upper: (typle/list/None) The upper threshold value along each channel.
        :param inverse: (bool) Whether inverse transform is needed or not.
        """
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
    """Nomralize the image (shape [C, D, H, W] or [C, H, W]) with percentiles for given channels
    """
    def __init__(self, params):
        """
        :param chanels: (None or tuple/list) the indices of channels to be noramlized.
        :param percentile_lower: (tuple/list/None) The lower percentile along each channel.
        :param percentile_upper: (typle/list/None) The upper percentile along each channel.
        :param inverse: (bool) Whether inverse transform is needed or not.
        """
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