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
            region or the mask region.
        :param mask:  (bool) Only used when mean and std are not given.
        :param random_fill:  (bool) When mask is used, set non-mask region to random. 
        :param inverse: (bool) Whether inverse transform is needed or not.
        """
        super(NormalizeWithMeanStd, self).__init__(params)
        self.chns = params['NormalizeWithMeanStd_channels'.lower()]
        self.mean = params['NormalizeWithMeanStd_mean'.lower()]
        self.std  = params['NormalizeWithMeanStd_std'.lower()]
        self.mask_enable = params['NormalizeWithMeanStd_mask'.lower()]
        self.random_fill = params['NormalizeWithMeanStd_random_fill'.lower()]
        self.inverse = params['NormalizeWithMeanStd_inverse'.lower()]

    def __call__(self, sample):
        image= sample['image']
        chns = self.chns if self.chns is not None else range(image.shape[0])
        if self.mask_enable:
            mask = sample['mask']

        for i in range(len(chns)):
            chn = chns[i]
            if(self.mean is None or self.std is None):
                pixels = image[chn][mask > 0] if self.mask_enable else image[chn]
                chn_mean, chn_std = pixels.mean(), pixels.std()
            else:
                chn_mean, chn_std = self.mean[i], self.std[i]
            chn_norm = (image[chn] - chn_mean)/chn_std
            if(self.mask_enable and self.random_fill):
                chn_random = np.random.normal(0, 1, size = chn_norm.shape)
                chn_norm[mask == 0] = chn_random[mask == 0]
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
        self.inverse = params['NormalizeWithMinMax_inverse'.lower()]

    def __call__(self, sample):
        image= sample['image']
        chns = self.chns if self.chns is not None else range(image.shape[0])
        for i in range(len(chns)):
            chn = chns[i]
            img_chn = image[chn]
            v0 = img_chn.min() if self.thred_lower is None else self.thred_lower[i]
            v1 = img_chn.max() if self.thred_upper is None else self.thred_upper[i]

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
        self.inverse = params['NormalizeWithPercentiles_inverse'.lower()]

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