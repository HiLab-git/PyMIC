import json
import math
import random
import numpy as np
from scipy import ndimage
from pymic.transform.abstract_transform import AbstractTransform
from pymic.util.image_process import *


class ChannelWiseNormalize(AbstractTransform):
    """Nomralize the image (shape [C, D, H, W] or [C, H, W]) for each channel
    """
    def __init__(self, params):
        """
        mean (None or tuple/list): The mean values along each channel.
        std  (None or tuple/list): The std values along each channel.
            if mean and std are None, calculate them from non-zero region
        chns (None, or tuple/list): The list of channel indices
        zero_to_random (bool, or tuple/list or bool): indicate whether zero values
             in each channel is replaced  with random values.
        """
        self.mean = params['ChannelWiseNormalize_mean'.lower()]
        self.std  = params['ChannelWiseNormalize_std'.lower()]
        self.chns = params['ChannelWiseNormalize_channels'.lower()]
        self.zero_to_random = params['ChannelWiseNormalize_zero_to_random'.lower()]
        self.inverse = params['ChannelWiseNormalize_inverse'.lower()]

    def __call__(self, sample):
        image= sample['image']
        mask = image[0] > 0
        chns = self.chns
        if(chns is None):
            chns = range(image.shape[0])
        zero_to_random = self.zero_to_random
        if(isinstance(zero_to_random, bool)):
            zero_to_random = [zero_to_random]*len(chns)
        if(not(self.mean is None and self.std is None)):
            assert(len(self.mean) == len(self.std))
            assert(len(self.mean) == len(chns))
        for i in range(len(chns)):
            chn = chns[i]
            if(self.mean is None and self.std is None):
                pixels = image[chn][mask > 0]
                chn_mean = pixels.mean()
                chn_std  = pixels.std()
            else:
                chn_mean = self.mean[i]
                chn_std  = self.std[i]
            chn_norm = (image[chn] - chn_mean)/chn_std
            if(zero_to_random[i]):
                chn_random = np.random.normal(0, 1, size = chn_norm.shape)
                chn_norm[mask == 0] = chn_random[mask == 0]
            image[chn] = chn_norm

        sample['image'] = image
        return sample
