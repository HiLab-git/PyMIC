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


class GammaCorrection(AbstractTransform):
    """
    apply random gamma correction to each channel
    """
    def __init__(self, params):
        """
        (gamma_min, gamma_max) specify the range of gamma
        """
        super(GammaCorrection, self).__init__(params)
        self.channels =  params['GammaCorrection_channels'.lower()]
        self.gamma_min = params['GammaCorrection_gamma_min'.lower()]
        self.gamma_max = params['GammaCorrection_gamma_max'.lower()]
        self.prob      = params.get('GammaCorrection_probability'.lower(), 0.5)
        self.inverse = params.get('GammaCorrection_inverse'.lower(), False)
    
    def __call__(self, sample):
        if(np.random.uniform() > self.prob):
            return sample
        image= sample['image']
        for chn in self.channels:
            gamma_c = random.random() * (self.gamma_max - self.gamma_min) + self.gamma_min
            img_c = image[chn]
            v_min = img_c.min()
            v_max = img_c.max()
            img_c = (img_c - v_min)/(v_max - v_min)
            img_c = np.power(img_c, gamma_c)*(v_max - v_min) + v_min
            image[chn] = img_c

        sample['image'] = image
        return sample

class GaussianNoise(AbstractTransform):
    """
    apply random gamma correction to each channel
    """
    def __init__(self, params):
        """
        (gamma_min, gamma_max) specify the range of gamma
        """
        super(GaussianNoise, self).__init__(params)
        self.channels = params['GaussianNoise_channels'.lower()]
        self.mean     = params['GaussianNoise_mean'.lower()]
        self.std      = params['GaussianNoise_std'.lower()]
        self.prob     = params.get('GaussianNoise_probability'.lower(), 0.5)
        self.inverse  = params.get('GaussianNoise_inverse'.lower(), False)
    
    def __call__(self, sample):
        if(np.random.uniform() > self.prob):
            return sample
        image= sample['image']
        for chn in self.channels:
            img_c = image[chn]
            noise = np.random.normal(self.mean, self.std, img_c.shape)
            image[chn] = img_c + noise

        sample['image'] = image
        return sample
