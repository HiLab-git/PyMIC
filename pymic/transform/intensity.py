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
    Apply random gamma correction to given channels.

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `GammaCorrection_channels`: (list) A list of int for specifying the channels.
    :param `GammaCorrection_gamma_min`: (float) The minimal gamma value.
    :param `GammaCorrection_gamma_max`: (float) The maximal gamma value.
    :param `GammaCorrection_probability`: (optional, float) 
        The probability of applying GammaCorrection. Default is 0.5.
    :param `GammaCorrection_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `False`.
    """
    def __init__(self, params):
        super(GammaCorrection, self).__init__(params)
        self.channels =  params['GammaCorrection_channels'.lower()]
        self.gamma_min = params['GammaCorrection_gamma_min'.lower()]
        self.gamma_max = params['GammaCorrection_gamma_max'.lower()]
        self.prob      = params.get('GammaCorrection_probability'.lower(), 0.5)
        self.inverse   = params.get('GammaCorrection_inverse'.lower(), False)
    
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
    Add Gaussian Noise to given channels.

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `GaussianNoise_channels`: (list) A list of int for specifying the channels.
    :param `GaussianNoise_mean`: (float) The mean value of noise.
    :param `GaussianNoise_std`: (float) The std of noise.
    :param `GaussianNoise_probability`: (optional, float) 
        The probability of applying GaussianNoise. Default is 0.5.
    :param `GaussianNoise_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `False`.
    """
    def __init__(self, params):
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

class GrayscaleToRGB(AbstractTransform):
    """
    Convert gray scale images to RGB by copying channels. 
    """
    def __init__(self, params):
        super(GrayscaleToRGB, self).__init__(params)
        self.inverse = params.get('GrayscaleToRGB_inverse'.lower(), False)
    
    def __call__(self, sample):
        image= sample['image']
        assert(image.shape[0] == 1 or image.shape[0] == 3)
        if(image.shape[0] == 1):
            sample['image'] = np.concatenate([image, image, image])
        return sample