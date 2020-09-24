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


class ChannelWiseGammaCorrection(AbstractTransform):
    """
    apply random gamma correction to each channel
    """
    def __init__(self, params):
        """
        (gamma_min, gamma_max) specify the range of gamma
        """
        super(ChannelWiseGammaCorrection, self).__init__(params)
        self.gamma_min = params['ChannelWiseGammaCorrection_gamma_min'.lower()]
        self.gamma_max = params['ChannelWiseGammaCorrection_gamma_max'.lower()]
        self.inverse = params['ChannelWiseGammaCorrection_inverse'.lower()]
    
    def __call__(self, sample):
        image= sample['image']
        for chn in range(image.shape[0]):
            gamma_c = random.random() * (self.gamma_max - self.gamma_min) + self.gamma_min
            img_c = image[chn]
            v_min = img_c.min()
            v_max = img_c.max()
            img_c = (img_c - v_min)/(v_max - v_min)
            img_c = np.power(img_c, gamma_c)*(v_max - v_min) + v_min
            image[chn] = img_c

        sample['image'] = image
        return sample

