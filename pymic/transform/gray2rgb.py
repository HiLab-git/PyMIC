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


class GrayscaleToRGB(AbstractTransform):
    """
    apply random gamma correction to each channel
    """
    def __init__(self, params):
        """
        (gamma_min, gamma_max) specify the range of gamma
        """
        super(GrayscaleToRGB, self).__init__(params)
        self.inverse = params['GrayscaleToRGB_inverse'.lower()]
    
    def __call__(self, sample):
        image= sample['image']
        assert(image.shape[0] == 1 or image.shape[0] == 3)
        if(image.shape[0] == 1):
            sample['image'] = np.concatenate([image, image, image])
        return sample

