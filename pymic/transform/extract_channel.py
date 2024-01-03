# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import json
import math
import random
import numpy as np
from scipy import ndimage
from pymic import TaskType
from pymic.transform.abstract_transform import AbstractTransform
from pymic.util.image_process import *


class ExtractChannel(AbstractTransform):
    """ Random flip the image. The shape is [C, D, H, W] or [C, H, W].
    
    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `RandomFlip_flip_depth`: (bool) 
        Random flip along depth axis or not, only used for 3D images.
    :param `RandomFlip_flip_height`: (bool) Random flip along height axis or not.
    :param `RandomFlip_flip_width`: (bool) Random flip along width axis or not.    
    :param `RandomFlip_inverse`: (optional, bool) Is inverse transform needed for inference.
        Default is `True`.
    """
    def __init__(self, params):
        super(ExtractChannel, self).__init__(params)
        self.channels  = params['ExtractChannel_channels'.lower()]
        self.inverse   = params.get('ExtractChannel_inverse'.lower(), False)

    def __call__(self, sample):
        image = sample['image']
        image_extract = []
        for i in self.channels:
            image_extract.append(image[i])
        sample['image'] = np.asarray(image_extract)            
        return sample
