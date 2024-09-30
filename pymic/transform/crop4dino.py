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
from pymic.transform.crop import CenterCrop
from pymic.transform.intensity import *
from pymic.util.image_process import *

class Crop4Dino(CenterCrop):
    """
    Randomly crop an volume into two views with augmentation. This is used for
    self-supervised pretraining such as DeSD.

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `DualViewCrop_output_size`: (list/tuple) Desired output size [D, H, W].
        The output channel is the same as the input channel. 
    :param `DualViewCrop_scale_lower_bound`: (list/tuple) Lower bound of the range of scale
        for each dimension. e.g. (1.0, 0.5, 0.5).
    param `DualViewCrop_scale_upper_bound`: (list/tuple) Upper bound of the range of scale
        for each dimension. e.g. (1.0, 2.0, 2.0).
    :param `DualViewCrop_inverse`: (optional, bool) Is inverse transform needed for inference.
        Default is `False`. Currently, the inverse transform is not supported, and 
        this transform is assumed to be used only during training stage. 
    """
    def __init__(self, params):
        self.output_size = params['Crop4Dino_output_size'.lower()]
        self.scale_lower = params['Crop4Dino_resize_lower_bound'.lower()]
        self.scale_upper = params['Crop4Dino_resize_upper_bound'.lower()]
        self.prob        = params.get('Crop4Dino_resize_prob'.lower(), 0.5)
        self.noise_std_range   = params.get('Crop4Dino_noise_std_range'.lower(), (0.05, 0.1))
        self.blur_sigma_range  = params.get('Crop4Dino_blur_sigma_range'.lower(), (1.0, 3.0))
        self.gamma_range      = params.get('Crop4Dino_gamma_range'.lower(), (0.75, 1.25))
        self.inverse     = params.get('Crop4Dino_inverse'.lower(), False)
        self.task        = params['Task'.lower()]
        assert isinstance(self.output_size, (list, tuple))
        assert isinstance(self.scale_lower, (list, tuple))
        assert isinstance(self.scale_upper, (list, tuple))
        
    def __call__(self, sample):
        image = sample['image']
        channel, input_size = image.shape[0], image.shape[1:]
        input_dim   = len(input_size)
        assert(input_dim == len(self.output_size))

        # # center crop first
        # crop_size   = self.output_size
        # crop_margin = [input_size[i] - crop_size[i] for i in range(input_dim)]
        # crop_min = [int(item/2) for item in crop_margin]
        # crop_max = [crop_min[i] + crop_size[i] for i in range(input_dim)]
        # crop_min = [0] + crop_min
        # crop_max = [channel] + crop_max
        # crop0    = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)

        crop_num = 2
        crop_img = []
        for crop_i in range(crop_num):
            resize = random.random() < self.prob
            if(resize):
                scale = [self.scale_lower[i] + (self.scale_upper[i] - self.scale_lower[i]) * random.random() \
                    for i in range(input_dim)]
                crop_size = [int(self.output_size[i] * scale[i])  for i in range(input_dim)]
            else:
                crop_size = self.output_size

            crop_margin = [input_size[i] - crop_size[i] for i in range(input_dim)]
            pad_image   = min(crop_margin) < 0
            if(pad_image): # pad the image if necessary
                pad_size = [max(0, -crop_margin[i]) for  i in range(input_dim)]
                pad_lower = [int(pad_size[i] / 2) for i in range(input_dim)]
                pad_upper = [pad_size[i] - pad_lower[i] for i in range(input_dim)]
                pad = [(pad_lower[i], pad_upper[i]) for  i in range(input_dim)]
                pad = tuple([(0, 0)] + pad)
                image = np.pad(image, pad, 'reflect')
                crop_margin = [max(0, crop_margin[i]) for i in range(input_dim)]


            crop_min = [random.randint(0, item) for item in crop_margin]
            crop_max = [crop_min[i] + crop_size[i] for i in range(input_dim)]
            crop_min = [0] + crop_min
            crop_max = [channel] + crop_max

            crop_out = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
            if(resize):
                scale = [(self.output_size[i] + 0.0)/crop_size[i] for i in range(input_dim)]
                scale = [1.0] + scale
                crop_out = ndimage.interpolation.zoom(crop_out, scale, order = 1)
            
            # add intensity augmentation
            C = crop_out.shape[0]
            for c in range(C):
                if(random.random() < 0.8):
                    crop_out[c] = gaussian_noise(crop_out[c], self.noise_std_range[0], self.noise_std_range[1])

                if(random.uniform(0, 1) < 0.5):
                    crop_out[c] = gaussian_blur(crop_out[c], self.blur_sigma_range[0], self.blur_sigma_range[1])
                else:
                    alpha = random.uniform(0.0, 2.0)
                    crop_out[c] = gaussian_sharpen(crop_out[c], self.blur_sigma_range[0], self.blur_sigma_range[1], alpha)
                if(random.random() < 0.8):    
                    crop_out[c] = gamma_correction(crop_out[c], self.gamma_range[0], self.gamma_range[1])
                if(random.random() < 0.8):    
                    crop_out[c] = window_level_augment(crop_out[c])
            crop_img.append(crop_out)
        sample['image'] = crop_img
        return sample

    def __call__backup(self, sample):
        image = sample['image']
        channel, input_size = image.shape[0], image.shape[1:]
        input_dim   = len(input_size)
        assert(input_dim == len(self.output_size))

        # center crop first
        crop_size   = self.output_size
        crop_margin = [input_size[i] - crop_size[i] for i in range(input_dim)]
        crop_min = [int(item/2) for item in crop_margin]
        crop_max = [crop_min[i] + crop_size[i] for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = [channel] + crop_max
        crop0    = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)

        # crop_num = 2
        # crop_img = []
        # for crop_i in range(crop_num):
            # get another resized crop size
        resize = random.random() < self.prob
        if(resize):
            scale = [self.scale_lower[i] + (self.scale_upper[i] - self.scale_lower[i]) * random.random() \
                for i in range(input_dim)]
            crop_size = [int(self.output_size[i] * scale[i])  for i in range(input_dim)]
        else:
            crop_size = self.output_size

        crop_margin = [input_size[i] - crop_size[i] for i in range(input_dim)]
        pad_image   = min(crop_margin) < 0
        if(pad_image): # pad the image if necessary
            pad_size = [max(0, -crop_margin[i]) for  i in range(input_dim)]
            pad_lower = [int(pad_size[i] / 2) for i in range(input_dim)]
            pad_upper = [pad_size[i] - pad_lower[i] for i in range(input_dim)]
            pad = [(pad_lower[i], pad_upper[i]) for  i in range(input_dim)]
            pad = tuple([(0, 0)] + pad)
            image = np.pad(image, pad, 'reflect')
            crop_margin = [max(0, crop_margin[i]) for i in range(input_dim)]


        crop_min = [random.randint(0, item) for item in crop_margin]
        crop_max = [crop_min[i] + crop_size[i] for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = [channel] + crop_max

        crop_out = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
        if(resize):
            scale = [(self.output_size[i] + 0.0)/crop_size[i] for i in range(input_dim)]
            scale = [1.0] + scale
            crop_out = ndimage.interpolation.zoom(crop_out, scale, order = 1)
            # crop_img.append(crop_out)
        crop_img = [crop0, crop_out]    
            # add intensity augmentation
            # image_t = gaussian_noise(image_t, self.noise_std_range[0], self.noise_std_range[1], 0.8)
            # image_t = gaussian_blur(image_t, self.blur_sigma_range[0], self.blur_sigma_range[1], 0.8)
            # image_t = brightness_multiplicative(image_t, self.inten_multi_range[0],  self.inten_multi_range[1], 0.8)
            # image_t = brightness_additive(image_t, self.inten_add_range[0], self.inten_add_range[1], 0.8)
            # image_t = contrast_augment(image_t, self.contrast_f_range[0], self.contrast_f_range[1], 0.8)
            # image_t = gamma_correction(image_t, self.gamma_range[0], self.gamma_range[1], 0.8)
        sample['image'] = crop_img
        return sample
