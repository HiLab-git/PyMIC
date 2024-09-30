# -*- coding: utf-8 -*-
from __future__ import print_function, division
import random
import numpy as np
from pymic.transform.abstract_transform import AbstractTransform
from pymic.util.image_process import *
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

def random_resized_crop(x, output_shape):
    img_shape = x.shape[1:]
    ratio = [img_shape[i] / output_shape[i] for i in range(3)]
    r_max = [min(ratio[i], 1.25) for i in range(3)]
    r_min = (0.8, 0.8, 0.8)
    scale = [r_min[i] + random.random() * (r_max[i] - r_min[i]) for i in range(3)]
    crop_size = [int(output_shape[i] * scale[i])  for i in range(3)]

    bb_min = [random.randint(0, img_shape[i] - crop_size[i]) for i in range(3)]
    bb_max = [bb_min[i] + crop_size[i] for i in range(3)]
    bb_min = [0] + bb_min
    bb_max = [x.shape[0]] + bb_max
    crop_volume = crop_ND_volume_with_bounding_box(x, bb_min, bb_max) 

    scale = [(output_shape[i] + 0.0)/crop_size[i] for i in range(3)]
    scale = [1.0] + scale
    y = ndimage.interpolation.zoom(crop_volume, scale, order = 1)
    return y 

def nonlinear_transform(x):
    v_min = np.min(x)
    v_max = np.max(x)
    x = (x - v_min)/(v_max - v_min)
    a = random.random() * 0.7 + 0.15
    b = random.random() * 0.7 + 0.15
    alpha = b / a 
    beta  = (1 - b) / (1 - a)
    if(alpha < 1.0 ):
        y = np.maximum(alpha*x, beta*x + 1 - beta)
    else:
        y = np.minimum(alpha*x, beta*x + 1 - beta)
    if(random.random() < 0.5):
        y = 1.0 - y 
    y = y * (v_max - v_min) + v_min
    return y 

def random_flip(x):
    flip_axis = []
    if(random.random() > 0.5):
        flip_axis.append(-1)
    if(random.random() > 0.5):
        flip_axis.append(-2)
    if(random.random() > 0.5):
        flip_axis.append(-3)

    if(len(flip_axis) > 0):
        # use .copy() to avoid negative strides of numpy array
        # current pytorch does not support negative strides
        y = np.flip(x, flip_axis).copy()
    else:
        y = x 
    return y 


class VolumeFusion(AbstractTransform):
    """
    fusing two subvolumes of an image, used for self-supervised learning 
    """
    def __init__(self, params):
        super(VolumeFusion, self).__init__(params)
        self.inverse        = params.get('VolumeFusion_inverse'.lower(), False)
        self.crop_size      = params.get('VolumeFusion_crop_size'.lower(), [64, 128, 128])
        self.block_range    = params.get('VolumeFusion_block_range'.lower(), [20, 40])
        self.size_min = params.get('VolumeFusion_size_min'.lower(), [8, 16, 16])
        self.size_max = params.get('VolumeFusion_size_max'.lower(), [16, 32, 32])

    def __call__(self, sample):
        x  = sample['image']
        x0 = random_resized_crop(x, self.crop_size)
        x1 = random_resized_crop(x, self.crop_size)
        x0 = random_flip(x0)
        x1 = random_flip(x1)
        # nonlinear transform 
        x0a = nonlinear_transform(x0)
        x0b = nonlinear_transform(x0)
        x1  = nonlinear_transform(x1)
       
        D, H, W = x0.shape[1:]
        mask  = np.zeros_like(x0, np.uint8)
        p_num = random.randint(self.block_range[0], self.block_range[1])
        for i in range(p_num):
            d = random.randint(self.size_min[0], self.size_max[0])
            h = random.randint(self.size_min[1], self.size_max[1])
            w = random.randint(self.size_min[2], self.size_max[2])
            dc = random.randint(0, D - 1)
            hc = random.randint(0, H - 1)
            wc = random.randint(0, W - 1)
            d0 = dc - d // 2
            h0 = hc - h // 2
            w0 = wc - w // 2
            d1 = min(D, d0 + d)
            h1 = min(H, h0 + h)
            w1 = min(W, w0 + w)
            d0, h0, w0 = max(0, d0), max(0, h0), max(0, w0) 
            temp_m = np.ones([d1 - d0, h1 - h0, w1 - w0])
            if(random.random() < 0.5):
                temp_m = temp_m * 2
            mask[:, d0:d1, h0:h1, w0:w1] = temp_m
    
        mask1 = np.asarray(mask == 1, np.uint8)
        mask2 = np.asarray(mask == 2, np.uint8)
        y = x0a * (1.0 - mask1) + x0b * mask1
        y = y * (1.0 - mask2) + x1 * mask2
        sample['image'] = y
        sample['label'] = mask
        return sample
