# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch

import json
import math
import random
import numpy as np
from imops import crop_to_box
from typing import *
from scipy import ndimage
from pymic import TaskType
from pymic.transform.abstract_transform import AbstractTransform
from pymic.transform.crop import CenterCrop
from pymic.util.image_process import *
from pymic.transform.intensity import *


def random_resized_crop(image, output_size, scale_lower, scale_upper):
    input_size = image.shape
    scale = [scale_lower[i] + (scale_upper[i] - scale_lower[i]) * random.random() \
                for i in range(3)]
    crop_size = [min(int(output_size[i] * scale[i]), input_size[1+i])  for i in range(3)]
    crop_margin = [input_size[1+i] - crop_size[i] for i in range(3)]
    crop_min = [random.randint(0, item) for item in crop_margin]
    crop_max = [crop_min[i] + crop_size[i] for i in range(3)]
    crop_min = [0] + crop_min
    crop_max = [input_size[0]] + crop_max

    image_t = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
    scale = [(output_size[i] + 0.0)/crop_size[i] for i in range(3)]
    scale = [1.0] + scale
    image_t = ndimage.interpolation.zoom(image_t, scale, order = 1)
    return image_t 

def random_flip(image):
    flip_axis = []
    if(random.random() > 0.5):
        flip_axis.append(-1)
    if(random.random() > 0.5):
        flip_axis.append(-2)
    if(random.random() > 0.5):
        flip_axis.append(-3)
    if(len(flip_axis) > 0):
        image = np.flip(image , flip_axis)
    return image 


class Crop4VolumeFusion(AbstractTransform):
    """
    Randomly crop an volume into two views with augmentation. This is used for
    self-supervised pretraining in Vox2vec.

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `Crop4VolumeFusion_output_size`: (list/tuple) Desired output size [D, H, W].
        The output channel is the same as the input channel. 
    :param `Crop4VolumeFusion_rescale_lower_bound`: (list/tuple) Lower bound of the range of scale
        for each dimension. e.g. (1.0, 0.5, 0.5).
    param `Crop4VolumeFusion_rescale_upper_bound`: (list/tuple) Upper bound of the range of scale
        for each dimension. e.g. (1.0, 2.0, 2.0).
    :param `Crop4VolumeFusion_augentation_mode`: (optional, int) The mode for augmentation of cropped volume.
        0: no spatial or intensity augmentatin.
        1: intensity augmentation only 
`       2: spatial augmentation only 
        3: Both intensity and spatial augmentation (default).  
    """
    def __init__(self, params):
        self.output_size = params['Crop4VolumeFusion_output_size'.lower()]
        self.scale_lower = params.get('Crop4VolumeFusion_rescale_lower_bound'.lower(), [0.7, 0.7, 0.7])
        self.scale_upper = params.get('Crop4VolumeFusion_rescale_upper_bound'.lower(), [1.5, 1.5, 1.5])
        self.aug_mode    = params.get('Crop4VolumeFusion_augentation_mode'.lower(), 3)
        self.task        = params['Task'.lower()]
        assert isinstance(self.output_size, (list, tuple))
        
    def __call__(self, sample):
        image = sample['image']
        channel, input_size = image.shape[0], image.shape[1:]
        input_dim = len(input_size)
        assert channel == 1
        assert(input_dim == len(self.output_size))

        if(self.aug_mode == 0 or self.aug_mode == 1):
            self.scale_lower = [1.0, 1.0, 1.0]
            self.scale_upper = [1.0, 1.0, 1.0]
        patch_1 = random_resized_crop(image, self.output_size, self.scale_lower, self.scale_upper)
        patch_2 = random_resized_crop(image, self.output_size, self.scale_lower, self.scale_upper)
        if(self.aug_mode > 1):
            patch_1 = random_flip(patch_1)
            patch_2 = random_flip(patch_2)
        if(self.aug_mode == 1 or self.aug_mode == 3):
            p0, p1  = random.uniform(0.1, 2.0), random.uniform(98, 99.9)
            patch_1 = adaptive_contrast_adjust(patch_1, p0, p1)
            patch_1 = gamma_correction(patch_1, 0.7, 1.5)

            p0, p1  = random.uniform(0.1, 2.0), random.uniform(98, 99.9)
            patch_2 = adaptive_contrast_adjust(patch_2, p0, p1)
            patch_2 = gamma_correction(patch_2, 0.7, 1.5)

            if(random.random() < 0.25):
                patch_1 = 1.0 - patch_1
                patch_2 = 1.0 - patch_2

        sample['image'] = patch_1, patch_2
        return sample

class VolumeFusion(AbstractTransform):
    """
    Randomly crop an volume into two views with augmentation. This is used for
    self-supervised pretraining in Vox2vec.

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
        self.cls_num  = params.get('VolumeFusion_cls_num'.lower(), 5)
        self.ratio    = params.get('VolumeFusion_foreground_ratio'.lower(), 0.7)
        self.size_min = params.get('VolumeFusion_patchsize_min'.lower(), [8, 8, 8])
        self.size_max = params.get('VolumeFusion_patchsize_max'.lower(), [32, 32, 32])
        self.task     = params['Task'.lower()]
        
    def __call__(self, sample):
        K = self.cls_num - 1
        image1, image2 = sample['image']
        C, D, H, W =  image1.shape
        db = random.randint(self.size_min[0], self.size_max[0])
        hb = random.randint(self.size_min[1], self.size_max[1])
        wb = random.randint(self.size_min[2], self.size_max[2])
        d_offset = random.randint(0, D % db)
        h_offset = random.randint(0, H % hb)
        w_offset = random.randint(0, W % wb)
        d_n = D // db 
        h_n = H // hb 
        w_n = W // wb 
        Nblock = d_n * h_n * w_n 
        Nfg = int(d_n * h_n * w_n * self.ratio)
        list_fg  = [1] * Nfg + [0] * (Nblock - Nfg) 
        random.shuffle(list_fg)
        mask = np.zeros([1, D, H, W], np.uint8)
        for d in range(d_n):
            for h in range(h_n):
                for w in range(w_n):
                    d0, h0, w0 = d*db + d_offset, h*hb + h_offset, w*wb + w_offset 
                    d1, h1, w1 = d0 + db, h0 + hb, w0 + wb
                    idx = d*h_n*w_n + h*w_n + w 
                    if(list_fg[idx]> 0):
                        cls_k = random.randint(1, K)
                        mask[:, d0:d1, h0:h1, w0:w1] = cls_k
        alpha   = mask * 1.0 / K
        x_fuse = alpha*image1 + (1.0 - alpha)*image2     
        sample['image'] = x_fuse 
        sample['label'] = mask 
        return sample 

class VolumeFusionShuffle(AbstractTransform):
    """
    Randomly crop an volume into two views with augmentation. This is used for
    self-supervised pretraining in Vox2vec.

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
        self.cls_num  = params.get('VolumeFusionShuffle_cls_num'.lower(), 5)
        self.ratio    = params.get('VolumeFusionShuffle_foreground_ratio'.lower(), 0.7)
        self.size_min = params.get('VolumeFusionShuffle_patchsize_min'.lower(), [8, 8, 8])
        self.size_max = params.get('VolumeFusionShuffle_patchsize_max'.lower(), [32, 32, 32])
        self.task     = params['Task'.lower()]
        
    def __call__(self, sample):
        K = self.cls_num - 1
        image1, image2 = sample['image']
        C, D, H, W =  image1.shape
        x_fuse = image2 * 1.0
        mask = np.zeros([1, D, H, W], np.uint8)
        db = random.randint(self.size_min[0], self.size_max[0])
        hb = random.randint(self.size_min[1], self.size_max[1])
        wb = random.randint(self.size_min[2], self.size_max[2])
        d_offset = random.randint(0, D % db)
        h_offset = random.randint(0, H % hb)
        w_offset = random.randint(0, W % wb)
        d_n = D // db 
        h_n = H // hb 
        w_n = W // wb 
        coord_list_source = []
        for di in range(d_n):
            for hi in range(h_n):
                for wi in range(w_n):
                    coord_list_source.append([di, hi, wi])
        coord_list_target = copy.deepcopy(coord_list_source)
        random.shuffle(coord_list_source)
        random.shuffle(coord_list_target)
        for i in range(int(len(coord_list_source)*self.ratio)):
            ds_l = d_offset + db * coord_list_source[i][0]
            hs_l = h_offset + hb * coord_list_source[i][1]    
            ws_l = w_offset + wb * coord_list_source[i][2]    
            dt_l = d_offset + db * coord_list_target[i][0]
            ht_l = h_offset + hb * coord_list_target[i][1]    
            wt_l = w_offset + wb * coord_list_target[i][2]  
            s_crop = image1[:, ds_l:ds_l+db, hs_l:hs_l+hb, ws_l:ws_l+wb]
            t_crop = image2[:, dt_l:dt_l+db, ht_l:ht_l+hb, wt_l:wt_l+wb]
            fg_m = random.randint(1, K)
            fg_w = fg_m / (K + 0.0)
            x_fuse[:, dt_l:dt_l+db, ht_l:ht_l+hb, wt_l:wt_l+wb] = t_crop * (1.0 - fg_w) + s_crop * fg_w
            mask[0, dt_l:dt_l+db, ht_l:ht_l+hb, wt_l:wt_l+wb] = \
                np.ones([1, db, hb, wb]) * fg_m    
        sample['image'] = x_fuse 
        sample['label'] = mask 
        return sample 

