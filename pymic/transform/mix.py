# -*- coding: utf-8 -*-
from __future__ import print_function, division
import copy 
import json
import math
import random
import numpy as np
from pymic.transform.abstract_transform import AbstractTransform
from pymic.util.image_process import *
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb


class CopyPaste(AbstractTransform):
    """
    In-painting of an input image, used for self-supervised learning 
    """
    def __init__(self, params):
        super(CopyPaste, self).__init__(params)
        self.inverse  = params.get('CopyPaste_inverse'.lower(), False)
        self.block_range = params.get('CopyPaste_block_range'.lower(), (1, 6))
        self.block_size_min = params.get('CopyPaste_block_size_min'.lower(), None)
        self.block_size_max = params.get('CopyPaste_block_size_max'.lower(), None)

    def __call__(self, sample):
        image= sample['image']       
        img_shape = image.shape
        img_dim = len(img_shape) - 1
        assert(img_dim == 2 or img_dim == 3)

        if(self.block_size_min is None):
            block_size_min = [img_shape[1+i]//6 for i in range(img_dim)]
        elif(isinstance(self.block_size_min, int)):
            block_size_min = [self.block_size_min] * img_dim
        else:
            assert(len(self.block_size_min) == img_dim)
            block_size_min = self.block_size_min

        if(self.block_size_max is None):
            block_size_max = [img_shape[1+i]//3 for i in range(img_dim)]
        elif(isinstance(self.block_size_min, int)):
            block_size_max = [self.block_size_max] * img_dim
        else:
            assert(len(self.block_size_max) == img_dim)
            block_size_max = self.block_size_max
        block_num = random.randint(self.block_range[0], self.block_range[1])

        for n in range(block_num):
            block_size = [random.randint(block_size_min[i], block_size_max[i]) \
                for i in range(img_dim)]    
            coord_min = [random.randint(3, img_shape[1+i] - block_size[i] - 3) \
                for i in range(img_dim)]
            if(img_dim == 2):
                random_block = np.random.rand(img_shape[0], block_size[0], block_size[1])
                image[:, coord_min[0]:coord_min[0] + block_size[0], 
                         coord_min[1]:coord_min[1] + block_size[1]] = random_block
            else:
                random_block = np.random.rand(img_shape[0], block_size[0], 
                                              block_size[1], block_size[2])
                image[:, coord_min[0]:coord_min[0] + block_size[0], 
                         coord_min[1]:coord_min[1] + block_size[1],
                         coord_min[2]:coord_min[2] + block_size[2]] = random_block
        sample['image'] = image
        return sample
    
class PatchMix(AbstractTransform):
    """
    In-painting of an input image, used for self-supervised learning 
    """
    def __init__(self, params):
        super(PatchMix, self).__init__(params)
        self.inverse        = params.get('PatchMix_inverse'.lower(), False)
        self.threshold      = params.get('PatchMix_threshold'.lower(), 0)
        self.crop_size      = params.get('PatchMix_crop_size'.lower(), [64, 128, 128])
        self.fg_cls_num     = params.get('PatchMix_cls_num'.lower(), [4, 40])
        self.patch_num_range= params.get('PatchMix_patch_range'.lower(), [4, 40])
        self.patch_size_min = params.get('PatchMix_patch_size_min'.lower(), [4, 4, 4])
        self.patch_size_max = params.get('PatchMix_patch_size_max'.lower(), [20, 40, 40])

    def __call__(self, sample):
        x0 = self._random_crop_and_flip(sample)
        x1 = self._random_crop_and_flip(sample)
        C, D, H, W = x0.shape
        # generate mask 
        fg_mask = np.zeros_like(x0, np.uint8)
        patch_num = random.randint(self.patch_num_range[0], self.patch_num_range[1])
        for patch in range(patch_num):
            d = random.randint(self.patch_size_min[0], self.patch_size_max[0]) 
            h = random.randint(self.patch_size_min[1], self.patch_size_max[1]) 
            w = random.randint(self.patch_size_min[2], self.patch_size_max[2]) 
            d_c = random.randint(0, D)
            h_c = random.randint(0, H)
            w_c = random.randint(0, W)
            d0, d1 = max(0, d_c - d // 2), min(D, d_c + d // 2)
            h0, h1 = max(0, h_c - h // 2), min(H, h_c + h // 2)
            w0, w1 = max(0, w_c - w // 2), min(W, w_c + w // 2)
            temp_m = np.ones([C, d1-d0, h1-h0, w1-w0]) * random.randint(1, self.fg_cls_num)
            fg_mask[:, d0:d1, h0:h1, w0:w1] = temp_m
        fg_w   = fg_mask * 1.0 / self.fg_cls_num
        x_fuse = fg_w*x0 + (1.0 - fg_w)*x1 # x1 is used as background
        
        sample['image'] = x_fuse
        sample['label'] = fg_mask
        return sample
    
    def _random_crop_and_flip(self, sample):
        image       = sample['image']
        input_dim   = len(image.shape) - 1
        assert(input_dim == 3)
        C, D, H, W  = image.shape

        half_size = [x // 2 for x in self.crop_size]
        dc = random.randint(half_size[0], D - half_size[0])
        image2d = image[0, dc, :, :]
        mask2d  = np.zeros_like(image2d)
        mask2d[half_size[1]:H+1-half_size[1], half_size[2]:W+1-half_size[2]] = \
            np.ones([H-self.crop_size[1]+1, W-self.crop_size[2]+1])
        if('label' in sample):
            temp_mask = sample['label'][0, dc, :, :] > 0
            mask2d = temp_mask * mask2d
        elif(self.threshold is not None):
            temp_mask = image2d > self.threshold
            se         = np.ones([3,3])
            temp_mask  = ndimage.binary_opening(temp_mask, se, iterations = 2)
            temp_mask  = get_largest_k_components(temp_mask, 1)
            mask2d = temp_mask * mask2d

        indices = np.where(mask2d)
        n = random.randint(0, len(indices[0])-1)
        center = [indices[i][n] for i in range(2)]
        crop_min = [dc - half_size[0], center[0]-half_size[1], center[1] - half_size[2]]
        crop_max = [crop_min[i] + self.crop_size[i] for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = [C] + crop_max
        x = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)

        flip_axis = []
        if(random.random() > 0.5):
            flip_axis.append(-1)
        if(random.random() > 0.5):
            flip_axis.append(-2)
        if(random.random() > 0.5):
            flip_axis.append(-3)
        if(len(flip_axis) > 0):
            x = np.flip(x, flip_axis).copy()

        if(x.shape[1] == 63):
            print("crop shape == 63", x.shape)
            print(sample['names'])
            print(image.shape, crop_min, crop_max)
        return x