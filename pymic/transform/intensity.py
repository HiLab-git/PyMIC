# -*- coding: utf-8 -*-
from __future__ import print_function, division
import copy 
import itertools
import json
import math
import random
import numpy as np
from scipy import ndimage
from skimage import exposure
from pymic.transform.abstract_transform import AbstractTransform
from pymic.util.image_process import *
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


class IntensityClip(AbstractTransform):
    """
    Clip the intensity for input image

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `IntensityClip_channels`: (list) A list of int for specifying the channels.
    :param `IntensityClip_lower`: (list) The lower bound for clip in each channel.
    :param `IntensityClip_upper`: (list) The upper bound for clip in each channel.
    :param `IntensityClip_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `False`.
    """
    def __init__(self, params):
        super(IntensityClip, self).__init__(params)
        self.channels =  params['IntensityClip_channels'.lower()]
        self.lower = params.get('IntensityClip_lower'.lower(), None)
        self.upper = params.get('IntensityClip_upper'.lower(), None)
        self.perct = params.get('IntensityClip_percentile_mode'.lower(), False)
        self.inverse   = params.get('IntensityClip_inverse'.lower(), False)
    
    def __call__(self, sample):
        image = sample['image']
        lower = self.lower if self.lower is not None else [None] * len(self.channels)
        upper = self.upper if self.upper is not None else [None] * len(self.channels)
        for chn in self.channels:
            lower_c, upper_c = lower[chn], upper[chn]
            if(lower_c is None):
                lower_c = np.percentile(image[chn], 0.05)
            elif(self.perct):
                lower_c = np.percentile(image[chn], lower_c)
            if(upper_c is None):
                upper_c = np.percentile(image[chn], 99.95)
            elif(self.perct):
                upper_c = np.percentile(image[chn], upper_c)
            image[chn] = np.clip(image[chn], lower_c, upper_c)
        sample['image'] = image
        return sample

class HistEqual(AbstractTransform):
    """
    Histogram equalization. Note that the output will be in the range of [0, 1].

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `HistEqual_channels`: (list) A list of int for specifying the channels.
    :param `HistEqual_bin`: (int) The number of bins.
    :param `HistEqual_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `False`.
    """
    def __init__(self, params):
        super(HistEqual, self).__init__(params)
        self.channels =  params.get('HistEqual_channels'.lower(), None)
        # self.min = params.get('HistEqual_min'.lower(), None)
        # self.max = params.get('HistEqual_max'.lower(), None)
        self.bin = params.get('HistEqual_bin'.lower(), 2000)
        self.inverse   = params.get('HistEqual_inverse'.lower(), False)
    
    def __call__(self, sample):
        image = sample['image']
        C = image.shape[0] 
        chns = range(C) if self.channels is None else self.channels 
        for i in range(len(chns)):
            c = chns[i]
            image[c] = exposure.equalize_hist(image[c],nbins= self.bin)
        sample['image'] = image
        return sample

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
        self.channels =  params.get('GammaCorrection_channels'.lower(), None)
        self.gamma_min = params.get('GammaCorrection_gamma_min'.lower(), 0.7)
        self.gamma_max = params.get('GammaCorrection_gamma_max'.lower(), 1.5)
        self.flip_prob = params.get('GammaCorrection_intensity_flip_probability'.lower(), 0.0)
        self.prob      = params.get('GammaCorrection_probability'.lower(), 0.5)
        self.inverse   = params.get('GammaCorrection_inverse'.lower(), False)
    
    def __call__(self, sample):
        image= sample['image']
        if(self.channels is None):
            self.channels = range(image.shape[0])
        for chn in self.channels:
            if(np.random.uniform() > self.prob):
                continue
            gamma_c = random.random() * (self.gamma_max - self.gamma_min) + self.gamma_min
            img_c = image[chn]
            v_min = img_c.min()
            v_max = img_c.max()
            if(v_min < v_max):
                img_c = (img_c - v_min)/(v_max - v_min)
                if(np.random.uniform() < self.flip_prob):
                    img_c = 1.0 - img_c
                img_c = np.power(img_c, gamma_c)*(v_max - v_min) + v_min
            image[chn] = img_c

        sample['image'] = image
        return sample

def gaussian_noise(image, std_min, std_max,):
    """
    The input has a shape of [C, D, H, W] or [D, H, W]. 
    In the former case, volume-level noise will be added.
    In the latter case, slice-level noise will ba added. 
    """
    v_min  = image.min()
    v_max  = image.max()
    std    = random.random() * (std_max - std_min) + std_min
    noise  = np.random.normal(0, std, image.shape)
    out = image + noise
    out = np.clip(out, v_min, v_max)
    return out 

def gaussian_blur(image, sigma_min, sigma_max):
    sigma = random.random() * (sigma_max - sigma_min) + sigma_min
    out   = ndimage.gaussian_filter(image, sigma, order = 0)
    return out 

def gaussian_sharpen(image, sigma_min, sigma_max, alpha = 10.0):
    blurred = gaussian_blur(image, sigma_min, sigma_max)
    out = image + (image - blurred) * alpha
    return out

def window_level_augment(image, offset = 0.1):
    v_min  = image.min()
    v_max  = image.max()
    margin = (v_max - v_min) * offset
    v0 = random.uniform(v_min - margin, v_min + margin)
    v1 = random.uniform(v_max - margin, v_max + margin)
    out = np.clip((image - v0) / (v1 - v0), 0, 1)
    return out  

def gamma_correction(image, gamma_min, gamma_max):
    v_min  = image.min()
    v_max  = image.max()
    if(v_min < v_max):
        image = (image - v_min)/(v_max - v_min)
        gamma = random.random() * (gamma_max - gamma_min) + gamma_min
        image = np.power(image, gamma)*(v_max - v_min) + v_min
    return image  

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
        self.channels = params.get('GaussianNoise_channels'.lower(), None)
        self.std_min  = params.get('GaussianNoise_std_min'.lower(), 0.02)
        self.std_max  = params.get('GaussianNoise_std_max'.lower(), 0.1)
        self.prob     = params.get('GaussianNoise_probability'.lower(), 0.5)
        self.inverse  = params.get('GaussianNoise_inverse'.lower(), False)
    
    def __call__(self, sample):
        image = sample['image']
        if(self.channels is None):
            self.channels = range(image.shape[0])
        for chn in self.channels:
            if(np.random.uniform() < self.prob):
                image[chn] = gaussian_noise(image[chn], self.std_min, self.std_max)
        sample['image'] = image
        return sample

def adaptive_contrast_adjust(image, p0=0.1, p1=99.9):
    v_min = image.min()
    v_max = image.max()
    v0 = np.percentile(image, p0)
    v1 = np.percentile(image, p1)
    mask_l = image < v0 
    mask_m = (image >= v0) * (image <= v1)
    mask_u = image > v1
    image[mask_l] = (image[mask_l] - v_min)  * 0.1 / (v0 - v_min)
    image[mask_m] = (image[mask_m] - v0) / (v1 - v0)*0.8 + 0.1
    image[mask_u] = 0.9 + 0.1 * (image[mask_u] - v1) / (v_max - v1)
    return image 

class AdaptiveContrastAdjust(AbstractTransform):
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
        super(AdaptiveContrastAdjust, self).__init__(params)
        self.channels = params.get('AdaptiveContrastAdjust_channels'.lower(), None)
        self.p0       = params.get('AdaptiveContrastAdjust_percent_lower'.lower(), 2)
        self.p1       = params.get('AdaptiveContrastAdjust_percent_upper'.lower(), 98)
        self.prob     = params.get('AdaptiveContrastAdjust_probability'.lower(), 0.5)
        self.inverse  = params.get('AdaptiveContrastAdjust_inverse'.lower(), False)
    
    def __call__(self, sample):
        image = sample['image'] * 1.0
        if(self.channels is None):
            self.channels = range(image.shape[0])
        for chn in self.channels:
            if(np.random.uniform() < self.prob):
                image[chn] = adaptive_contrast_adjust(image[chn], self.p0, self.p1)
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
    
class NonLinearTransform(AbstractTransform):
    def __init__(self, params):
        super(NonLinearTransform, self).__init__(params)
        self.channels = params.get('NonLinearTransform_channels'.lower(), None)
        self.prob     = params.get('NonLinearTransform_probability'.lower(), 0.5)
        self.inverse  = params.get('NonLinearTransform_inverse'.lower(), False)
        self.block_range = params.get('NonLinearTransform_block_range'.lower(), None)
        self.block_size  = params.get('NonLinearTransform_block_size'.lower(), [4, 8, 8])
        
    
    def apply_nonlinear_transform(self, img):
        """
        the input img should be normlized to [0, 1]"""
        points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
        xvals, yvals = bezier_curve(points, nTimes=10000)
        if random.random() < 0.5: # Half chance to get flip
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        
        img = np.interp(img, xvals, yvals)
        return img

    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample

        image = sample['image']
        img_shape = image.shape 
        img_dim = len(img_shape) - 1
        channels = self.channels if self.channels is not None else range(image.shape[0])
        for chn in channels:
            # normalize the image intensity to [0, 1] before the non-linear tranform
            img_c = image[chn]
            v_min, v_max = img_c.min(), img_c.max()
            if(v_min < v_max):
                img_c = (img_c - v_min)/(v_max - v_min)
                if(self.block_range is None): # apply non-linear transform to the entire image
                    img_c = self.apply_nonlinear_transform(img_c)
                else:  # non-linear transform to random blocks
                    img_c_sr = copy.deepcopy(img_c)
                    for n in range(self.block_range[0], self.block_range[1]): 
                        coord_min = [random.randint(0, img_shape[1+i] - self.block_size[i]) \
                            for i in range(img_dim)]
                        window = img_c_sr[coord_min[0]:coord_min[0] + self.block_size[0], 
                                    coord_min[1]:coord_min[1] + self.block_size[1],
                                    coord_min[2]:coord_min[2] + self.block_size[2]]
                        img_c[coord_min[0]:coord_min[0] + self.block_size[0], 
                            coord_min[1]:coord_min[1] + self.block_size[1],
                            coord_min[2]:coord_min[2] + self.block_size[2]] = \
                            self.apply_nonlinear_transform(window)
                image[chn] = img_c * (v_max - v_min) + v_min
        sample['image']  = image 
        return sample

class LocalShuffling(AbstractTransform):
    """
    local pixel shuffling of an input image, used for self-supervised learning 
    """
    def __init__(self, params):
        super(LocalShuffling, self).__init__(params)
        self.inverse  = params.get('LocalShuffling_inverse'.lower(), False)
        self.prob     = params.get('LocalShuffling_probability'.lower(), 0.5)
        self.block_range = params.get('LocalShuffling_block_range'.lower(), [40, 80])
        self.block_size  = params.get('LocalShuffling_block_size'.lower(), [4, 8, 8])

    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample

        image= sample['image']       
        img_shape = image.shape
        img_dim = len(img_shape) - 1
        assert(img_dim == 2 or img_dim == 3)
        img_out = copy.deepcopy(image)
    
        block_num = random.randint(self.block_range[0], self.block_range[1])

        for n in range(block_num):
            coord_min = [random.randint(0, img_shape[1+i] - self.block_size[i]) \
                for i in range(img_dim)]
            if(img_dim == 2):
                window = image[:, coord_min[0]:coord_min[0] + self.block_size[0], 
                                  coord_min[1]:coord_min[1] + self.block_size[1]]
                n_pixels = self.block_size[0] * self.block_size[1]
            else:
                window = image[:, coord_min[0]:coord_min[0] + self.block_size[0], 
                                  coord_min[1]:coord_min[1] + self.block_size[1],
                                  coord_min[2]:coord_min[2] + self.block_size[2]]
                n_pixels = self.block_size[0] * self.block_size[1] * self.block_size[2]
            window = np.reshape(window, [-1, n_pixels])
            np.random.shuffle(np.transpose(window))
            window = np.transpose(window)
            if(img_dim == 2):
                window = np.reshape(window, [-1, self.block_size[0], self.block_size[1]])
                img_out[:, coord_min[0]:coord_min[0] + self.block_size[0], 
                           coord_min[1]:coord_min[1] + self.block_size[1]] = window
            else:
                window = np.reshape(window, [-1, self.block_size[0], self.block_size[1], self.block_size[2]])
                img_out[:, coord_min[0]:coord_min[0] + self.block_size[0], 
                           coord_min[1]:coord_min[1] + self.block_size[1],
                           coord_min[2]:coord_min[2] + self.block_size[2]] = window
        sample['image'] = img_out
        return sample

class InPainting(AbstractTransform):
    """
    In-painting of an input image, used for self-supervised learning 
    """
    def __init__(self, params):
        super(InPainting, self).__init__(params)
        self.inverse  = params.get('InPainting_inverse'.lower(), False)
        self.prob     = params.get('InPainting_probability'.lower(), 0.5)
        self.block_range = params.get('InPainting_block_range'.lower(), (20, 40))
        self.block_size  = params.get('InPainting_block_size'.lower(), [4, 8, 8])
       
    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample

        image= sample['image']       
        img_shape = image.shape
        img_dim = len(img_shape) - 1
        assert(img_dim == 2 or img_dim == 3)

        block_num = random.randint(self.block_range[0], self.block_range[1])

        for n in range(block_num):    
            coord_min = [random.randint(3, img_shape[1+i] - self.block_size[i] - 3) \
                for i in range(img_dim)]
            if(img_dim == 2):
                random_block = np.random.rand(img_shape[0], self.block_size[0], self.block_size[1]) * 2 -1 
                image[:, coord_min[0]:coord_min[0] + self.block_size[0], 
                         coord_min[1]:coord_min[1] + self.block_size[1]] = random_block
            else:
                random_block = np.random.rand(img_shape[0], self.block_size[0], 
                                              self.block_size[1], self.block_size[2]) * 2 -1
                image[:, coord_min[0]:coord_min[0] + self.block_size[0], 
                         coord_min[1]:coord_min[1] + self.block_size[1],
                         coord_min[2]:coord_min[2] + self.block_size[2]] = random_block
        sample['image'] = image
        return sample

class OutPainting(AbstractTransform):
    """
    Out-painting of an input image, used for self-supervised learning 
    """
    def __init__(self, params):
        super(OutPainting, self).__init__(params)
        self.inverse  = params.get('OutPainting_inverse'.lower(), False)
        self.prob     = params.get('OutPainting_probability'.lower(), 0.5)
        self.block_range = params.get('OutPainting_block_range'.lower(), (2, 8))
        self.block_size  = params.get('OutPainting_block_size'.lower(), None)

    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample

        image= sample['image']       
        img_shape = image.shape
        img_dim = len(img_shape) - 1
        assert(img_dim == 2 or img_dim == 3)
        img_out = np.random.rand(*img_shape) * 2 -1

        if(self.block_size is None):
            margin = [16, 32, 32]
            block_size = [img_shape[1+i] - margin[i] for i in range(img_dim)]
        else:
            assert(len(self.block_size) == img_dim)
            block_size = self.block_size

        block_num = random.randint(self.block_range[0], self.block_range[1])

        for n in range(block_num):
            coord_min = [random.randint(3, img_shape[1+i] - block_size[i] - 3) \
                for i in range(img_dim)]
            if(img_dim == 2):
                img_out[:, coord_min[0]:coord_min[0] + block_size[0], 
                           coord_min[1]:coord_min[1] + block_size[1]] = \
                    image[:, coord_min[0]:coord_min[0] + block_size[0], 
                             coord_min[1]:coord_min[1] + block_size[1]]
            else:
                img_out[:, coord_min[0]:coord_min[0] + block_size[0], 
                           coord_min[1]:coord_min[1] + block_size[1],
                           coord_min[2]:coord_min[2] + block_size[2]] = \
                    image[:, coord_min[0]:coord_min[0] + block_size[0], 
                             coord_min[1]:coord_min[1] + block_size[1],
                             coord_min[2]:coord_min[2] + block_size[2]] 
        sample['image'] = img_out
        return sample

class InOutPainting(AbstractTransform):
    """
    Apply in-painting or out-patining randomly. They are mutually exclusive.
    """
    def __init__(self, params):
        super(InOutPainting, self).__init__(params)
        self.inverse  = params.get('InOutPainting_inverse'.lower(), False)
        self.prob     = params.get('InOutPainting_probability'.lower(), 0.5)
        self.in_prob  = params.get('InPainting_probability'.lower(), 0.5)
        params['InPainting_probability'.lower()]  = 1.0
        params['OutPainting_probability'.lower()] = 1.0
        self.inpaint  = InPainting(params)
        self.outpaint = OutPainting(params)

    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample
        if(random.random() < self.in_prob):
            sample = self.inpaint(sample)
        else:
            sample = self.outpaint(sample)
        return sample

class PatchSwaping(AbstractTransform):
    """
    Apply patch swaping for context restoration in self-supervised learning. 
    Reference: Liang Chen et al., Self-supervised learning for medical image analysis
        using image context restoration, Medical Image Analysis, 2019. 
    """
    def __init__(self, params):
        super(PatchSwaping, self).__init__(params)
        self.block_range = params.get('PatchSwaping_block_range'.lower(), (10, 20))
        self.block_size  = params.get('PatchSwaping_block_size'.lower(), [8, 16, 16])
        self.inverse  = params.get('PatchSwaping_inverse'.lower(), False)

    def __call__(self, sample): 
        image= sample['image']       
        img_shape = image.shape
        img_dim = len(img_shape) - 1
        assert(img_dim == 2 or img_dim == 3)
        img_out = copy.deepcopy(image)
        
        block_num = random.randint(self.block_range[0], self.block_range[1])
        for t in range(block_num):
            pos_a0 = [random.randint(0, img_shape[-3+i] - self.block_size[i]) for i in range(img_dim)]
            pos_b0 = [random.randint(0, img_shape[-3+i] - self.block_size[i]) for i in range(img_dim)]
            pos_a1 = [pos_a0[i] + self.block_size[i] for i in range(img_dim)]
            pos_b1 = [pos_b0[i] + self.block_size[i] for i in range(img_dim)]
            img_out[:, pos_a0[0]:pos_a1[0], pos_a0[1]:pos_a1[1], pos_a0[2]:pos_a1[2]] = \
                image[:, pos_b0[0]:pos_b1[0], pos_b0[1]:pos_b1[1], pos_b0[2]:pos_b1[2]]
            img_out[:, pos_b0[0]:pos_b1[0], pos_b0[1]:pos_b1[1], pos_b0[2]:pos_b1[2]] = \
                image[:, pos_a0[0]:pos_a1[0], pos_a0[1]:pos_a1[1], pos_a0[2]:pos_a1[2]]

        sample['image'] = img_out
        sample['label'] = image
        return sample

class MaskedImageModeling(AbstractTransform):
    """
    Apply masking for context restoration in self-supervised learning. 
    Reference:  Zekai Chen et al., Masked Image Modeling Advances 3D Medical Image Analysis, 
    WACV, 2023 . 
    """
    def __init__(self, params):
        super(MaskedImageModeling, self).__init__(params)
        self.ratio       = params.get('MaskedImageModeling_ratio'.lower(), 0.45)
        self.block_size  = params.get('MaskedImageModeling_block_size'.lower(), [8, 16, 16])
        self.inverse  = params.get('MaskedImageModeling_inverse'.lower(), False)

    def __call__(self, sample): 
        image= sample['image']       
        C, D, H, W = image.shape
        img_out = copy.deepcopy(image)
        
        block = np.zeros([C] + list(self.block_size))
        for d in range(0, D, self.block_size[0]):
            d1 = d + self.block_size[0] 
            if d1 > D:
                continue 
            for h in range(0, H, self.block_size[1]):
                h1 = h + self.block_size[1]
                if  h1 > H:
                    continue
                for w in range(0, W, self.block_size[2]):
                    w1 = w + self.block_size[2]
                    if w1 > W:
                        continue 
                    r = random.random()
                    if ( r < self.ratio):
                        img_out[:, d:d1, h:h1, w:w1] = block

        sample['image'] = img_out
        sample['label'] = image
        return sample