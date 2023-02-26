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

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

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
    
class NonLinearTransform(AbstractTransform):
    def __init__(self, params):
        super(NonLinearTransform, self).__init__(params)
        self.inverse  = params.get('NonLinearTransform_inverse'.lower(), False)
        self.prob     = params.get('NonLinearTransform_probability'.lower(), 0.5)
    
    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample

        image= sample['image'] 
        points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
        xvals, yvals = bezier_curve(points, nTimes=100000)
        if random.random() < 0.5: # Half change to get flip
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        image = np.interp(image, xvals, yvals)
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
        self.block_range = params.get('LocalShuffling_block_range'.lower(), (5000, 10000))
        self.block_size_min = params.get('LocalShuffling_block_size_min'.lower(), None)
        self.block_size_max = params.get('LocalShuffling_block_size_max'.lower(), None)

    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample

        image= sample['image']       
        img_shape = image.shape
        img_dim = len(img_shape) - 1
        assert(img_dim == 2 or img_dim == 3)
        img_out = copy.deepcopy(image)
        if(self.block_size_min is None):
            block_size_min = [2] * img_dim
        elif(isinstance(self.block_size_min, int)):
            block_size_min = [self.block_size_min] * img_dim
        else:
            assert(len(self.block_size_min) == img_dim)
            block_size_min = self.block_size_min

        if(self.block_size_max is None):
            block_size_max = [img_shape[1+i]//10 for i in range(img_dim)]
        elif(isinstance(self.block_size_min, int)):
            block_size_max = [self.block_size_max] * img_dim
        else:
            assert(len(self.block_size_max) == img_dim)
            block_size_max = self.block_size_max
        block_num = random.randint(self.block_range[0], self.block_range[1])

        for n in range(block_num):
            block_size = [random.randint(block_size_min[i], block_size_max[i]) \
                for i in range(img_dim)]    
            coord_min = [random.randint(0, img_shape[1+i] - block_size[i]) \
                for i in range(img_dim)]
            if(img_dim == 2):
                window = image[:, coord_min[0]:coord_min[0] + block_size[0], 
                                  coord_min[1]:coord_min[1] + block_size[1]]
                n_pixels = block_size[0] * block_size[1]
            else:
                window = image[:, coord_min[0]:coord_min[0] + block_size[0], 
                                  coord_min[1]:coord_min[1] + block_size[1],
                                  coord_min[2]:coord_min[2] + block_size[2]]
                n_pixels = block_size[0] * block_size[1] * block_size[2]
            window = np.reshape(window, [-1, n_pixels])
            np.random.shuffle(np.transpose(window))
            window = np.transpose(window)
            if(img_dim == 2):
                window = np.reshape(window, [-1, block_size[0], block_size[1]])
                img_out[:, coord_min[0]:coord_min[0] + block_size[0], 
                           coord_min[1]:coord_min[1] + block_size[1]] = window
            else:
                window = np.reshape(window, [-1, block_size[0], block_size[1], block_size[2]])
                img_out[:, coord_min[0]:coord_min[0] + block_size[0], 
                           coord_min[1]:coord_min[1] + block_size[1],
                           coord_min[2]:coord_min[2] + block_size[2]] = window
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
        self.block_range = params.get('InPainting_block_range'.lower(), (1, 6))
        self.block_size_min = params.get('InPainting_block_size_min'.lower(), None)
        self.block_size_max = params.get('InPainting_block_size_max'.lower(), None)

    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample

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

class OutPainting(AbstractTransform):
    """
    Out-painting of an input image, used for self-supervised learning 
    """
    def __init__(self, params):
        super(OutPainting, self).__init__(params)
        self.inverse  = params.get('OutPainting_inverse'.lower(), False)
        self.prob     = params.get('OutPainting_probability'.lower(), 0.5)
        self.block_range = params.get('OutPainting_block_range'.lower(), (1, 6))
        self.block_size_min = params.get('OutPainting_block_size_min'.lower(), None)
        self.block_size_max = params.get('OutPainting_block_size_max'.lower(), None)

    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample

        image= sample['image']       
        img_shape = image.shape
        img_dim = len(img_shape) - 1
        assert(img_dim == 2 or img_dim == 3)
        img_out = np.random.rand(*img_shape)

        if(self.block_size_min is None):
            block_size_min = [img_shape[1+i] - 4 * img_shape[1+i]//7 for i in range(img_dim)]
        elif(isinstance(self.block_size_min, int)):
            block_size_min = [self.block_size_min] * img_dim
        else:
            assert(len(self.block_size_min) == img_dim)
            block_size_min = self.block_size_min

        if(self.block_size_max is None):
            block_size_max = [img_shape[1+i] - 3 * img_shape[1+i]//7 for i in range(img_dim)]
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
        params['InPainting_probability']  = 1.0
        params['outPainting_probability'] = 1.0
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