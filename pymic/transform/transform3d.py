# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import json
import math
import random
import numpy as np

from scipy import ndimage
from pymic.util.image_process import *
import matplotlib.pyplot as plt

class Rescale(object):
    """Rescale the image in a sample to a given size."""
    def __init__(self, params):
        """
        output_size (tuple/list or int): Desired output size. 
            If tuple/list, output_size should in the format of [D, H, W] or [H, W].
            Channel number is kept the same as the input. If D is None, the input image
            is only reslcaled in 2D.
            If int, the smallest axis is matched to output_size keeping 
            aspect ratio the same.
        """
        self.output_size = params["Rescale_output_size".lower()]
        self.inverse     = params["Rescale_inverse".lower()]
        assert isinstance(self.output_size, (int, list, tuple))

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim   = len(input_shape) - 1

        if isinstance(self.output_size, (list, tuple)):
            output_size = self.output_size
            if(output_size[0] is None):
                output_size[0] = input_shape[1]
            assert(len(output_size) == input_dim)
        else:
            min_edge = min(input_shape[1:])
            output_size = [self.output_size * input_shape[i+1] / min_edge for \
                            i in range(input_dim)]
        scale = [(output_size[i] + 0.0)/input_shape[1:][i] for i in range(input_dim)]
        scale = [1.0] + scale
        image_t = ndimage.interpolation.zoom(image, scale, order = 1)

        sample['image'] = image_t
        sample['Rescale_origin_shape'] = json.dumps(input_shape)
        if('label' in sample and sample['label'].shape[1:] == image.shape[1:]):
            label = sample['label']
            label = ndimage.interpolation.zoom(label, scale, order = 0)
            sample['label'] = label
        if('weight' in sample and sample['weight'].shape[1:] == image.shape[1:]):
            weight = sample['weight']
            weight = ndimage.interpolation.zoom(weight, scale, order = 1)
            sample['weight'] = weight
        
        return sample

    def inverse_transform_for_prediction(self, sample):
        ''' rescale sample['predict'] (5D or 4D) to the original spatial shape.
         assume batch size is 1, otherwise scale may be different for 
         different elemenets in the batch.

        origin_shape is a 4D or 3D vector as saved in __call__().'''
        if(isinstance(sample['Rescale_origin_shape'], list) or \
            isinstance(sample['Rescale_origin_shape'], tuple)):
            origin_shape = json.loads(sample['Rescale_origin_shape'][0])
        else:
            origin_shape = json.loads(sample['Rescale_origin_shape'])
        origin_dim   = len(origin_shape) - 1
        predict = sample['predict']
        input_shape = predict.shape
        scale = [(origin_shape[1:][i] + 0.0)/input_shape[2:][i] for \
                i in range(origin_dim)]
        scale = [1.0, 1.0] + scale

        output_predict = ndimage.interpolation.zoom(predict, scale, order = 1)
        sample['predict'] = output_predict
        return sample

class RandomFlip(object):
    """ random flip the image (shape [C, D, H, W] or [C, H, W]) """
    def __init__(self, params):
        """
            flip_depth (bool) : random flip along depth axis or not, only used for 3D images
            flip_height (bool): random flip along height axis or not
            flip_width (bool) : random flip along width axis or not
        """
        self.flip_depth  = params['RandomFlip_flip_depth'.lower()]
        self.flip_height = params['RandomFlip_flip_height'.lower()]
        self.flip_width  = params['RandomFlip_flip_width'.lower()]
        self.inverse = params['RandomFlip_inverse'.lower()]

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        flip_axis = []
        if(self.flip_width):
            if(random.random() > 0.5):
                flip_axis.append(-1)
        if(self.flip_height):
            if(random.random() > 0.5):
                flip_axis.append(-2)
        if(input_dim == 3 and self.flip_depth):
            if(random.random() > 0.5):
                flip_axis.append(-3)

        sample['RandomFlip_Param'] = json.dumps(flip_axis)
        if(len(flip_axis) > 0):
            # use .copy() to avoid negative strides of numpy array
            # current pytorch does not support negative strides
            image_t = np.flip(image, flip_axis).copy()
            sample['image'] = image_t
            if('label' in sample and sample['label'].shape[1:] == image.shape[1:]):
                sample['label'] = np.flip(sample['label'] , flip_axis).copy()
            if('weight' in sample and sample['weight'].shape[1:] == image.shape[1:]):
                sample['weight'] = np.flip(sample['weight'] , flip_axis).copy()
            
        return sample

    def  inverse_transform_for_prediction(self, sample):
        ''' flip sample['predict'] (5D or 4D) to the original direction.
         assume batch size is 1, otherwise flip parameter may be different for 
         different elemenets in the batch.

        flip_axis is a list as saved in __call__().'''
        if(isinstance(sample['RandomFlip_Param'], list) or \
            isinstance(sample['RandomFlip_Param'], tuple)):
            flip_axis = json.loads(sample['RandomFlip_Param'][0]) 
        else:
            flip_axis = json.loads(sample['RandomFlip_Param']) 
        if(len(flip_axis) > 0):
            sample['predict']  = np.flip(sample['predict'] , flip_axis).copy()
        return sample

class RandomRotate(object):
    """
    random rotate the image (shape [C, D, H, W] or [C, H, W]) 
    """
    def __init__(self, params): 
        """
        angle_range_d (tuple/list/None) : rorate angle range along depth axis (degree),
               only used for 3D images
        angle_range_h (tuple/list/None) : rorate angle range along height axis (degree)
        angle_range_w (tuple/list/None) : rorate angle range along width axis (degree)
        """
        self.angle_range_d  = params['RandomRotate_angle_range_d'.lower()]
        self.angle_range_h  = params['RandomRotate_angle_range_h'.lower()]
        self.angle_range_w  = params['RandomRotate_angle_range_w'.lower()]
        self.inverse = params['RandomRotate_inverse'.lower()]

    def __apply_transformation(self, image, transform_param_list, order = 1):
        """
        apply rotation transformation to an ND image
        Args:
            image (nd array): the input nd image
            transform_param_list (list): a list of roration angle and axes
            order (int): interpolation order
        """
        for angle, axes in transform_param_list:
            image = ndimage.rotate(image, angle, axes, reshape = False, order = order)
        return image

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        
        transform_param_list = []
        if(self.angle_range_d is not None):
            angle_d = np.random.uniform(self.angle_range_d[0], self.angle_range_d[1])
            transform_param_list.append([angle_d, (-1, -2)])
        if(input_dim == 3):
            if(self.angle_range_h is not None):
                angle_h = np.random.uniform(self.angle_range_h[0], self.angle_range_h[1])
                transform_param_list.append([angle_h, (-1, -3)])
            if(self.angle_range_w is not None):
                angle_w = np.random.uniform(self.angle_range_w[0], self.angle_range_w[1])
                transform_param_list.append([angle_w, (-2, -3)])
        assert(len(transform_param_list) > 0)

        sample['RandomRotate_Param'] = json.dumps(transform_param_list)
        image_t = self.__apply_transformation(image, transform_param_list, 1)
        sample['image'] = image_t
        if('label' in sample and sample['label'].shape[1:] == image.shape[1:]):
            sample['label'] = self.__apply_transformation(sample['label'] , 
                                transform_param_list, 0)
        if('weight' in sample and sample['weight'].shape[1:] == image.shape[1:]):
            sample['weight'] = self.__apply_transformation(sample['weight'] , 
                                transform_param_list, 1)
        return sample

    def  inverse_transform_for_prediction(self, sample):
        ''' rorate sample['predict'] (5D or 4D) to the original direction.
        assume batch size is 1, otherwise rotate parameter may be different for 
        different elemenets in the batch.

        transform_param_list is a list as saved in __call__().'''
        # get the paramters for invers transformation
        if(isinstance(sample['RandomRotate_Param'], list) or \
            isinstance(sample['RandomRotate_Param'], tuple)):
            transform_param_list = json.loads(sample['RandomRotate_Param'][0]) 
        else:
            transform_param_list = json.loads(sample['RandomRotate_Param']) 
        transform_param_list.reverse()
        for i in range(len(transform_param_list)):
            transform_param_list[i][0] = - transform_param_list[i][0]
        sample['predict'] = self.__apply_transformation(sample['predict'] , 
                                transform_param_list, 1)
        return sample

class Pad(object):
    """
    Pad the image (shape [C, D, H, W] or [C, H, W]) to an new spatial shape, 
    the real output size will be max(image_size, output_size)
    """
    def __init__(self, params):
        """
        output_size (tuple/list): the size along each spatial axis. 
        ceil_mode (bool): if true, the real output size is integer multiples of output_size.
        """
        self.output_size = params['Pad_output_size'.lower()]
        self.ceil_mode   = params['Pad_ceil_mode'.lower()]
        self.inverse = params['Pad_inverse'.lower()]

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        assert(len(self.output_size) == input_dim)
        if(self.ceil_mode):
            multiple = [int(math.ceil(float(input_shape[1+i])/self.output_size[i]))\
                for i in range(input_dim)]
            output_size = [multiple[i] * self.output_size[i] \
                for i in range(input_dim)]
        else:
            output_size = self.output_size
        margin = [max(0, output_size[i] - input_shape[1+i]) \
            for i in range(input_dim)]

        margin_lower = [int(margin[i] / 2) for i in range(input_dim)]
        margin_upper = [margin[i] - margin_lower[i] for i in range(input_dim)]
        sample['Pad_Param'] = json.dumps((margin_lower, margin_upper))

        pad = [(margin_lower[i], margin_upper[i]) for  i in range(input_dim)]
        pad = tuple([(0, 0)] + pad)
        image_t = np.pad(image, pad, 'reflect') if(max(margin) > 0) else image

        sample['image'] = image_t
        
        if('label' in sample and sample['label'].shape[1:] == image.shape[1:]):
            label = sample['label']
            label = np.pad(label, pad, 'reflect') if(max(margin) > 0) else label
            sample['label'] = label
        if('weight' in sample and sample['weight'].shape[1:] == image.shape[1:]):
            weight = sample['weight']
            weight = np.pad(weight, pad, 'reflect') if(max(margin) > 0) else weight
            sample['weight'] = weight
        return sample
    
    def inverse_transform_for_prediction(self, sample):
        ''' crop sample['predict'] (5D or 4D) to the original spatial shape.
         assume batch size is 1, otherwise scale may be different for 
         different elemenets in the batch.

        origin_shape is a 4D or 3D vector as saved in __call__().'''
        # raise ValueError("not implemented")
        if(isinstance(sample['Pad_Param'], list) or isinstance(sample['Pad_Param'], tuple)):
            params = json.loads(sample['Pad_Param'][0]) 
        else:
            params = json.loads(sample['Pad_Param']) 
        margin_lower = params[0]
        margin_upper = params[1]
        predict = sample['predict']
        if(isinstance(predict, tuple) or isinstance(predict, list)):
            output_predict = []
            for predict_i in predict:
                predict_shape = predict_i.shape
                crop_min = [0, 0] + margin_lower
                crop_max = [predict_shape[2:][i] - margin_upper[i] \
                    for i in range(len(margin_lower))]
                crop_max = list(predict_shape[:2]) + crop_max
                crop_predict = crop_ND_volume_with_bounding_box(predict_i, crop_min, crop_max)
                output_predict.append(crop_predict)
        else:
            predict_shape = predict.shape
            crop_min = [0, 0] + margin_lower
            crop_max = [predict_shape[2:][i] - margin_upper[i] \
                for i in range(len(margin_lower))]
            crop_max = list(predict_shape[:2]) + crop_max

            output_predict = crop_ND_volume_with_bounding_box(predict, crop_min, crop_max)
        sample['predict'] = output_predict
        return sample

class CropWithBoundingBox(object):
    """Crop the image (shape [C, D, H, W] or [C, H, W]) based on bounding box
    """
    def __init__(self, params):
        """
        start (None or tuple/list): The start index along each spatial axis.
            if None, calculate the start index automatically so that 
            the cropped region is centered at the non-zero region.
        output_size (None or tuple/list): Desired spatial output size.
            if None, set it as the size of bounding box of non-zero region 
        """
        self.start       = params['CropWithBoundingBox_start'.lower()]
        self.output_size = params['CropWithBoundingBox_output_size'.lower()]
        self.inverse = params['CropWithBoundingBox_inverse'.lower()]
        
    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim   = len(input_shape) - 1
        bb_min, bb_max = get_ND_bounding_box(image)
        bb_min, bb_max = bb_min[1:], bb_max[1:]
        if(self.start is None):
            if(self.output_size is None):
                crop_min, crop_max = bb_min, bb_max
            else:
                assert(len(self.output_size) == input_dim)
                crop_min = [int((bb_min[i] + bb_max[i] + 1)/2) - int(self.output_size[i]/2) \
                    for i in range(input_dim)]
                crop_min = [max(0, crop_min[i]) for i in range(input_dim)]
                crop_max = [crop_min[i] + self.output_size[i] for i in range(input_dim)]
        else:
            assert(len(self.start) == input_dim)
            crop_min = self.start
            if(self.output_size is None):
                assert(len(self.output_size) == input_dim)
                crop_max = [crop_min[i] + bb_max[i] - bb_min[i] \
                    for i in range(input_dim)]
            else:
                crop_max =  [crop_min[i] + self.output_size[i] for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max
        sample['CropWithBoundingBox_Param'] = json.dumps((input_shape, crop_min, crop_max))

        image_t = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
        sample['image'] = image_t
        
        if('label' in sample and sample['label'].shape[1:] == image.shape[1:]):
            label = sample['label']
            crop_max[0] = label.shape[0]
            label = crop_ND_volume_with_bounding_box(label, crop_min, crop_max)
            sample['label'] = label
        if('weight' in sample and sample['weight'].shape[1:] == image.shape[1:]):
            weight = sample['weight']
            crop_max[0] = weight.shape[0]
            weight = crop_ND_volume_with_bounding_box(weight, crop_min, crop_max)
            sample['weight'] = weight
        return sample

    def inverse_transform_for_prediction(self, sample):
        ''' rescale sample['predict'] (5D or 4D) to the original spatial shape.
         assume batch size is 1, otherwise scale may be different for 
         different elemenets in the batch.

        origin_shape is a 4D or 3D vector as saved in __call__().'''
        if(isinstance(sample['CropWithBoundingBox_Param'], list) or \
            isinstance(sample['CropWithBoundingBox_Param'], tuple)):
            params = json.loads(sample['CropWithBoundingBox_Param'][0]) 
        else:
            params = json.loads(sample['CropWithBoundingBox_Param']) 
        origin_shape = params[0]
        crop_min     = params[1]
        crop_max     = params[2]
        predict = sample['predict']
        if(isinstance(predict, tuple) or isinstance(predict, list)):
            output_predict = []
            for predict_i in predict:
                origin_shape     = list(predict_i.shape[:2]) + origin_shape[1:]
                output_predict_i = np.zeros(origin_shape, predict_i.dtype)
                crop_min = [0, 0] + crop_min[1:]
                crop_max = list(predict_i.shape[:2]) + crop_max[1:]
                output_predict_i = set_ND_volume_roi_with_bounding_box_range(output_predict_i,
                    crop_min, crop_max, predict_i)
                output_predict.append(output_predict_i)
        else:
            origin_shape   = list(predict.shape[:2]) + origin_shape[1:]
            output_predict = np.zeros(origin_shape, predict.dtype)
            crop_min = [0, 0] + crop_min[1:]
            crop_max = list(predict.shape[:2]) + crop_max[1:]
            output_predict = set_ND_volume_roi_with_bounding_box_range(output_predict,
                crop_min, crop_max, predict)

        sample['predict'] = output_predict
        return sample

class RandomCrop(object):
    """Randomly crop the input image (shape [C, D, H, W] or [C, H, W]) 
    """
    def __init__(self, params):
        """
        output_size (tuple or list): Desired output size [D, H, W] or [H, W].
            the output channel is the same as the input channel.
        foreground_focus (bool): If true, allow crop around the foreground.
        foreground_ratio (float): Specifying the probability of foreground 
            focus cropping when foreground_focus is true.
        mask_label (None, or tuple / list): Specifying the foreground labels for foreground 
            focus cropping
        """
        self.output_size = params['RandomCrop_output_size'.lower()]
        self.fg_focus    = params['RandomCrop_foreground_focus'.lower()]
        self.fg_ratio    = params['RandomCrop_foreground_ratio'.lower()]
        self.mask_label  = params['RandomCrop_mask_label'.lower()]
        self.inverse     = params['RandomCrop_inverse'.lower()]
        assert isinstance(self.output_size, (list, tuple))
        if(self.mask_label is not None):
            assert isinstance(self.mask_label, (list, tuple))

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim   = len(input_shape) - 1

        assert(input_dim == len(self.output_size))
        crop_margin = [input_shape[i + 1] - self.output_size[i]\
            for i in range(input_dim)]
        crop_min = [random.randint(0, item) for item in crop_margin]
        if(self.fg_focus and random.random() < self.fg_ratio):
            label = sample['label']
            mask  = np.zeros_like(label)
            for temp_lab in self.mask_label:
                mask = np.maximum(mask, label == temp_lab)
            if(mask.sum() == 0):
                bb_min = [0] * (input_dim + 1)
                bb_max = mask.shape
            else:
                bb_min, bb_max = get_ND_bounding_box(mask)
            bb_min, bb_max = bb_min[1:], bb_max[1:]
            crop_min = [random.randint(bb_min[i], bb_max[i]) - int(self.output_size[i]/2) \
                for i in range(input_dim)]
            crop_min = [max(0, item) for item in crop_min]
            crop_min = [min(crop_min[i], input_shape[i+1] - self.output_size[i]) \
                for i in range(input_dim)]

        crop_max = [crop_min[i] + self.output_size[i] \
            for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max
        sample['RandomCrop_Param'] = json.dumps((input_shape, crop_min, crop_max))

        image_t = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
        sample['image'] = image_t
        
        if('label' in sample and sample['label'].shape[1:] == image.shape[1:]):
            label = sample['label']
            crop_max[0] = label.shape[0]
            label = crop_ND_volume_with_bounding_box(label, crop_min, crop_max)
            sample['label'] = label
        if('weight' in sample and sample['weight'].shape[1:] == image.shape[1:]):
            weight = sample['weight']
            crop_max[0] = weight.shape[0]
            weight = crop_ND_volume_with_bounding_box(weight, crop_min, crop_max)
            sample['weight'] = weight
        return sample

    def inverse_transform_for_prediction(self, sample):
        ''' rescale sample['predict'] (5D or 4D) to the original spatial shape.
         assume batch size is 1, otherwise scale may be different for 
         different elemenets in the batch.

        origin_shape is a 4D or 3D vector as saved in __call__().'''
        if(isinstance(sample['RandomCrop_Param'], list) or \
            isinstance(sample['RandomCrop_Param'], tuple)):
            params = json.loads(sample['RandomCrop_Param'][0]) 
        else:
            params = json.loads(sample['RandomCrop_Param']) 
        origin_shape = params[0]
        crop_min     = params[1]
        crop_max     = params[2]
        predict = sample['predict']
        origin_shape   = list(predict.shape[:2]) + origin_shape[1:]
        output_predict = np.zeros(origin_shape, predict.dtype)
        crop_min = [0, 0] + crop_min[1:]
        crop_max = list(predict.shape[:2]) + crop_max[1:]
        output_predict = set_ND_volume_roi_with_bounding_box_range(output_predict,
            crop_min, crop_max, predict)

        sample['predict'] = output_predict
        return sample

class ChannelWiseGammaCorrection(object):
    """
    apply random gamma correction to each channel
    """
    def __init__(self, params):
        """
        (gamma_min, gamma_max) specify the range of gamma
        """
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
    
    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class ChannelWiseNormalize(object):
    """Nomralize the image (shape [C, D, H, W] or [C, H, W]) for each channel
    """
    def __init__(self, params):
        """
        mean (None or tuple/list): The mean values along each channel.
        std  (None or tuple/list): The std values along each channel.
            if mean and std are None, calculate them from non-zero region
        chns (None, or tuple/list): The list of channel indices
        zero_to_random (bool, or tuple/list or bool): indicate whether zero values
             in each channel is replaced  with random values.
        """
        self.mean = params['ChannelWiseNormalize_mean'.lower()]
        self.std  = params['ChannelWiseNormalize_std'.lower()]
        self.chns = params['ChannelWiseNormalize_channels'.lower()]
        self.zero_to_random = params['ChannelWiseNormalize_zero_to_random'.lower()]
        self.inverse = params['ChannelWiseNormalize_inverse'.lower()]

    def __call__(self, sample):
        image= sample['image']
        mask = image[0] > 0
        chns = self.chns
        if(chns is None):
            chns = range(image.shape[0])
        zero_to_random = self.zero_to_random
        if(isinstance(zero_to_random, bool)):
            zero_to_random = [zero_to_random]*len(chns)
        if(not(self.mean is None and self.std is None)):
            assert(len(self.mean) == len(self.std))
            assert(len(self.mean) == len(chns))
        for i in range(len(chns)):
            chn = chns[i]
            if(self.mean is None and self.std is None):
                pixels = image[chn][mask > 0]
                chn_mean = pixels.mean()
                chn_std  = pixels.std()
            else:
                chn_mean = self.mean[i]
                chn_std  = self.std[i]
            chn_norm = (image[chn] - chn_mean)/chn_std
            if(zero_to_random[i]):
                chn_random = np.random.normal(0, 1, size = chn_norm.shape)
                chn_norm[mask == 0] = chn_random[mask == 0]
            image[chn] = chn_norm

        sample['image'] = image
        return sample

    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class ChannelWiseThreshold(object):
    """Threshold the image (shape [C, D, H, W] or [C, H, W]) for each channel
    """
    def __init__(self, params):
        """
        threshold (tuple/list): The threshold value along each channel.
        """
        self.threshold = params['ChannelWiseThreshold_threshold'.lower()]
        self.inverse = params['ChannelWiseThreshold_inverse'.lower()]

    def __call__(self, sample):
        image= sample['image']
        for chn in range(image.shape[0]):
            mask = np.asarray(image[chn] > self.threshold[chn], image.dtype)
            image[chn] = mask * (image[chn] - self.threshold[chn])

        sample['image'] = image
        return sample

    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class ChannelWiseThresholdWithNormalize(object):
    """Threshold the image (shape [C, D, H, W] or [C, H, W]) for each channel
       and then normalize the image based on remaining pixels
    """
    def __init__(self, params):
        """
        threshold_lower (tuple/list/None): The lower threshold value along each channel.
        threshold_upper (typle/list/None): The upper threshold value along each channel.
        mean_std_mode (bool): If true, nomalize the image based on mean and std values,
            and pixels values outside the threshold value are replaced random number.
            If false, use the min and max values for normalization.
        """
        self.threshold_lower = params['ChannelWiseThresholdWithNormalize_threshold_lower'.lower()]
        self.threshold_upper = params['ChannelWiseThresholdWithNormalize_threshold_upper'.lower()]
        self.mean_std_mode   = params['ChannelWiseThresholdWithNormalize_mean_std_mode'.lower()]
        self.inverse = params['ChannelWiseThresholdWithNormalize_inverse'.lower()]

    def __call__(self, sample):
        image= sample['image']
        for chn in range(image.shape[0]):
            v0 = self.threshold_lower[chn]
            v1 = self.threshold_upper[chn]
            if(self.mean_std_mode == True):
                mask = np.ones_like(image[chn])
                if(v0 is not None):
                    mask = mask * np.asarray(image[chn] > v0)
                if(v1 is not None):
                    mask = mask * np.asarray(image[chn] < v1)
                pixels   = image[chn][mask > 0]
                chn_mean = pixels.mean()
                chn_std  = pixels.std()
                chn_norm = (image[chn] - chn_mean)/chn_std
                chn_random = np.random.normal(0, 1, size = chn_norm.shape)
                chn_norm[mask == 0] = chn_random[mask == 0]
                image[chn] = chn_norm
            else:
                img_chn = image[chn]
                if(v0 is not None):
                    img_chn[img_chn < v0] = v0
                    min_value = v0 
                else:
                    min_value = img_chn.min()
                if(v1 is not None):
                    img_chn[img_chn > v1] = v1 
                    max_value = img_chn.max() 
                else:
                    max_value = img_chn.max() 
                img_chn = (img_chn - min_value) / (max_value - min_value)
                image[chn] = img_chn
        sample['image'] = image
        return sample

class ReduceLabelDim(object):
    """
    remove the first dimension of label tensor
    """
    def __init__(self, params):
        self.inverse = params['ReduceLabelDim_inverse'.lower()]
    
    def __call__(self, sample):
        label = sample['label']
        label_converted = label[0]
        sample['label'] = label_converted
        return sample
    
    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class LabelConvert(object):
    """ Convert a list of labels to another list"""
    def __init__(self, params):
        """
        source_list (tuple/list): A list of labels to be converted
        target_list (tuple/list): The target label list
        """
        self.source_list = params['LabelConvert_source_list'.lower()]
        self.target_list = params['LabelConvert_target_list'.lower()]
        self.inverse = params['LabelConvert_inverse'.lower()]
        assert(len(self.source_list) == len(self.target_list))
    
    def __call__(self, sample):
        label = sample['label']
        label_converted = convert_label(label, self.source_list, self.target_list)
        sample['label'] = label_converted
        return sample
    
    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class LabelConvertNonzero(object):
    """ Convert label into binary (nonzero as 1)"""
    def __init__(self, params):
        self.inverse = params['LabelConvertNonzero_inverse'.lower()]
    
    def __call__(self, sample):
        label = sample['label']
        label_converted = np.asarray(label > 0, np.uint8)
        sample['label'] = label_converted
        return sample
    
    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class LabelToProbability(object):
    """Convert one-channel label map to one-hot multi-channel probability map"""

    def __init__(self, params): 
        """
        class_num (int): the class number in the label map
        """
        self.class_num = params['LabelToProbability_class_num'.lower()]
        self.inverse   = params['LabelToProbability_inverse'.lower()]
    
    def __call__(self, sample):
        label = sample['label'][0]
        label_prob = []
        for i in range(self.class_num):
            temp_prob = label == i*np.ones_like(label)
            label_prob.append(temp_prob)
        label_prob = np.asarray(label_prob, np.float32)
   
        sample['label_prob'] = label_prob
        return sample
    
    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

TransformDict = {
    'ChannelWiseGammaCorrection': ChannelWiseGammaCorrection,
    'ChannelWiseNormalize': ChannelWiseNormalize,
    'ChannelWiseThreshold': ChannelWiseThreshold,
    'ChannelWiseThresholdWithNormalize': ChannelWiseThresholdWithNormalize,
    'CropWithBoundingBox': CropWithBoundingBox,
    'LabelConvert': LabelConvert,
    'LabelConvertNonzero': LabelConvertNonzero,
    'LabelToProbability': LabelToProbability,
    'RandomCrop': RandomCrop,
    'RandomFlip': RandomFlip,
    'RandomRotate': RandomRotate,
    'ReduceLabelDim': ReduceLabelDim,
    'Rescale': Rescale,
    'Pad': Pad,
}
