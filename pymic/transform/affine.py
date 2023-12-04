# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import json
import math
import random
import numpy as np
from skimage import transform
from scipy import ndimage
from pymic import TaskType
from pymic.transform.abstract_transform import AbstractTransform
from pymic.util.image_process import *

class Affine(AbstractTransform):
    """
    Apply Affine Transform to an ND volume in the x-y plane. 
    Input shape should be [C, D, H, W] or [C, H, W].

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `Affine_scale_range`: (list or tuple) The range for scaling, e.g., (0.5, 2.0)
    :param `Affine_shear_range`: (list or tuple) The range for shearing angle, e.g., (0, 30)
    :param `Affine_rotate_range`: (list or tuple) The range for rotation, e.g., (-45, 45)
    :param `Affine_output_size`: (None, list or tuple of length 2) The output size after affine transformation.
        For 3D volumes, as we only apply affine transformation in x-y plane, the output slice
        number will be the same as the input slice number, so only the output height and width 
        need to be given here,  e.g., (H, W). By default (`None`), the output size will be the
        same as the input size.
    """
    def __init__(self, params):
        super(Affine, self).__init__(params)
        self.scale_range = params['Affine_scale_range'.lower()]
        self.shear_range = params['Affine_shear_range'.lower()]
        self.rotat_range = params['Affine_rotate_range'.lower()]
        self.output_shape= params.get('Affine_output_size'.lower(), None)
        self.inverse     = params.get('Affine_inverse'.lower(), True)

    def _get_affine_param(self, sample, output_shape):
        """
        output_shape should only has two dimensions, e.g., (H, W)
        """
        input_shape = sample['image'].shape
        input_dim   = len(input_shape) - 1
        assert(len(output_shape) >=2)

        in_y,  in_x  = input_shape[-2:]
        out_y, out_x = output_shape[-2:]
        points = [[0, out_y], 
                  [0, 0],
                  [out_x, 0],
                  [out_x, out_y]]

        sx  =  random.random() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        sy  =  random.random() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        shx = (random.random() * (self.shear_range[1] - self.shear_range[0]) + self.shear_range[0]) * 3.14159/180
        shy = (random.random() * (self.shear_range[1] - self.shear_range[0]) + self.shear_range[0]) * 3.14159/180
        rot = (random.random() * (self.rotat_range[1] - self.rotat_range[0]) + self.rotat_range[0]) * 3.14159/180
        # get affine transform parameters
        new_points = []
        for p in points:
            x = sx * p[0] * (math.cos(rot) + math.tan(shy) * math.sin(rot)) - \
                sy * p[1] * (math.tan(shx) * math.cos(rot) + math.sin(rot))
            y = sx * p[0] * (math.sin(rot) - math.tan(shy) * math.cos(rot)) - \
                sy * p[1] * (math.tan(shx) * math.sin(rot) - math.cos(rot))
            new_points.append([x,y])
        bb_min = np.array(new_points).min(axis = 0)
        bb_max = np.array(new_points).max(axis = 0)
        bbx, bby = int(bb_max[0] - bb_min[0]), int(bb_max[1] - bb_min[1])
        # transform the points to the image coordinate
        margin_x = in_x - bbx 
        margin_y = in_y - bby 
        p0x = random.random() * margin_x if margin_x > 0 else margin_x / 2
        p0y = random.random() * margin_y if margin_y > 0 else margin_y / 2
        dst = [[new_points[i][0] - bb_min[0] + p0x, new_points[i][1] - bb_min[1] + p0y] \
            for i in range(3)]

        tform = transform.AffineTransform()   
        tform.estimate(np.array(points[:3]), np.array(dst))
        # to do: need to find a solution to save the affine transform matrix
        # Use the matplotlib.transforms.Affine2D function to generate transform matrices, 
        # and the scipy.ndimage.warp function to warp images using the transform matrices. 
        # The skimage AffineTransform shear functionality is weird, 
        # and the scipy affine_transform function for warping images swaps the X and Y axes.
        # sample['Affine_Param'] = json.dumps((input_shape, tform["matrix"]))
        return sample, tform

    def _apply_affine_to_ND_volume(self, image, output_shape, tform, order = 3):
        """
        output_shape should only has two dimensions, e.g., (H, W)
        """
        dim = len(image.shape) - 1
        if(dim == 2):
            C, H, W = image.shape
            output = np.zeros([C] + output_shape)
            for c in range(C):
                output[c] = ndimage.affine_transform(image[c], tform, 
                                output_shape = output_shape, mode='mirror', order = order)
        elif(dim == 3):
            C, D, H, W = image.shape 
            output = np.zeros([C, D] + output_shape)
            for c in range(C):
                for d in range(D):
                    output[c,d] = ndimage.affine_transform(image[c,d], tform, 
                                output_shape = output_shape, mode='mirror', order = order)
        return output 

    def __call__(self, sample):
        image = sample['image']
        input_shape = sample['image'].shape
        output_shape= input_shape if self.output_shape is None else self.output_shape
        aff_out_shape = output_shape[-2:]
        sample, tform = self._get_affine_param(sample, aff_out_shape)
        image_t = self._apply_affine_to_ND_volume(image, aff_out_shape, tform)
        sample['image'] = image_t
        
        if('label' in sample and \
            self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
            label = sample['label']
            label = self._apply_affine_to_ND_volume(label, aff_out_shape, tform, order = 0)
            sample['label'] = label
        if('pixel_weight' in sample and \
            self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
            weight = sample['pixel_weight']
            weight = self._apply_affine_to_ND_volume(weight, aff_out_shape, tform)
            sample['pixel_weight'] = weight
        return sample

    def _get_param_for_inverse_transform(self, sample):
        if(isinstance(sample['Affine_Param'], list) or \
            isinstance(sample['Affine_Param'], tuple)):
            params = json.loads(sample['Affine_Param'][0]) 
        else:
            params = json.loads(sample['Affine_Param']) 
        return params

    # def inverse_transform_for_prediction(self, sample):
    #     params = self._get_param_for_inverse_transform(sample)
    #     origin_shape = params[0]
    #     tform        = params[1]
        
    #     predict = sample['predict']
    #     if(isinstance(predict, tuple) or isinstance(predict, list)):
    #         output_predict = []
    #         for predict_i in predict:
    #             aff_out_shape = origin_shape[-2:]
    #             output_predict_i = self._apply_affine_to_ND_volume(predict_i, 
    #                 aff_out_shape, tform.inverse)
    #             output_predict.append(output_predict_i)
    #     else:
    #         aff_out_shape = origin_shape[-2:]
    #         output_predict = self._apply_affine_to_ND_volume(predict, aff_out_shape, tform.inverse)
        
    #     sample['predict'] = output_predict
    #     return sample
