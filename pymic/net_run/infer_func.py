# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import math
import torch
import numpy as np

def volume_infer(image, net, class_num, sliding_window = False,
        window_size = None, window_stride  = None, output_num = 1):
    """
    Obtain net(image)
    sampling the image with mini_patch_shape and use the patch as input of network
    if mini_patch_shape is None, use the whole image as input of the network 
    image : the input tensor on cuda
    device: device name
    net   : the network on cuda
    class_num : number of class for segmentation
    mini_patch_shape: the shape of an inference patch
    output_num: number of outputs, when >1, the network obtains a list of output array

    return outputs: a pytorch tensor or a list of tensors after inference
    """
    if(sliding_window is False):
        outputs = net(image)
    else:
        outputs = volume_infer_by_patch(image, net, class_num, 
            window_size, window_stride, output_num)
    if(isinstance(outputs, tuple) or isinstance(outputs, list)):
        outputs = outputs[:output_num] if output_num > 1 else outputs[0]
    return outputs

def volume_infer_by_patch(image, net, class_num,
        window_size, window_stride, output_num):
    '''
        Test one image with sliding windows
    '''
    img_full_shape = image.shape
    img_shape = list(img_full_shape[2:])
    img_dim   = len(img_shape)
    if(img_dim != 2 and img_dim !=3):
        raise ValueError("volume_infer_by_patch only supports 2D and 3D images")

    for i in range(img_dim):
        if window_size[i] is None:
            window_size[i]  = img_shape[i]
        if window_stride[i] is None:
            window_stride[i] = window_size[i]

    crop_start_list  = []
    for w in range(0, img_shape[-1], window_stride[-1]):
        w_min = min(w, img_shape[-1] - window_size[-1])
        for h in range(0, img_shape[-2], window_stride[-2]):
            h_min = min(h, img_shape[-2] - window_size[-2])
            if(img_dim == 2):
                crop_start_list.append([h_min, w_min])
            else:
                for d in range(0, img_shape[0], window_stride[0]):
                    d_min = min(d, img_shape[0] - window_size[0])
                    crop_start_list.append([d_min, h_min, w_min])
    
    output_shape = [img_full_shape[0], class_num] + img_shape
    output_list  = [torch.zeros(output_shape).cuda() for i in range(output_num)]
    pred_num_arr = torch.zeros(output_shape).cuda()
    mask_shape = [img_full_shape[0], class_num] + window_size
    temp_mask    = torch.ones(mask_shape).cuda()
    
    for c0 in crop_start_list:
        c1 = [c0[i] + window_size[i] for i in range(img_dim)]
        if(img_dim == 2):
            patch_in = image[:, :, c0[0]:c1[0], c0[1]:c1[1]]
        else:
            patch_in = image[:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
        patch_out = net(patch_in) 
        if(not(isinstance(patch_out, tuple) or isinstance(patch_out, list))):
            patch_out = [patch_out]
        for i in range(output_num):
            if(img_dim == 2):
                output_list[i][:, :, c0[0]:c1[0], c0[1]:c1[1]] += patch_out[i]
                pred_num_arr[:, :, c0[0]:c1[0], c0[1]:c1[1]] += temp_mask
            else:
                output_list[i][:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]] += patch_out[i]
                pred_num_arr[:, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]] += temp_mask
    
    output_list = [item / pred_num_arr for item in output_list]
    return output_list
