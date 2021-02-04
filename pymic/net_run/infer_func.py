# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import math
import torch
import numpy as np
from pymic.util.image_process import *

def volume_infer(image, net, device, class_num, 
        mini_batch_size = None, mini_patch_inshape = None, mini_patch_outshape = None,
        stride  = None, output_num = 1):
    """
    Obtain net(image)
    sampling the image with patch_shape and use the patch as input of network
    if patch_size is None, use the whole image as input of the network 
    image : the input tensor on cuda
    device: device name
    net   : the network on cuda
    class_num : number of class for segmentation
    output_num: number of outputs, when >1, the network obtains a list of output array

    return outputs: a numpy array after inference
    """
    image = image.to(device)
    if(mini_patch_inshape is None):
        outputs = net(image)
        if(isinstance(outputs, tuple) or isinstance(outputs, list)):
            outputs = [item.cpu().numpy() for item in outputs]
            outputs = outputs[:output_num]
        else:
            outputs = outputs.cpu().numpy()
    else:
        outputs = volume_infer_by_patch(image, net, device, class_num,
            mini_batch_size, mini_patch_inshape, mini_patch_outshape, stride, output_num)
    if(isinstance(outputs, tuple) or isinstance(outputs, list)):
        if(output_num == 1):
            outputs = outputs[0]
    return outputs

def volume_infer_by_patch(image, net, device, class_num,
        mini_batch_size, mini_patch_inshape, mini_patch_outshape, stride, output_num):
    '''
        Test one image with sub regions along x, y, z axis
    '''
    image = image.cpu().numpy()
    img_full_shape = image.shape
    assert(img_full_shape[0] == 1)
    img_shape = img_full_shape[2:]
    img_dim   = len(img_shape)
    if(img_dim != 3):
        raise ValueError("volume_infer_by_patch only supports 3D images")
    [D, H, W] = img_shape
    for i in range(3):
        if mini_patch_inshape[i] is None:
            mini_patch_inshape[i]  = img_shape[i]
            mini_patch_outshape[i] = img_shape[i]
            stride[i] = img_shape[i]
    # pad the input image in case mini_patch_inshape > mini_patch_outshape
    margin = [mini_patch_inshape[i] - mini_patch_outshape[i] \
        for i in range(img_dim)]
    assert(min(margin) >= 0)
    if(max(margin) > 0):
        margin_lower = [int(margin[i] / 2) for i in range(img_dim)]
        margin_upper = [margin[i] - margin_lower[i] for i in range(img_dim)]
        pad = [(margin_lower[i], margin_upper[i]) for  i in range(img_dim)]
        pad = tuple([(0, 0, (0, 0))] + pad)
        image_pad = np.pad(image, pad, 'reflect')
    else:
        margin_lower = [0] * img_dim
        margin_upper = [0] * img_dim
        image_pad = image
    [padD, padH, padW] = image_pad.shape[2:]
    sub_image_patches = []
    sub_image_starts  = []
    

    for d in range(0, padD, stride[0]):
        d_min = min(d, padD - mini_patch_inshape[0])
        d_max = d_min + mini_patch_inshape[0]
        for h in range(0, padH, stride[1]):
            h_min = min(h, padH - mini_patch_inshape[1])
            h_max = h_min + mini_patch_inshape[1]
            for w in range(0, padW, stride[2]):
                w_min = min(w, padW - mini_patch_inshape[2])
                w_max = w_min + mini_patch_inshape[2]
                crop_start = [d_min, h_min, w_min]
                crop_end   = [d_max, h_max, w_max]
                crop_start_full = [0, 0] + crop_start
                crop_end_full   = list(img_full_shape[:2]) + crop_end
                sub_image_starts.append(crop_start)
                sub_image = crop_ND_volume_with_bounding_box(image, 
                    crop_start_full, crop_end_full)
                sub_image_patches.append(sub_image)

    # inference with image patches
    out_shape = [img_full_shape[0], class_num] + list(image_pad.shape[2:])
    out_list  = [np.zeros(out_shape, np.float32) for i in range(output_num)]
    out_mask  = np.zeros(out_shape, np.float32)
    total_batch = len(sub_image_patches)
    max_mini_batch = int((total_batch + mini_batch_size -1)/mini_batch_size)
    for mini_batch_idx in range(max_mini_batch):
        batch_end_idx = min((mini_batch_idx+1)*mini_batch_size, total_batch)
        batch_start_idx = batch_end_idx - mini_batch_size
        data_mini_batch = sub_image_patches[batch_start_idx:batch_end_idx]
        data_mini_batch = np.concatenate(data_mini_batch, axis = 0)
        data_mini_batch = torch.from_numpy(data_mini_batch)
        data_mini_batch = data_mini_batch.to(device)

        out_mini_batch  = net(data_mini_batch) # the network may give multiple predictions
        if(not(isinstance(out_mini_batch, tuple) or isinstance(out_mini_batch, list))):
            out_mini_batch = [out_mini_batch]
        out_mini_batch  = [item.cpu().numpy() for item in out_mini_batch]

        # use a mask to store overlapping regions
        mask_mini_batch = np.ones_like(out_mini_batch[0])
        for batch_idx in range(batch_start_idx, batch_end_idx):
            crop_start = sub_image_starts[batch_idx]
            crop_start = [crop_start[i] + margin_lower[i] for i in range(img_dim)]
            crop_end   = [crop_start[i] + mini_patch_outshape[i] for i in range(img_dim)]
            crop_start = [0, 0] + crop_start
            crop_end   = [1, class_num] + crop_end
            for i in range(output_num):
                out_list[i] = set_ND_volume_roi_with_bounding_box_range(out_list[i], crop_start, crop_end, 
                     out_mini_batch[i][batch_idx-batch_start_idx])
            temp_mask = np.zeros_like(out_mask)
            temp_mask = set_ND_volume_roi_with_bounding_box_range(temp_mask, crop_start, crop_end, 
                     mask_mini_batch[batch_idx-batch_start_idx])
            out_mask = out_mask + temp_mask
    if(max(margin) > 0):
        crop_start = [0, 0] + margin_lower
        crop_end   = [img_shape[i] - margin_upper[i] for i in range(img_dim)]
        crop_end   = [1, class_num] + crop_end
        out_list = [crop_ND_volume_with_bounding_box(item, crop_start, crop_end) for item in out_list]
        out_mask = crop_ND_volume_with_bounding_box(out_mask, crop_start, crop_end) 
    out_list = [item / out_mask for item in out_list]
    return out_list


