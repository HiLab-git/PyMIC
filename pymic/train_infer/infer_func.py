# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import math
import torch
import numpy as np
from pymic.util.image_process import *

def volume_infer(image, net, device, class_num, 
        mini_batch_size = None, 
        mini_patch_inshape = None,
        mini_patch_outshape = None):
    """
    Obtain net(image)
    sampling the image with patch_shape and use the patch as input of network
    if patch_size is None, use the whole image as input of the network 
    """
    image = image.to(device)
    if(mini_patch_inshape is None):
        outputs = net(image).cpu().numpy()
    else:
        outputs = volume_infer_by_patch(image, net, device, class_num,
            mini_batch_size, mini_patch_inshape, mini_patch_outshape)
    return outputs

def volume_infer_by_patch(image, net, device, class_num,
        mini_batch_size, mini_patch_inshape, mini_patch_outshape):
    '''
        Test one image with sub regions along x, y, z axis
        img        : a 4D numpy array with shape [D, H, W, C]
        data_shape : input 4d tensor shape
        label_shape: output 4d tensor shape
        class_num  : number of output class
        batch_size : batch size for testing
        sess       : tensorflow session that can run a graph
        x          : input tensor of the graph
        proby      : output tensor of the graph
        '''
    image = image.cpu().numpy()
    img_full_shape = image.shape
    assert(img_full_shape[0] == 1)
    img_shape = img_full_shape[2:]
    img_dim   = len(img_shape)
    if(img_dim != 3):
        raise ValueError("volume_infer_by_patch only supports 3D images")
    [D, H, W] = img_shape
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
    nd = math.ceil(D/mini_patch_outshape[0])
    nh = math.ceil(H/mini_patch_outshape[1])
    nw = math.ceil(W/mini_patch_outshape[2])
    sub_image_patches = []
    sub_image_starts  = []
    

    for di in range(nd):
        d_min = min(di*mini_patch_inshape[0], padD - mini_patch_inshape[0])
        d_max = d_min + mini_patch_inshape[0]
        for hi in range(nh):
            h_min = min(hi*mini_patch_inshape[1], padH - mini_patch_inshape[1])
            h_max = h_min + mini_patch_inshape[1]
            for wi in range(nw):
                w_min = min(wi*mini_patch_inshape[2], padW - mini_patch_inshape[2])
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
    out = np.zeros(out_shape, np.float32)
    total_batch = len(sub_image_patches)
    max_mini_batch = int((total_batch + mini_batch_size -1)/mini_batch_size)
    for mini_batch_idx in range(max_mini_batch):
        batch_end_idx = min((mini_batch_idx+1)*mini_batch_size, total_batch)
        batch_start_idx = batch_end_idx - mini_batch_size
        data_mini_batch = sub_image_patches[batch_start_idx:batch_end_idx]
        data_mini_batch = np.concatenate(data_mini_batch, axis = 0)
        data_mini_batch = torch.from_numpy(data_mini_batch)
        data_mini_batch = torch.tensor(data_mini_batch)
        data_mini_batch = data_mini_batch.to(device)
        out_mini_batch  = net(data_mini_batch).cpu().numpy()

        for batch_idx in range(batch_start_idx, batch_end_idx):
            crop_start = sub_image_starts[batch_idx]
            crop_start = [crop_start[i] + margin_lower[i] for i in range(img_dim)]
            crop_end   = [crop_start[i] + mini_patch_outshape[i] for i in range(img_dim)]
            crop_start = [0, 0] + crop_start
            crop_end   = [1, class_num] + crop_end

            out = set_ND_volume_roi_with_bounding_box_range(out, crop_start, crop_end, 
                     out_mini_batch[batch_idx-batch_start_idx])

    if(max(margin) > 0):
        crop_start = [0, 0] + margin_lower
        crop_end   = [img_shape[i] - margin_upper[i] for i in range(img_dim)]
        crop_end   = [1, class_num] + crop_end
        out = crop_ND_volume_with_bounding_box(out, crop_start, crop_end) 
    return out
