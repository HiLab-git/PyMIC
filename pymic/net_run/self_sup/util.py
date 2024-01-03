# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os 
import torch
import random
import numpy as np 
from scipy import ndimage
from pymic.io.image_read_write import *
from pymic.util.image_process import *


def get_human_region_mask(img):
    """
    Get the mask of human region in CT volumes
    """
    dim = len(img.shape)
    if( dim == 4):
        img = img[0]
    mask = np.asarray(img > -600)
    se = np.ones([3,3,3])
    mask = ndimage.binary_opening(mask, se, iterations = 2)
    D, H, W = mask.shape 
    for h in range(H):
        if(mask[:,h,:].sum() < 2000):
            mask[:,h, :] = np.zeros((D, W))
    mask = get_largest_k_components(mask, 1)
    mask_close = ndimage.binary_closing(mask, se, iterations = 2)

    D, H, W = mask.shape
    for d in [1, 2, D-3, D-2]:
        mask_close[d] = mask[d]
    for d in range(0, D, 2):
        mask_close[d, 2:-2, 2:-2] = np.ones((H-4, W-4))
    
    # get background component
    bg = np.zeros_like(mask)
    bgs = get_largest_k_components(1- mask_close, 10)
    for bgi in bgs:
        indices = np.where(bgi)
        if(bgi.sum() < 1000):
            break
        if(indices[0].min() == 0 or indices[1].min() == 0 or indices[2].min() ==0 or \
           indices[0].max() == D-1 or indices[1].max() == H-1 or indices[2].max() ==W-1):
            bg = bg + bgi
    fg = 1 - bg 

    fg = ndimage.binary_opening(fg, se, iterations = 1)
    fg = get_largest_k_components(fg, 1)
    if(dim == 4):
        fg = np.expand_dims(fg, 0)
    fg = np.asarray(fg, np.uint8)
    return fg 

def get_human_region_mask_fast(img, itk_spacing):
    # downsample
    D, H, W = img.shape 
    # scale_down = [1, 1, 1]
    if(itk_spacing[2] <= 1):
        scale_down = [1/2, 1/2, 1/2]
    else:
        scale_down = [1, 1/2, 1/2]
    img_sub    = ndimage.interpolation.zoom(img, scale_down, order = 0)
    mask       = get_human_region_mask(img_sub)
    D1, H1, W1 = mask.shape 
    scale_up = [D/D1, H/H1, W/W1]
    mask = ndimage.interpolation.zoom(mask, scale_up, order = 0)
    return mask

def crop_ct_scan(input_img, output_img, input_lab = None, output_lab = None, z_axis_density = 0.5):
    """
    Crop a CT scan based on the bounding box of the human region. 
    """
    img_obj = sitk.ReadImage(input_img)
    img     = sitk.GetArrayFromImage(img_obj)
    mask    = np.asarray(img > -600)
    mask2d  = np.mean(mask, axis = 0) > z_axis_density
    se      = np.ones([3,3])
    mask2d  = ndimage.binary_opening(mask2d, se, iterations = 2)
    mask2d  = get_largest_k_components(mask2d, 1)
    bbmin, bbmax = get_ND_bounding_box(mask2d, margin = [0, 0])
    bbmin   = [0] + bbmin
    bbmax   = [img.shape[0]] + bbmax
    img_sub = crop_ND_volume_with_bounding_box(img, bbmin, bbmax)
    img_sub_obj = sitk.GetImageFromArray(img_sub)
    img_sub_obj.SetSpacing(img_obj.GetSpacing())
    img_sub_obj.SetDirection(img_obj.GetDirection())
    sitk.WriteImage(img_sub_obj, output_img)
    if(input_lab is not None):
        lab_obj  = sitk.ReadImage(input_lab)
        lab = sitk.GetArrayFromImage(lab_obj)
        lab_sub = crop_ND_volume_with_bounding_box(lab, bbmin, bbmax)
        lab_sub_obj = sitk.GetImageFromArray(lab_sub)
        lab_sub_obj.SetSpacing(img_obj.GetSpacing())
        sitk.WriteImage(lab_sub_obj, output_lab)

def get_human_body_mask_and_crop(input_dir, out_img_dir, out_mask_dir):
    if(not os.path.exists(out_img_dir)):
        os.mkdir(out_img_dir)
        os.mkdir(out_mask_dir)

    img_names = [item for item in os.listdir(input_dir) if "nii.gz" in item]
    img_names  = sorted(img_names)
    for img_name in img_names:
        print(img_name)
        input_name = input_dir + "/" + img_name
        out_name   = out_img_dir + "/" + img_name 
        mask_name  = out_mask_dir + "/" + img_name 
        if(os.path.isfile(out_name)):
            continue
        img_obj = sitk.ReadImage(input_name)
        img     = sitk.GetArrayFromImage(img_obj)
        spacing = img_obj.GetSpacing()

        # downsample
        D, H, W = img.shape 
        spacing = img_obj.GetSpacing()
        # scale_down = [1, 1, 1]
        if(spacing[2] <= 1):
            scale_down = [1/2, 1/2, 1/2]
        else:
            scale_down = [1, 1/2, 1/2]
        img_sub    = ndimage.interpolation.zoom(img, scale_down, order = 0)
        mask       = get_human_region_mask(img_sub)
        D1, H1, W1 = mask.shape 
        scale_up = [D/D1, H/H1, W/W1]
        mask = ndimage.interpolation.zoom(mask, scale_up, order = 0)

        bbmin, bbmax = get_ND_bounding_box(mask)
        img_crop  = crop_ND_volume_with_bounding_box(img, bbmin, bbmax)
        mask_crop = crop_ND_volume_with_bounding_box(mask, bbmin, bbmax)

        out_img_obj = sitk.GetImageFromArray(img_crop)
        out_img_obj.SetSpacing(spacing)
        sitk.WriteImage(out_img_obj, out_name)
        mask_obj = sitk.GetImageFromArray(mask_crop)
        mask_obj.CopyInformation(out_img_obj)
        sitk.WriteImage(mask_obj, mask_name)


def volume_fusion(x, fg_num, block_range, size_min, size_max):
    """
    Fuse a subregion of an impage with another one to generate
    images and labels for self-supervised segmentation.
    input x should be a batch of tensors
    """
    #n_min, n_max,  
    N, C, D, H, W = list(x.shape)
    fg_mask = torch.zeros_like(x).to(torch.int32)
    # generate mask 
    for n in range(N):
        p_num = random.randint(block_range[0], block_range[1])
        for i in range(p_num):
            d = random.randint(size_min[0], size_max[0])
            h = random.randint(size_min[1], size_max[1])
            w = random.randint(size_min[2], size_max[2])
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
            temp_m = torch.ones([C, d1 - d0, h1 - h0, w1 - w0]) * random.randint(1, fg_num)
            fg_mask[n, :, d0:d1, h0:h1, w0:w1] = temp_m
    fg_w   = fg_mask * 1.0 / fg_num
    x_roll = torch.roll(x, 1, 0)
    x_fuse = fg_w*x_roll + (1.0 - fg_w)*x     
    # y_prob = get_one_hot_seg(fg_mask.to(torch.int32), fg_num + 1)
    return x_fuse, fg_mask 
