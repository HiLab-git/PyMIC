# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os 
import torch
import random
import numpy as np 
from scipy import ndimage
from pymic.io.image_read_write import *
from pymic.util.image_process import *
from pymic.util.general import get_one_hot_seg

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

def crop_ct_scan(input_img, output_img, input_lab = None, output_lab = None):
    """
    Crop a CT scan based on the bounding box of the human region. 
    """
    img_obj = sitk.ReadImage(input_img)
    img  = sitk.GetArrayFromImage(img_obj)
    mask = np.asarray(img > -600)
    se   = np.ones([3,3,3])
    mask = ndimage.binary_opening(mask, se, iterations = 2)
    mask = get_largest_k_components(mask, 1)
    bbmin, bbmax = get_ND_bounding_box(mask, margin = [5, 10, 10])
    img_sub = crop_ND_volume_with_bounding_box(img, bbmin, bbmax)
    img_sub_obj = sitk.GetImageFromArray(img_sub)
    img_sub_obj.SetSpacing(img_obj.GetSpacing())
    sitk.WriteImage(img_sub_obj, output_img)
    if(input_lab is not None):
        lab_obj  = sitk.ReadImage(input_lab)
        lab = sitk.GetArrayFromImage(lab_obj)
        lab_sub = crop_ND_volume_with_bounding_box(lab, bbmin, bbmax)
        lab_sub_obj = sitk.GetImageFromArray(lab_sub)
        lab_sub_obj.SetSpacing(img_obj.GetSpacing())
        sitk.WriteImage(lab_sub_obj, output_lab)


def patch_mix(x, fg_num, patch_num, size_d, size_h, size_w):
    """
    Copy a sub region of an impage and paste to another one to generate
    images and labels for self-supervised segmentation.
    """
    N, C, D, H, W = list(x.shape)
    fg_mask = torch.zeros_like(x)
    # generate mask 
    for n in range(N):
        p_num = random.randint(patch_num[0], patch_num[1])
        for i in range(p_num):
            d = random.randint(size_d[0], size_d[1])
            h = random.randint(size_h[0], size_h[1])
            w = random.randint(size_w[0], size_w[1])
            d_c = random.randint(0, D)
            h_c = random.randint(0, H)
            w_c = random.randint(0, W)
            d0, d1 = max(0, d_c - d), min(D, d_c + d)
            h0, h1 = max(0, h_c - h), min(H, h_c + h)
            w0, w1 = max(0, w_c - w), min(W, w_c + w)
            temp_m = torch.ones([C, d1-d0, h1-h0, w1-w0]) * random.randint(1, fg_num)
            fg_mask[n, :, d0:d1, h0:h1, w0:w1] = temp_m
    fg_w   = fg_mask * 1.0 / fg_num
    x_roll = torch.roll(x, 1, 0)
    x_fuse = fg_w*x_roll + (1.0 - fg_w)*x     
    y_prob = get_one_hot_seg(fg_mask.to(torch.int32), fg_num + 1)
    return x_fuse, y_prob 

def create_mixed_dataset(input_dir, output_dir, fg_num = 1,  crop_num = 1, 
        mask_dir = None, data_format = "nii.gz"):
    """
    Create dataset based on patch mix. 

    :param input_dir: (str) The path of folder for input images
    :param output_dir: (str) The path of folder for output images
    :param fg_num: (int) The number of foreground classes
    :param crop_num: (int) The number of patches to crop for each input image
    :param mask: ND array to specify a mask, or 'default' or None. If default, 
        a mask for body region is automatically generated (just for CT).
    :param data_format: (str) The format of images.  
    """
    img_names = os.listdir(input_dir)
    img_names = [item for item in img_names if item.endswith(data_format)]
    img_names = sorted(img_names)
    out_img_dir = output_dir + "/image"
    out_lab_dir = output_dir + "/label"
    if(not os.path.exists(out_img_dir)):
        os.mkdir(out_img_dir)
    if(not os.path.exists(out_lab_dir)):
        os.mkdir(out_lab_dir)

    img_num = len(img_names)
    print("image number", img_num)
    i_range = range(img_num)
    j_range = list(i_range)
    random.shuffle(j_range)
    for i in i_range:
        print(i, img_names[i])
        j = j_range[i]
        if(i == j):
            j = i + 1 if i < img_num - 1 else 0 
        img_i = load_image_as_nd_array(input_dir + "/" + img_names[i])['data_array']
        img_j = load_image_as_nd_array(input_dir + "/" + img_names[j])['data_array']

        chns  = img_i.shape[0]
        # random crop to patch size
        if(mask_dir is None):
            mask_i = get_human_region_mask(img_i)
            mask_j = get_human_region_mask(img_j)
        else:
            mask_i = load_image_as_nd_array(mask_dir + "/" + img_names[i])['data_array']
            mask_j = load_image_as_nd_array(mask_dir + "/" + img_names[j])['data_array']
        for k in range(crop_num):
            # if(mask is None):
            #     img_ik = random_crop_ND_volume(img_i, [chns, 96, 96, 96])
            #     img_jk = random_crop_ND_volume(img_j, [chns, 96, 96, 96])
            # else:
            img_ik = random_crop_ND_volume_with_mask(img_i, [chns, 96, 96, 96], mask_i)
            img_jk = random_crop_ND_volume_with_mask(img_j, [chns, 96, 96, 96], mask_j)
            C, D, H, W = img_ik.shape
            # generate mask 
            fg_mask = np.zeros_like(img_ik, np.uint8)
            patch_num = random.randint(4, 40)
            for patch in range(patch_num):
                d = random.randint(4, 20) # half of window size
                h = random.randint(4, 40)
                w = random.randint(4, 40)
                d_c = random.randint(0, D)
                h_c = random.randint(0, H)
                w_c = random.randint(0, W)
                d0, d1 = max(0, d_c - d), min(D, d_c + d)
                h0, h1 = max(0, h_c - h), min(H, h_c + h)
                w0, w1 = max(0, w_c - w), min(W, w_c + w)
                temp_m = np.ones([C, d1-d0, h1-h0, w1-w0]) * random.randint(1, fg_num)
                fg_mask[:, d0:d1, h0:h1, w0:w1] = temp_m
            fg_w   = fg_mask * 1.0 / fg_num
            x_fuse = fg_w*img_jk + (1.0 - fg_w)*img_ik

            out_name = img_names[i]
            if crop_num > 1:
                out_name = out_name.replace(".nii.gz", "_{0:}.nii.gz".format(k))
            save_nd_array_as_image(x_fuse[0], out_img_dir + "/" + out_name, 
                reference_name = input_dir + "/" + img_names[i]) 
            save_nd_array_as_image(fg_mask[0], out_lab_dir + "/" + out_name, 
                reference_name = input_dir + "/" + img_names[i]) 

