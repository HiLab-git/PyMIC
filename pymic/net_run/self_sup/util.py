# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os 
import copy 
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
    fg_mask = torch.zeros_like(x[:, :1, :, :, :]).to(torch.int32)
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
            temp_m = torch.ones([1, d1 - d0, h1 - h0, w1 - w0]) * random.randint(1, fg_num)
            fg_mask[n, :, d0:d1, h0:h1, w0:w1] = temp_m
    fg_w   = fg_mask * 1.0 / fg_num
    x_roll = torch.roll(x, 1, 0)
    x_fuse = fg_w*x_roll + (1.0 - fg_w)*x     
    # y_prob = get_one_hot_seg(fg_mask.to(torch.int32), fg_num + 1)
    return x_fuse, fg_mask 

def nonlinear_transform(x):
    v_min = torch.min(x)
    v_max = torch.max(x)
    x = (x - v_min)/(v_max - v_min)
    a = random.random() * 0.7 + 0.15
    b = random.random() * 0.7 + 0.15
    alpha = b / a 
    beta  = (1 - b) / (1 - a)
    if(alpha < 1.0 ):
        y = torch.maximum(alpha*x, beta*x + 1 - beta)
    else:
        y = torch.minimum(alpha*x, beta*x + 1 - beta)
    if(random.random() < 0.5):
        y = 1.0 - y 
    y = y * (v_max - v_min) + v_min
    return y 

def nonlienar_volume_fusion(x, block_range, size_min, size_max):
    """
    Fuse a subregion of an impage with another one to generate
    images and labels for self-supervised segmentation.
    input x should be a batch of tensors
    """
    #n_min, n_max,  
    N, C, D, H, W = list(x.shape)
    # apply nonlinear transform to x:
    x_nl1 = torch.zeros_like(x).to(torch.float32)
    x_nl2 = torch.zeros_like(x).to(torch.float32)
    for n in range(N):
        x_nl1[n] = nonlinear_transform(x[n])
        x_nl2[n] = nonlinear_transform(x[n])
    x_roll = torch.roll(x_nl2, 1, 0)
    mask   = torch.zeros_like(x).to(torch.int32)
    p_num = random.randint(block_range[0], block_range[1])
    for n in range(N):
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
            temp_m = torch.ones([C, d1 - d0, h1 - h0, w1 - w0])
            if(random.random() < 0.5):
                temp_m = temp_m * 2
            mask[n, :, d0:d1, h0:h1, w0:w1] = temp_m
    
    mask1 = (mask == 1).to(torch.int32)
    mask2 = (mask == 2).to(torch.int32)
    y = x_nl1 * (1.0 - mask1) + x_nl2 * mask1
    y = y * (1.0 - mask2) + x_roll * mask2
    return y, mask
    
def augmented_volume_fusion(x, size_min, size_max):
    """
    Fuse a subregion of an impage with another one to generate
    images and labels for self-supervised segmentation.
    input x should be a batch of tensors
    """
    #n_min, n_max,  
    N, C, D, H, W = list(x.shape)
    # apply nonlinear transform to x:
    x1   = torch.zeros_like(x).to(torch.float32)
    y    = torch.zeros_like(x).to(torch.float32)
    mask = torch.zeros_like(x).to(torch.int32)
    for n in range(N):
        x1[n] = nonlinear_transform(x[n])
        y[n]  = nonlinear_transform(x[n])
    x2 = torch.roll(x1, 1, 0)

    for n in range(N): 
        block_size = [random.randint(size_min[i], size_max[i]) for i in range(3)]
        d_start = random.randint(0, block_size[0] // 2)
        h_start = random.randint(0, block_size[1] // 2)
        w_stat  = random.randint(0, block_size[2] // 2)
        for d in range(d_start, D, block_size[0]):
            if(D - d < block_size[0] // 2):
                continue
            d1 = min(d + block_size[0], D)
            for h in range(h_start, H, block_size[1]):
                if(H - h < block_size[1] // 2):
                    continue
                h1 = min(h + block_size[1], H)
                for w in range(w_stat, W, block_size[2]):
                    if(W - w < block_size[2] // 2):
                        continue
                    w1 = min(w + block_size[2], W)
                    p = random.random()
                    if(p < 0.15): # nonlinear intensity augmentation
                        mask[n, :, d:d1, h:h1, w:w1] = 1
                        y[n, :, d:d1, h:h1, w:w1] = x1[n, :, d:d1, h:h1, w:w1]
                    elif(p < 0.3): # random flip across a certain axis
                        mask[n, :, d:d1, h:h1, w:w1] = 2
                        flip_axis = random.randint(-3, -1)
                        y[n, :, d:d1, h:h1, w:w1] = torch.flip(y[n, :, d:d1, h:h1, w:w1], (flip_axis,))
                    elif(p < 0.45): # nonlinear intensity augmentation and random flip across a certain axis
                        mask[n, :, d:d1, h:h1, w:w1] = 3
                        flip_axis = random.randint(-3, -1)
                        y[n, :, d:d1, h:h1, w:w1] = torch.flip(x1[n, :, d:d1, h:h1, w:w1], (flip_axis,))
                    elif(p < 0.6):  # paste from another volume
                        mask[n, :, d:d1, h:h1, w:w1] = 4
                        y[n, :, d:d1, h:h1, w:w1] = x2[n, :, d:d1, h:h1, w:w1]
    return y, mask 

def self_volume_fusion(x, fg_num, fuse_ratio, size_min, size_max):
    """
    Fuse a subregion of an impage with another one to generate
    images and labels for self-supervised segmentation.
    input x should be a batch of tensors
    """
    #n_min, n_max,  
    N, C, D, H, W = list(x.shape)
    y = 1.0 * x 
    fg_mask = torch.zeros_like(x[:, :1, :, :, :]).to(torch.int32)
 
    for n in range(N):
        db = random.randint(size_min[0], size_max[0])
        hb = random.randint(size_min[1], size_max[1])
        wb = random.randint(size_min[2], size_max[2])
        d0 = random.randint(0, D % db)
        h0 = random.randint(0, H % hb)
        w0 = random.randint(0, W % wb)
        coord_list_source = []
        for di in range(D // db):
            for hi in range(H // hb):
                for wi in range(W // wb):
                    coord_list_source.append([di, hi, wi])
        coord_list_target = copy.deepcopy(coord_list_source)
        random.shuffle(coord_list_source)
        random.shuffle(coord_list_target)
        for i in range(int(len(coord_list_source)*fuse_ratio)):
            ds_l = d0 + db * coord_list_source[i][0]
            hs_l = h0 + hb * coord_list_source[i][1]    
            ws_l = w0 + wb * coord_list_source[i][2]    
            dt_l = d0 + db * coord_list_target[i][0]
            ht_l = h0 + hb * coord_list_target[i][1]    
            wt_l = w0 + wb * coord_list_target[i][2]  
            s_crop = x[n, :, ds_l:ds_l+db, hs_l:hs_l+hb, ws_l:ws_l+wb]
            t_crop = x[n, :, dt_l:dt_l+db, ht_l:ht_l+hb, wt_l:wt_l+wb]
            fg_m = random.randint(1, fg_num)
            fg_w = fg_m / (fg_num + 0.0)
            y[n, :, dt_l:dt_l+db, ht_l:ht_l+hb, wt_l:wt_l+wb] = t_crop * (1.0 - fg_w) + s_crop * fg_w
            fg_mask[n, 0, dt_l:dt_l+db, ht_l:ht_l+hb, wt_l:wt_l+wb] = \
                torch.ones([1, db, hb, wb]) * fg_m
    return y, fg_mask 