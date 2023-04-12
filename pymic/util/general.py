# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import numpy as np 

def keyword_match(a,b):
    """
    Test if two string are the same when converted to lower case. 
    """
    return a.lower() == b.lower()

def tensor_shape_match(a,b):
    """
    Test if two tensors have the same shape"""
    shape_a = list(a.shape)
    shape_b = list(b.shape)
    len_a  = len(shape_a)
    len_b  = len(shape_b)
    if(len_a != len_b):
        return False 
    elif(len_a == 0):
        return True 
    else:
        for i in range(len_a):
            if(shape_a[i] != shape_b[i]):
                return False 
        return True 

def is_image_name(x):
    valid_names = ["jpg", "jpeg", "png", "bmp", "nii.gz",
                   "tif", "nii", "nii.gz", "mha"]
    valid = False 
    for item in valid_names:
        if(x.endswith(item)):
            valid = True 
            break 
    return valid

def get_one_hot_seg(label, class_num):
    """
    Convert a segmentation label to one-hot.

    :param label: A tensor with a shape of [N, 1, D, H, W] or [N, 1, H, W]
    :param class_num: Class number. 
    
    :return: a one-hot tensor with a shape of [N, C, D, H, W] or [N, C, H, W].
    """
    size = list(label.size())
    if(size[1] != 1):
        raise ValueError("The channel should be 1, \
            rather than {0:} before one-hot encoding".format(size[1]))
    label = label.view(-1)
    ones  = torch.sparse.torch.eye(class_num).to(label.device)
    one_hot = ones.index_select(0, label)
    size.append(class_num)
    one_hot = one_hot.view(*size)
    one_hot = torch.transpose(one_hot, 1, -1)
    one_hot = torch.squeeze(one_hot, -1)
    return one_hot

def mixup(inputs, labels):
    """Shuffle a minibatch and do linear interpolation between images and labels.
    Both classification and segmentation labels are supported. The targets should
    be one-hot labels.
    
    :param inputs: a tensor of input images with size N X C0 x H x W.
    :param labels: a tensor of one-hot labels. The shape is N X C for classification
        tasks, and N X C X H X W for segmentation tasks. 
    """
    input_shape = list(inputs.shape)
    label_shape = list(labels.shape)
    img_dim     = len(input_shape) - 2
    N = input_shape[0] # batch size
    C = label_shape[1] # class number
    rp1 = torch.randperm(N)
    inputs1 = inputs[rp1]
    labels1 = labels[rp1]
    
    rp2 = torch.randperm(N)
    inputs2 = inputs[rp2]
    labels2 = labels[rp2]

    a = np.random.beta(1, 1, [N, 1])
    if(img_dim == 2):
        b = np.tile(a[..., None, None], [1] + input_shape[1:])
    elif(img_dim == 3):
        b = np.tile(a[..., None, None, None], [1] + input_shape[1:])
    else:
        raise ValueError("MixUp only supports 2D and 3D images, but the " +
            "input image has {0:} dimensions".format(img_dim))

    inputs1 = inputs1 * torch.from_numpy(b).float()
    inputs2 = inputs2 * torch.from_numpy(1 - b).float()
    inputs_mix = inputs1 + inputs2

    if(len(label_shape) == 2): # for classification tasks
        c = np.tile(a, [1, C])
    elif(img_dim == 2):        # for 2D segmentation tasks
        c = np.tile(a[..., None, None], [1] + label_shape[1:])
    else:                      # for 3D segmentation tasks
        c = np.tile(a[..., None, None, None], [1] + label_shape[1:])
    
    labels1 = labels1 * torch.from_numpy(c).float()
    labels2 = labels2 * torch.from_numpy(1 - c).float()
    labels_mix = labels1 + labels2

    return inputs_mix, labels_mix
