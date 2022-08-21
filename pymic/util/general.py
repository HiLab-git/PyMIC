# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import numpy as np 

def keyword_match(a,b):
    return a.lower() == b.lower()

def get_one_hot_seg(label, class_num):
    """
    convert a segmentation label to one-hot
    label: a tensor with a shape of [N, 1, D, H, W] or [N, 1, H, W]
    class_num: class number. 
    output: an one-hot tensor with a shape of [N, C, D, H, W] or [N, C, H, W]
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