# -*- coding: utf-8 -*-
from __future__ import print_function, division

import logging
import torch
import torch.nn as nn
import numpy as np 

def get_conv_class(dim = 2):
    if(dim == 2):
        return nn.Conv2d 
    elif(dim == 3):
        return nn.Conv3d
    else:
        raise ValueError("dim should be 2 or 3 in get_conv_class")

def get_transpose_conv_class(dim = 2):
    if(dim == 2):
        return nn.ConvTranspose2d 
    elif(dim == 3):
        return nn.ConvTranspose3d
    else:
        raise ValueError("dim should be 2 or 3 in get_transpose_conv_class")

def get_norm_class(dim = 2, norm_type = "batch_norm"):
    if(dim == 2):
        if(norm_type == "batch_norm"):
            return nn.BatchNorm2d
        else:
            return nn.InstanceNorm2d
    elif(dim == 3):
        if(norm_type == "batch_norm"):
            return nn.BatchNorm3d
        else:
            return nn.InstanceNorm3d
    else:
        raise ValueError("dim should be 2 or 3 in get_norm_class")

def get_maxpool_class(dim = 2):
    if(dim == 2):
        return nn.MaxPool2d
    elif(dim == 3):
        return nn.MaxPool3d
    else:
        raise ValueError("dim should be 2 or 3 in get_maxpool_class")