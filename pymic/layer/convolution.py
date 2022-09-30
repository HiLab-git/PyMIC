# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn

class ConvolutionLayer(nn.Module):
    """
    A compose layer with the following components:
    convolution -> (batch_norm / layer_norm / group_norm / instance_norm) -> (activation) -> (dropout)
    Batch norm and activation are optional.

    :param in_channels: (int) The input channel number.
    :param out_channels: (int) The output channel number. 
    :param kernel_size: The size of convolution kernel. It can be either a single 
        int or a tupe of two or three ints. 
    :param dim: (int) The dimention of convolution (2 or 3).
    :param stride: (int) The stride of convolution. 
    :param padding: (int) Padding size. 
    :param dilation: (int) Dilation rate.
    :param conv_group: (int) The groupt number of convolution. 
    :param bias: (bool) Add bias or not for convolution. 
    :param norm_type: (str or None) Normalization type, can be `batch_norm`, 'group_norm'.
    :param norm_group: (int) The number of group for group normalization.
    :param acti_func: (str or None) Activation funtion. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim = 3,
            stride = 1, padding = 0, dilation = 1, conv_group = 1, bias = True, 
            norm_type = 'batch_norm', norm_group = 1, acti_func = None):
        super(ConvolutionLayer, self).__init__()
        self.n_in_chns  = in_channels
        self.n_out_chns = out_channels
        self.norm_type  = norm_type
        self.norm_group = norm_group
        self.acti_func  = acti_func

        assert(dim == 2 or dim == 3)
        if(dim == 2):
            self.conv = nn.Conv2d(in_channels, out_channels,
                kernel_size, stride, padding, dilation, conv_group, bias)
            if(self.norm_type == 'batch_norm'):
                self.bn = nn.BatchNorm2d(out_channels)
            elif(self.norm_type == 'group_norm'):
                self.bn = nn.GroupNorm(self.norm_group, out_channels)
            elif(self.norm_type is not None):
                raise ValueError("unsupported normalization method {0:}".format(norm_type))
        else:        
            self.conv = nn.Conv3d(in_channels, out_channels,
                kernel_size, stride, padding, dilation, conv_group, bias)
            if(self.norm_type == 'batch_norm'):
                self.bn = nn.BatchNorm3d(out_channels)
            elif(self.norm_type == 'group_norm'):
                self.bn = nn.GroupNorm(self.norm_group, out_channels)
            elif(self.norm_type is not None):
                raise ValueError("unsupported normalization method {0:}".format(norm_type))

    def forward(self, x):
        f = self.conv(x)
        if(self.norm_type is not None):
            f = self.bn(f)
        if(self.acti_func is not None):
            f = self.acti_func(f)
        return f

class DepthSeperableConvolutionLayer(nn.Module):
    """
    Depth seperable convolution with the following components:
    1x1 conv -> group conv -> (batch_norm / layer_norm / group_norm / instance_norm) -> (activation) -> (dropout)
    Batch norm and activation are optional.

    :param in_channels: (int) The input channel number.
    :param out_channels: (int) The output channel number. 
    :param kernel_size: The size of convolution kernel. It can be either a single 
        int or a tupe of two or three ints. 
    :param dim: (int) The dimention of convolution (2 or 3).
    :param stride: (int) The stride of convolution. 
    :param padding: (int) Padding size. 
    :param dilation: (int) Dilation rate.
    :param conv_group: (int) The groupt number of convolution. 
    :param bias: (bool) Add bias or not for convolution. 
    :param norm_type: (str or None) Normalization type, can be `batch_norm`, 'group_norm'.
    :param norm_group: (int) The number of group for group normalization.
    :param acti_func: (str or None) Activation funtion. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim = 3,
            stride = 1, padding = 0, dilation =1, conv_group = 1, bias = True, 
            norm_type = 'batch_norm', norm_group = 1, acti_func = None):
        super(DepthSeperableConvolutionLayer, self).__init__()
        self.n_in_chns  = in_channels
        self.n_out_chns = out_channels
        self.norm_type  = norm_type
        self.norm_group = norm_group
        self.acti_func  = acti_func

        assert(dim == 2 or dim == 3)
        if(dim == 2):
            self.conv1x1 = nn.Conv2d(in_channels, out_channels,
                kernel_size = 1, stride = stride, padding = 0, dilation = dilation, groups = conv_group, bias = bias)
            self.conv = nn.Conv2d(out_channels, out_channels,
                kernel_size, stride, padding, dilation, groups = out_channels, bias = bias)
            if(self.norm_type == 'batch_norm'):
                self.bn = nn.BatchNorm2d(out_channels)
            elif(self.norm_type == 'group_norm'):
                self.bn = nn.GroupNorm(self.norm_group, out_channels)
            elif(self.norm_type is not None):
                raise ValueError("unsupported normalization method {0:}".format(norm_type))
        else:     
            self.conv1x1 = nn.Conv3d(in_channels, out_channels,
                kernel_size = 1, stride = stride, padding = 0, dilation = dilation, groups = conv_group, bias = bias)   
            self.conv = nn.Conv3d(out_channels, out_channels,
                kernel_size, stride, padding, dilation, groups = out_channels, bias = bias)
            if(self.norm_type == 'batch_norm'):
                self.bn = nn.BatchNorm3d(out_channels)
            elif(self.norm_type == 'group_norm'):
                self.bn = nn.GroupNorm(self.norm_group, out_channels)
            elif(self.norm_type is not None):
                raise ValueError("unsupported normalization method {0:}".format(norm_type))

    def forward(self, x):
        f = self.conv1x1(x)
        f = self.conv(f)
        if(self.norm_type is not None):
            f = self.bn(f)
        if(self.acti_func is not None):
            f = self.acti_func(f)
        return f

