# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn

class DeconvolutionLayer(nn.Module):
    """
    A compose layer with the following components:
    deconvolution -> (batch_norm / layer_norm / group_norm / instance_norm) -> (activation) -> (dropout)
    Batch norm and activation are optional.

    :param in_channels: (int) The input channel number.
    :param out_channels: (int) The output channel number. 
    :param kernel_size: The size of convolution kernel. It can be either a single 
        int or a tupe of two or three ints. 
    :param dim: (int) The dimention of convolution (2 or 3).
    :param stride: (int) The stride of convolution. 
    :param padding: (int) Padding size. 
    :param dilation: (int) Dilation rate.
    :param groups: (int) The groupt number of convolution. 
    :param bias: (bool) Add bias or not for convolution. 
    :param batch_norm: (bool) Use batch norm or not.
    :param acti_func: (str or None) Activation funtion. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
            dim = 3, stride = 1, padding = 0, output_padding = 0, 
            dilation =1, groups = 1, bias = True, 
            batch_norm = True, acti_func = None):
        super(DeconvolutionLayer, self).__init__()
        self.n_in_chns  = in_channels
        self.n_out_chns = out_channels
        self.batch_norm = batch_norm
        self.acti_func  = acti_func
        
        assert(dim == 2 or dim == 3)
        if(dim == 2):
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                kernel_size, stride, padding, output_padding,
                groups, bias, dilation)
            if(self.batch_norm):
                self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                kernel_size, stride, padding, output_padding,
                groups, bias, dilation)
            if(self.batch_norm):
                self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        f = self.conv(x)
        if(self.batch_norm):
            f = self.bn(f)
        if(self.acti_func is not None):
            f = self.acti_func(f)
        return f

class  DepthSeperableDeconvolutionLayer(nn.Module):
    """
    Depth seperable deconvolution with the following components:
    1x1 conv -> deconv -> (batch_norm / layer_norm / group_norm / instance_norm) -> (activation) -> (dropout)
    Batch norm and activation are optional.

    :param in_channels: (int) The input channel number.
    :param out_channels: (int) The output channel number. 
    :param kernel_size: The size of convolution kernel. It can be either a single 
        int or a tupe of two or three ints. 
    :param dim: (int) The dimention of convolution (2 or 3).
    :param stride: (int) The stride of convolution. 
    :param padding: (int) Padding size for input.
    :param output_padding: (int) Padding size for ouput.
    :param dilation: (int) Dilation rate.
    :param groups: (int) The groupt number of convolution. 
    :param bias: (bool) Add bias or not for convolution. 
    :param batch_norm: (bool) Use batch norm or not.
    :param acti_func: (str or None) Activation funtion. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
        dim = 3, stride = 1, padding = 0, output_padding = 0,
        dilation =1, groups = 1, bias = True, 
        batch_norm = True, acti_func = None):
        super(DepthSeperableDeconvolutionLayer, self).__init__()
        self.n_in_chns  = in_channels
        self.n_out_chns = out_channels
        self.batch_norm = batch_norm
        self.acti_func  = acti_func
        self.groups     = groups
        assert(dim == 2 or dim == 3)
        if(dim == 2):
            self.conv1x1 = nn.Conv2d(in_channels, out_channels,
                kernel_size = 1, stride = 1, padding = 0, dilation = dilation, 
                groups = self.groups, bias = bias)
            self.conv = nn.ConvTranspose2d(out_channels, out_channels,
                kernel_size, stride, padding, output_padding,
                groups = out_channels, bias = bias, dilation = dilation)
            
            if(self.batch_norm):
                self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.conv1x1 = nn.Conv3d(in_channels, out_channels,
                kernel_size = 1, stride = 1, padding = 0, dilation = dilation, 
                groups = self.groups, bias = bias)
            self.conv = nn.ConvTranspose3d(out_channels, out_channels,
                kernel_size, stride, padding, output_padding,
                groups = out_channels, bias = bias, dilation = dilation)
            if(self.batch_norm):
                self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        f = self.conv1x1(x)
        f = self.conv(f)
        if(self.batch_norm):
            f = self.bn(f)
        if(self.acti_func is not None):
            f = self.acti_func(f)
        return f