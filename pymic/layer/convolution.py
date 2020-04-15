# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn

class ConvolutionLayer(nn.Module):
    """
    A compose layer with the following components:
    convolution -> (batch_norm / layer_norm / group_norm / instance_norm) -> activation -> (dropout)
    batch norm and dropout are optional
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
    A compose layer with the following components:
    convolution -> (batch_norm) -> activation -> (dropout)
    batch norm and dropout are optional
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

class ConvolutionSepAll3DLayer(nn.Module):
    """
    A compose layer with the following components:
    convolution -> (batch_norm) -> activation -> (dropout)
    batch norm and dropout are optional
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim = 3,
            stride = 1, padding = 0, dilation =1, groups = 1, bias = True, 
            batch_norm = True, acti_func = None):
        super(ConvolutionSepAll3DLayer, self).__init__()
        self.n_in_chns  = in_channels
        self.n_out_chns = out_channels
        self.batch_norm = batch_norm
        self.acti_func  = acti_func

        assert(dim == 3)
        chn = min(in_channels, out_channels)

        self.conv_intra_plane1 = nn.Conv2d(chn, chn,
            kernel_size, stride, padding, dilation, chn, bias)

        self.conv_intra_plane2 = nn.Conv2d(chn, chn,
            kernel_size, stride, padding, dilation, chn, bias)

        self.conv_intra_plane3 = nn.Conv2d(chn, chn,
            kernel_size, stride, padding, dilation, chn, bias)

        self.conv_space_wise  = nn.Conv2d(in_channels, out_channels,
            1, stride, 0, dilation, 1, bias)
        
        if(self.batch_norm):
                self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        in_shape = list(x.shape)
        assert(len(in_shape) == 5)
        [B, C, D, H, W] = in_shape
        f0 = x.permute(0, 2, 1, 3, 4)  #[B, D, C, H, W] 
        f0 = f0.contiguous().view([B*D, C, H, W])

        Cc = min(self.n_in_chns, self.n_out_chns)
        Co = self.n_out_chns
        if(self.n_in_chns > self.n_out_chns):
            f0 = self.conv_space_wise(f0)  #[B*D, Cc, H, W]  
   
        f1 = self.conv_intra_plane1(f0)
        f2 = f1.contiguous().view([B, D, Cc, H, W])
        f2 = f2.permute(0, 3, 2, 1, 4) #[B, H, Cc, D, W]
        f2 = f2.contiguous().view([B*H, Cc, D, W])
        f2 = self.conv_intra_plane2(f2)
        f3 = f2.contiguous().view([B, H, Cc, D, W])
        f3 = f3.permute(0, 4, 2, 3, 1) #[B, W, Cc, D, H]
        f3 = f3.contiguous().view([B*W, Cc, D, H])
        f3 = self.conv_intra_plane3(f3)
        if(self.n_in_chns <= self.n_out_chns):
            f3 = self.conv_space_wise(f3)  #[B*W, Co, D, H] 

        f3 = f3.contiguous().view([B, W, Co, D, H])
        f3 = f3.permute([0, 2, 3, 4, 1]) #[B, Co, D, H, W]

        if(self.batch_norm):
            f3 = self.bn(f3)
        if(self.acti_func is not None):
            f3 = self.acti_func(f3)
        return f3
