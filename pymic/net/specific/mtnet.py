# -*- coding: utf-8 -*-
from __future__ import print_function, division
import copy
import torch
import torch.nn as nn
from pymic.net.cnn.basic_layer import get_conv_class
from pymic.net.cnn.unet import ConvBlock, UpBlock, Encoder, Decoder
from pymic.net.specific.mcnet import MCNet
# from pymic.net.net2d.unet2d import *
# from pymic.net.net2d.unet2d_multi_decoder import UNet2D_TriBranch

class ChannelAttention(nn.Module):
    def __init__(self, dim, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        if(dim == 2):
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.max_pool = nn.AdaptiveMaxPool3d(1)
        conv_nd = get_conv_class(dim)
        self.fc1 = conv_nd(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = conv_nd(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x*self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1
        padding = (kernel_size - 1) / 2
        conv_nd = get_conv_class(dim)
        self.conv1   = conv_nd(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out    = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return x*self.sigmoid(y)

class CBAM(nn.Module):
    def __init__(self, dim, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(dim, in_planes, ratio)
        self.sa = SpatialAttention(dim, kernel_size)
        
    def forward(self, x):
        out = self.ca(x)
        result = self.sa(out)
        return result

class ConvAttentionBlock(ConvBlock):
    """
    Two convolutional blocks with spatial or channel attention.
    Each block consists of `Conv2d` + `BatchNorm2d` + `Attention` + `LeakyReLU`.
    A dropout layer is used between the two blocks.

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    :param attention_mode: (int) Attention mode. 0--spatial attention; 1--channel attention; 2--CBAM
    """
    def __init__(self, dim, in_channels, out_channels, dropout_p, attention_mode = 0):
        super(ConvAttentionBlock, self).__init__(dim, in_channels, out_channels, dropout_p)
        if(attention_mode == 0):
            self.att = SpatialAttention(dim)
        elif(attention_mode == 1):
            self.att  = ChannelAttention(dim, out_channels)
        else:
            self.att  = CBAM(dim, out_channels)
       
    def forward(self, x):
        x = self.conv_conv(x)
        x = self.att(x)
        return x

class UpBlockAttention(UpBlock):
    """Up-sampling followed by `ConvAttentionBlock` in U-Net structure.
    
    The parameters for the backbone should be given in the `params` dictionary. 
    See :mod:`pymic.net.net2d.unet2d.UpBlock` for details. 
    """
    def __init__(self, dim, in_channels1, in_channels2, out_channels, dropout_p, up_mode = 2, attention_mode = 0):
        super(UpBlockAttention, self).__init__(dim, in_channels1, in_channels2, out_channels, dropout_p, up_mode)
        self.conv = ConvAttentionBlock(dim, in_channels2 * 2, out_channels, dropout_p, attention_mode)

class DecoderAttention(Decoder):
    """
    Decoder of 2D UNet with Attention.

    The parameters for the backbone should be given in the `params` dictionary. 
    See :mod:`pymic.net.net2d.unet2d.Decoder` for details. 

    Parameters specific to this class:

    :param attention_mode: (int) 0--spatial attention; 1--channel attention; 2--CBAM
    """
    def __init__(self, params):
        super(DecoderAttention, self).__init__(params)
        att_mode = params.get("attention_mode", 0)

        self.up1 = UpBlockAttention(self.dim, self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], self.dropout[3], self.up_mode, att_mode) 
        self.up2 = UpBlockAttention(self.dim, self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], self.dropout[2], self.up_mode, att_mode) 
        self.up3 = UpBlockAttention(self.dim, self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], self.dropout[1], self.up_mode, att_mode) 
        self.up4 = UpBlock(self.dim, self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], self.dropout[0], self.up_mode) 


class MTNet(MCNet):
    """
    A tri-branch network based on different attentions using UNet2D as backbone.
    It is used originally designed for semi-supervised segmentation.

    * Lanfeng Zhong, et al. Semi-supervised Pathological Image Segmentation via Cross 
      Distillation of Multiple Attentions.
      `MICCAI 2023. <https://link.springer.com/chapter/10.1007/978-3-031-43987-2_55>`_

    * Lanfeng Zhong, et al. Semi-supervised pathological image 
      segmentation via cross distillation of multiple attentions and Seg-CAM consistency.
      `Pattern Recognition 2024. <https://doi.org/10.1016/j.patcog.2024.110492>`_ 

    The original code is at: https://github.com/HiLab-git/CDMA
    
    The parameters for the backbone should be given in the `params` dictionary. 
    See :mod:`pymic.net.net2d.unet2d.UNet2D` for details. 
    """
    def __init__(self, params):
        super(MTNet, self).__init__(params)
        params    = self.get_default_parameters(params)
        params1   = copy.deepcopy(params)
        params2   = copy.deepcopy(params)
        params3   = copy.deepcopy(params)
        params1['attention_mode'] = 2
        params2['attention_mode'] = 0
        params3['attention_mode'] = 1

        self.encoder  = Encoder(params1)
        self.decoder1 = DecoderAttention(params1)
        self.decoder2 = DecoderAttention(params2)
        self.decoder3 = DecoderAttention(params3)
    
    
