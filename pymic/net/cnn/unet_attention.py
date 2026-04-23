# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.net.cnn.basic_layer import *
from pymic.net.cnn.unet import *

class AttentionGateBlock(nn.Module):
    def __init__(self, dim, chns_l, chns_h):
        """
        chns_l: channel number of low-level features from the encoder
        chns_h: channel number of high-level features from the decoder
        """
        super(AttentionGateBlock, self).__init__()
        assert dim == 2 or dim == 3
        self.dim = dim
        self.in_chns_l = chns_l
        self.in_chns_h = chns_h

        conv_nd = get_conv_class(self.dim)
        self.out_chns = int(min(self.in_chns_l, self.in_chns_h)/2)
        self.conv1_l = conv_nd(self.in_chns_l, self.out_chns,
                kernel_size = 1, bias = True)
        self.conv1_h = conv_nd(self.in_chns_h, self.out_chns,
                kernel_size = 1, bias = True)
        self.conv2 = conv_nd(self.out_chns, 1,
                kernel_size = 1, bias = True)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, x_l, x_h):
        input_shape = list(x_l.shape)
        gate_shape  = list(x_h.shape)

        # resize low-level feature to the shape of high-level feature
        mode = 'bilinear' if self.dim == 2 else 'trilinear'
        x_l_reshape = nn.functional.interpolate(x_l, size = gate_shape[2:], mode = mode)
        f_l = self.conv1_l(x_l_reshape)
        f_h = self.conv1_h(x_h)
        f = f_l + f_h
        f = self.act1(f)
        f = self.conv2(f)
        att = self.act2(f)
        # resize attention map to the shape of low-level feature
        att = nn.functional.interpolate(att, size = input_shape[2:], mode = mode)
        # return calibrated low-level feature
        output = att * x_l
        return output


class UpBlockWithAttention(UpBlock):
    """Upsampling followed by ConvBlock"""
    def __init__(self, dim, in_channels1, in_channels2, out_channels, dropout_p, up_mode = 2,
            kernel_size = 3, padding = 1, stride = 1, dilation = 1, norm_type = 'batch_norm'):
        """
        in_channels1: channel of high-level features
        in_channels2: channel of low-level features
        out_channels: output channel number
        dropout_p: probability of dropout
        """
        super(UpBlockWithAttention, self).__init__(dim, in_channels1, in_channels2, out_channels, dropout_p, 
            up_mode, kernel_size, padding, stride, dilation, norm_type)
        self.ag   = AttentionGateBlock(dim, in_channels2, in_channels1)

    def forward(self, x1, x2):
        x2at = self.ag(x2, x1)
        if self.up_mode != "transconv":
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2at, x1], dim=1)
        return self.conv(x)

class DecoderWithAttention(Decoder):
    """
    Decoder of 2D UNet.

    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param in_chns: (int) Input channel number.
    :param feature_chns: (list) Feature channel for each resolution level. 
      The length should be 4 or 5, such as [16, 32, 64, 128, 256].
    :param dropout: (list) The dropout ratio for each resolution level. 
      The length should be the same as that of `feature_chns`.
    :param class_num: (int) The class number for segmentation task. 
    :param up_mode: (string or int) The mode for upsampling. The allowed values are:
        0 (or `TransConv`), 1 (or `Nearest`), 2 (or `Bilinear`), 3 (or `Bicubic`). 
        The default value is 2 (or `Bilinear`).
    :param multiscale_pred: (bool) Get multi-scale prediction. 
    """
    def __init__(self, params):
        super(DecoderWithAttention, self).__init__(params)
        self.up1 = UpBlockWithAttention(self.dim, self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], self.dropout[3], 
            self.up_mode, norm_type = self.norm_type) 
        self.up2 = UpBlockWithAttention(self.dim, self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], self.dropout[2], 
            self.up_mode, norm_type = self.norm_type) 
        self.up3 = UpBlockWithAttention(self.dim, self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], self.dropout[1], 
            self.up_mode, norm_type = self.norm_type) 
        self.up4 = UpBlockWithAttention(self.dim, self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], self.dropout[0], 
            self.up_mode, norm_type = self.norm_type) 


class AttentionUNet(UNet):
    """
    A Reimplementation of the attention U-Net paper:
        Ozan Oktay, Jo Schlemper et al.:
        Attentin U-Net: Looking Where to Look for the Pancreas. MIDL, 2018.

    Note that there are some modifications from the original paper, such as
    the use of batch normalization, dropout, and leaky relu here.
    """
    def __init__(self, params):
        super(AttentionUNet, self).__init__(params)
        self.decoder  = DecoderWithAttention(params)    