# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.net.net2d.unet2d import *
"""
A Reimplementation of the attention U-Net paper:
    Ozan Oktay, Jo Schlemper et al.:
    Attentin U-Net: Looking Where to Look for the Pancreas. MIDL, 2018.

Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
"""
class AttentionGateBlock(nn.Module):
    def __init__(self, chns_l, chns_h):
        """
        chns_l: channel number of low-level features from the encoder
        chns_h: channel number of high-level features from the decoder
        """
        super(AttentionGateBlock, self).__init__()
        self.in_chns_l = chns_l
        self.in_chns_h = chns_h

        self.out_chns = int(min(self.in_chns_l, self.in_chns_h)/2)
        self.conv1_l = nn.Conv2d(self.in_chns_l, self.out_chns,
                kernel_size = 1, bias = True)
        self.conv1_h = nn.Conv2d(self.in_chns_h, self.out_chns,
                kernel_size = 1, bias = True)
        self.conv2 = nn.Conv2d(self.out_chns, 1,
                kernel_size = 1, bias = True)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, x_l, x_h):
        input_shape = list(x_l.shape)
        gate_shape  = list(x_h.shape)

        # resize low-level feature to the shape of high-level feature
        x_l_reshape = nn.functional.interpolate(x_l, size = gate_shape[2:], mode = 'bilinear')
        f_l = self.conv1_l(x_l_reshape)
        f_h = self.conv1_h(x_h)
        f = f_l + f_h
        f = self.act1(f)
        f = self.conv2(f)
        att = self.act2(f)
        # resize attention map to the shape of low-level feature
        att = nn.functional.interpolate(att, size = input_shape[2:], mode = 'bilinear')
        # return calibrated low-level feature
        output = att * x_l
        return output


class UpBlockWithAttention(nn.Module):
    """Upsampling followed by ConvBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        """
        in_channels1: channel of high-level features
        in_channels2: channel of low-level features
        out_channels: output channel number
        dropout_p: probability of dropout
        """
        super(UpBlockWithAttention, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size = 1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)
        self.ag   = AttentionGateBlock(in_channels2, in_channels1)

    def forward(self, x1, x2):
        x2at = self.ag(x2, x1)
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2at, x1], dim=1)
        return self.conv(x)

class AttentionUNet2D(UNet2D):
    def __init__(self, params):
        super(AttentionUNet2D, self).__init__(params)
        self.up1    = UpBlockWithAttention(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p = 0.0)
        self.up2    = UpBlockWithAttention(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p = 0.0)
        self.up3    = UpBlockWithAttention(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p = 0.0)
        self.up4    = UpBlockWithAttention(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p = 0.0)

