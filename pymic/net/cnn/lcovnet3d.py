# -*- coding: utf-8 -*-
from __future__ import print_function, division

import logging
import torch
import torch.nn as nn
import numpy as np
from pymic.net.net_init import Initialization_He, Initialization_XavierUniform

class UnetBlock_Encode(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Encode, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_chns, self.out_chns, kernel_size=(1, 1, 3),
                      padding=(0, 0, 1)),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=(3, 3, 1),
                      padding=(1, 1, 0), groups=1),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.conv2_2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=4, stride=2, padding=1),
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=1,
                      padding=0),
            nn.BatchNorm3d(self.out_chns),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)

        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x2 = torch.sigmoid(x2)
        x = x1 + x2 * x
        return x


class UnetBlock_Encode_BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Encode_BottleNeck, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_chns, self.out_chns, kernel_size=(1, 1, 3),
                      padding=(0, 0, 1)),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=(3, 3, 1),
                      padding=(1, 1, 0), groups=self.out_chns),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.conv2_2 = nn.Sequential(
            # nn.AvgPool3d(kernel_size=4, stride=2),
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=1,
                      padding=0),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        x = self.conv1(x)

        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x2 = torch.sigmoid(x2)
        x = x1 + x2 * x
        return x


class UnetBlock_Down(nn.Module):
    def __init__(self):
        super(UnetBlock_Down, self).__init__()
        self.avg_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.avg_pool(x)
        return x


class UnetBlock_Up(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Up, self).__init__()
        self.conv = self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channel, kernel_size=1,
                      padding=0, groups=1),
            nn.BatchNorm3d(out_channel),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x


class LCOVNet(nn.Module):
    """
    An implementation of the LCOVNet.
        
    * Reference: Q. Zhao, L. Zhong, J. Xiao, J. Zhang, Y. Chen , W. Liao, S. Zhang, and G. Wang:
      Efficient Multi-Organ Segmentation From 3D Abdominal CT Images With Lightweight Network and Knowledge Distillation. 
      `IEEE TMI 42(9) 2023: 2513 - 2523. <https://ieeexplore.ieee.org/document/10083150/>`_

    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param in_chns: (int) Input channel number.
    :param feature_chns: (list) Feature channel for each resolution level. 
      The length should be 4 or 5, such as [16, 32, 64, 128, 256].
    :param dropout: (list) The dropout ratio for each resolution level. 
      The length should be the same as that of `feature_chns`.
    :param class_num: (int) The class number for segmentation task. 
    :param up_mode: (string or int) The mode for upsampling. The allowed values are:
        0 (or `TransConv`), 1 (`Nearest`), 2 (`Trilinear`). The default value
        is 2 (`Trilinear`).
    :param multiscale_pred: (bool) Get multi-scale prediction.
    """
    def __init__(self, params):
        super(LCOVNet, self).__init__()
        params = self.get_default_parameters(params)
        for p in params:
            print(p, params[p])
        self.stage    = 'train'
        # C_in=32, n_classes=17, m=1, is_ds=True):
        
        in_chns   = params['in_chns']
        n_class   = params['class_num']
        self.ft_chns  = params['feature_chns']
        self.mul_pred = params.get('multiscale_pred', False)
        
        self.Encode_block1 = UnetBlock_Encode(in_chns, self.ft_chns[0])
        self.down1 = UnetBlock_Down()

        self.Encode_block2 = UnetBlock_Encode(self.ft_chns[0], self.ft_chns[1])
        self.down2 = UnetBlock_Down()

        self.Encode_block3 = UnetBlock_Encode(self.ft_chns[1], self.ft_chns[2])
        self.down3 = UnetBlock_Down()

        self.Encode_block4 = UnetBlock_Encode(self.ft_chns[2], self.ft_chns[3])
        self.down4 = UnetBlock_Down()

        self.Encode_BottleNeck_block5 = UnetBlock_Encode_BottleNeck(
            self.ft_chns[3], self.ft_chns[4])

        self.up1 = UnetBlock_Up(self.ft_chns[4], self.ft_chns[3])
        self.Decode_block1 = UnetBlock_Encode(
            self.ft_chns[3]*2, self.ft_chns[3])
        self.segout1 = nn.Conv3d(
            self.ft_chns[3], n_class, kernel_size=1, padding=0)

        self.up2 = UnetBlock_Up(self.ft_chns[3], self.ft_chns[2])
        self.Decode_block2 = UnetBlock_Encode(
            self.ft_chns[2]*2, self.ft_chns[2])
        self.segout2 = nn.Conv3d(
            self.ft_chns[2], n_class, kernel_size=1, padding=0)

        self.up3 = UnetBlock_Up(self.ft_chns[2], self.ft_chns[1])
        self.Decode_block3 = UnetBlock_Encode(
            self.ft_chns[1]*2, self.ft_chns[1])
        self.segout3 = nn.Conv3d(
            self.ft_chns[1], n_class, kernel_size=1, padding=0)

        self.up4 = UnetBlock_Up(self.ft_chns[1], self.ft_chns[0])
        self.Decode_block4 = UnetBlock_Encode(
            self.ft_chns[0]*2, self.ft_chns[0])
        self.segout4 = nn.Conv3d(
            self.ft_chns[0], n_class, kernel_size=1, padding=0)

    def get_default_parameters(self, params):
        default_param = {
            'feature_chns': [32, 64, 128, 256, 512],
            'initialization': 'he',
            'multiscale_pred': False
        }
        for key in default_param:
            params[key] = params.get(key, default_param[key])
        for key in params:
                logging.info("{0:}  = {1:}".format(key, params[key]))
        return params
    
    def forward(self, x):
        _x1 = self.Encode_block1(x)
        x1 = self.down1(_x1)

        _x2 = self.Encode_block2(x1)
        x2 = self.down2(_x2)

        _x3 = self.Encode_block3(x2)
        x3 = self.down2(_x3)

        _x4 = self.Encode_block4(x3)
        x4 = self.down2(_x4)

        x5 = self.Encode_BottleNeck_block5(x4)

        x6 = self.up1(x5)
        x6 = torch.cat((x6, _x4), dim=1)
        x6 = self.Decode_block1(x6)
        segout1 = self.segout1(x6)

        x7 = self.up2(x6)
        x7 = torch.cat((x7, _x3), dim=1)
        x7 = self.Decode_block2(x7)
        segout2 = self.segout2(x7)

        x8 = self.up3(x7)
        x8 = torch.cat((x8, _x2), dim=1)
        x8 = self.Decode_block3(x8)
        segout3 = self.segout3(x8)

        x9 = self.up4(x8)
        x9 = torch.cat((x9, _x1), dim=1)
        x9 = self.Decode_block4(x9)
        segout4 = self.segout4(x9)

        if (self.mul_pred == True and self.stage == 'train'):
            return [segout4, segout3, segout2, segout1]
        else:
            return segout4