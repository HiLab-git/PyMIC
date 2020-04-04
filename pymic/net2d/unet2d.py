# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self,in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )
       
    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, 
                 bilinear=True, dropout = False):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        self.dropout  = dropout
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size = 1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        if dropout:
            self.drop = nn.Dropout(0.5)
        self.conv = ConvBlock(in_channels2 * 2, out_channels)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        if self.dropout:
            x = self.drop(x)
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, params):
        super(UNet2D, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.n_class   = self.params['class_num']
        self.bilinear  = self.params['bilinear']
        self.dropout   = self.params['dropout']
        assert(len(self.ft_chns) == 5)

        self.in_conv= ConvBlock(self.in_chns, self.ft_chns[0])
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1])
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2])
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3])
        self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4])
        self.up1    = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout = self.dropout)
        self.up2    = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout = self.dropout)
        self.up3    = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1])
        self.up4    = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0])
    
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,  
            kernel_size = 3, padding = 1)

    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
          [N, C, D, H, W] = x_shape
          new_shape = [N*D, C, H, W]
          x = torch.transpose(x, 1, 2)
          x = torch.reshape(x, new_shape)
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)

        if(len(x_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)
        return output
