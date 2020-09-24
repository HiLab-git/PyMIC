# -*- coding: utf-8 -*-
"""
Author: Guotai Wang
Date:   12 June, 2020
Implementation of of COPLENet for COVID-19 pneumonia lesion segmentation from CT images.
Reference: 
    G. Wang et al. A Noise-robust Framework for Automatic Segmentation of COVID-19 Pneumonia Lesions 
    from CT Images. IEEE Transactions on Medical Imaging, 39(8),2020:2653-2663. DOI:10.1109/TMI.2020.3000314.
"""

from __future__ import print_function, division
import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1):
        super(ConvLayer, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
       
    def forward(self, x):
        return self.conv(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, r):
        super(SEBlock, self).__init__()

        redu_chns = int(in_channels / r)
        self.se_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, redu_chns, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(redu_chns, in_channels, kernel_size=1, padding=0),
            nn.ReLU())
        
    def forward(self, x):
        f = self.se_layers(x)
        return f*x + x

class ASPPBlock(nn.Module):
    def __init__(self,in_channels, out_channels_list, kernel_size_list, dilation_list):
        super(ASPPBlock, self).__init__()
        self.conv_num = len(out_channels_list)
        assert(self.conv_num == 4)
        assert(self.conv_num == len(kernel_size_list) and self.conv_num == len(dilation_list))
        pad0 = int((kernel_size_list[0] - 1) / 2 * dilation_list[0])
        pad1 = int((kernel_size_list[1] - 1) / 2 * dilation_list[1])
        pad2 = int((kernel_size_list[2] - 1) / 2 * dilation_list[2])
        pad3 = int((kernel_size_list[3] - 1) / 2 * dilation_list[3])
        self.conv_1 = nn.Conv2d(in_channels, out_channels_list[0], kernel_size = kernel_size_list[0], 
                    dilation = dilation_list[0], padding = pad0 )
        self.conv_2 = nn.Conv2d(in_channels, out_channels_list[1], kernel_size = kernel_size_list[1], 
                    dilation = dilation_list[1], padding = pad1 )
        self.conv_3 = nn.Conv2d(in_channels, out_channels_list[2], kernel_size = kernel_size_list[2], 
                    dilation = dilation_list[2], padding = pad2 )
        self.conv_4 = nn.Conv2d(in_channels, out_channels_list[3], kernel_size = kernel_size_list[3], 
                    dilation = dilation_list[3], padding = pad3 )

        out_channels = out_channels_list[0] + out_channels_list[1] + out_channels_list[2] + out_channels_list[3] 
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU())
       
    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        y = torch.cat([x1, x2, x3, x4], dim=1)
        y = self.conv_1x1(y)
        return y

class ConvBNActBlock(nn.Module):
    """Two convolution layers with batch norm, leaky relu, dropout and SE block"""
    def __init__(self,in_channels, out_channels, dropout_p):
        super(ConvBNActBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            SEBlock(out_channels, 2)
        )
       
    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling by a concantenation of max-pool and avg-pool, followed by ConvBNActBlock
    """
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        self.conv = ConvBNActBlock(2 * in_channels, out_channels, dropout_p)
        
    def forward(self, x):
        x_max = self.maxpool(x)
        x_avg = self.avgpool(x)
        x_cat = torch.cat([x_max, x_avg], dim=1)
        y = self.conv(x_cat)
        return y + x_cat

class UpBlock(nn.Module):
    """Upssampling followed by ConvBNActBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, 
                 bilinear=True, dropout_p = 0.5):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size = 1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBNActBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1    = self.up(x1)
        x_cat = torch.cat([x2, x1], dim=1)
        y     = self.conv(x_cat)
        return y + x_cat

class COPLENet(nn.Module):
    def __init__(self, params):
        super(COPLENet, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.n_class   = self.params['class_num']
        self.bilinear  = self.params['bilinear']
        self.dropout   = self.params['dropout']
        assert(len(self.ft_chns) == 5)

        f0_half = int(self.ft_chns[0] / 2)
        f1_half = int(self.ft_chns[1] / 2)
        f2_half = int(self.ft_chns[2] / 2)
        f3_half = int(self.ft_chns[3] / 2)
        self.in_conv= ConvBNActBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])
        
        self.bridge0= ConvLayer(self.ft_chns[0], f0_half)
        self.bridge1= ConvLayer(self.ft_chns[1], f1_half)
        self.bridge2= ConvLayer(self.ft_chns[2], f2_half)
        self.bridge3= ConvLayer(self.ft_chns[3], f3_half)

        self.up1    = UpBlock(self.ft_chns[4], f3_half, self.ft_chns[3], dropout_p = self.dropout[3])
        self.up2    = UpBlock(self.ft_chns[3], f2_half, self.ft_chns[2], dropout_p = self.dropout[2])
        self.up3    = UpBlock(self.ft_chns[2], f1_half, self.ft_chns[1], dropout_p = self.dropout[1])
        self.up4    = UpBlock(self.ft_chns[1], f0_half, self.ft_chns[0], dropout_p = self.dropout[0])

        f4 = self.ft_chns[4]
        aspp_chns = [int(f4 / 4), int(f4 / 4), int(f4 / 4), int(f4 / 4)]
        aspp_knls = [1, 3, 3, 3]
        aspp_dila = [1, 2, 4, 6]
        self.aspp = ASPPBlock(f4, aspp_chns, aspp_knls, aspp_dila)
        
            
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,  
            kernel_size = 3, padding = 1)

    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
          [N, C, D, H, W] = x_shape
          new_shape = [N*D, C, H, W]
          x = torch.transpose(x, 1, 2)
          x = torch.reshape(x, new_shape)
        x0  = self.in_conv(x)
        x0b = self.bridge0(x0)
        x1  = self.down1(x0)
        x1b = self.bridge1(x1)
        x2  = self.down2(x1)
        x2b = self.bridge2(x2)
        x3  = self.down3(x2)
        x3b = self.bridge3(x3)
        x4  = self.down4(x3)
        x4  = self.aspp(x4) 

        x = self.up1(x4, x3b)
        x = self.up2(x, x2b)
        x = self.up3(x, x1b)
        x = self.up4(x, x0b)
        output = self.out_conv(x)

        if(len(x_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)
        return output