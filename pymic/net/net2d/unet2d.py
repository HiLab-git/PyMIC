# -*- coding: utf-8 -*-
"""
An implementation of the U-Net paper:
    Olaf Ronneberger, Philipp Fischer, Thomas Brox:
    U-Net: Convolutional Networks for Biomedical Image Segmentation. 
    MICCAI (3) 2015: 234-241
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np 
from torch.nn.functional import interpolate

class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self,in_channels, out_channels, dropout_p):
        """
        dropout_p: probability to be zeroed
        """
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
       
    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """Upsampling followed by ConvBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        """
        in_channels1: channel of high-level features
        in_channels2: channel of low-level features
        out_channels: output channel number
        dropout_p: probability of dropout
        """
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size = 1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, params):
        super(UNet2D, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.dropout   = self.params['dropout']
        self.n_class   = self.params['class_num']
        self.bilinear  = self.params['bilinear']
        self.deep_sup  = self.params['deep_supervise']

        assert(len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

        self.in_conv= ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        if(len(self.ft_chns) == 5):
            self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])
            self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], 0.0, self.bilinear) 
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], 0.0, self.bilinear) 
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], 0.0, self.bilinear) 
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], 0.0, self.bilinear) 
    
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size = 1)
        if(self.deep_sup):
            self.out_conv1 = nn.Conv2d(self.ft_chns[1], self.n_class, kernel_size = 1)
            self.out_conv2 = nn.Conv2d(self.ft_chns[2], self.n_class, kernel_size = 1)
            self.out_conv3 = nn.Conv2d(self.ft_chns[3], self.n_class, kernel_size = 1)

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
        if(len(self.ft_chns) == 5):
          x4 = self.down4(x3)
          x_d3 = self.up1(x4, x3)
        else:
          x_d3 = x3
        x_d2 = self.up2(x_d3, x2)
        x_d1 = self.up3(x_d2, x1)
        x_d0 = self.up4(x_d1, x0)
        output = self.out_conv(x_d0)
        if(self.deep_sup):
            out_shape = list(output.shape)[2:]
            output1 = self.out_conv1(x_d1)
            output1 = interpolate(output1, out_shape, mode = 'bilinear')
            output2 = self.out_conv2(x_d2)
            output2 = interpolate(output2, out_shape, mode = 'bilinear')
            output3 = self.out_conv3(x_d3)
            output3 = interpolate(output3, out_shape, mode = 'bilinear')
            output = [output, output1, output2, output3]

            if(len(x_shape) == 5):
                new_shape = [N, D] + list(output[0].shape)[1:]
                for i in range(len(output)):
                    output[i] = torch.transpose(torch.reshape(output[i], new_shape), 1, 2)
        elif(len(x_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)

        return output

if __name__ == "__main__":
    params = {'in_chns':4,
              'feature_chns':[2, 8, 32, 48, 64],
              'dropout':  [0, 0, 0.3, 0.4, 0.5],
              'class_num': 2,
              'bilinear': True}
    Net = UNet2D(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 10, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    print(len(y.size()))
    y = y.detach().numpy()
    print(y.shape)
