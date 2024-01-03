# -*- coding: utf-8 -*-
from __future__ import print_function, division

import logging
import torch
import torch.nn as nn
import numpy as np 

class ConvBlock(nn.Module):
    """
    Two convolution layers with batch norm and leaky relu.
    Droput is used between the two convolution layers.
    
    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self,in_channels, out_channels, dropout_p):
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
    """
    Downsampling followed by ConvBlock

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """
    Upsampling followed by ConvBlock
    
    :param in_channels1: (int) Channel number of high-level features.
    :param in_channels2: (int) Channel number of low-level features.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    :param up_mode: (string or int) The mode for upsampling. The allowed values are:
        0 (or `TransConv`), 1 (`Nearest`), 2 (`Bilinear`), 3 (`Bicubic`). The default value
        is 2 (`Bilinear`).
    """
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, up_mode = 2):
        super(UpBlock, self).__init__()
        if(isinstance(up_mode, int)):
            up_mode_values = ["transconv", "nearest", "bilinear", "bicubic"]
            if(up_mode > 3):
                raise ValueError("The upsample mode should be 0-3, but {0:} is given.".format(up_mode))
            self.up_mode = up_mode_values[up_mode]
        else:
            self.up_mode = up_mode.lower()

        if (self.up_mode == "transconv"):
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        else:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size = 1)
            if(self.up_mode == "nearest"):
                self.up = nn.Upsample(scale_factor=2, mode=self.up_mode)
            else:
                self.up = nn.Upsample(scale_factor=2, mode=self.up_mode, align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.up_mode != "transconv":
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Encoder(nn.Module):
    """
    Encoder of 2D UNet.

    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param in_chns: (int) Input channel number.
    :param feature_chns: (list) Feature channel for each resolution level. 
      The length should be 4 or 5, such as [16, 32, 64, 128, 256].
    :param dropout: (list) The dropout ratio for each resolution level. 
      The length should be the same as that of `feature_chns`.
    """
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.dropout   = self.params['dropout']
        assert(len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

        self.in_conv= ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        if(len(self.ft_chns) == 5):
            self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        output = [x0, x1, x2, x3]
        if(len(self.ft_chns) == 5):
          x4 = self.down4(x3)
          output.append(x4)
        return output

class Decoder(nn.Module):
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
        super(Decoder, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.dropout   = self.params['dropout']
        self.n_class   = self.params['class_num']
        self.up_mode   = self.params.get('up_mode', 2)
        self.mul_pred  = self.params.get('multiscale_pred', False)

        assert(len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

        if(len(self.ft_chns) == 5):
            self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], self.dropout[3], self.up_mode) 
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], self.dropout[2], self.up_mode) 
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], self.dropout[1], self.up_mode) 
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], self.dropout[0], self.up_mode) 
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size = 1)

        if(self.mul_pred and (self.training or self.mul_infer)):
            self.out_conv1 = nn.Conv2d(self.ft_chns[1], self.n_class, kernel_size = 1)
            self.out_conv2 = nn.Conv2d(self.ft_chns[2], self.n_class, kernel_size = 1)
            self.out_conv3 = nn.Conv2d(self.ft_chns[3], self.n_class, kernel_size = 1)
        self.stage = 'train'

    def set_stage(self, stage):
        self.stage = stage

    def forward(self, x):
        if(len(self.ft_chns) == 5):
            assert(len(x) == 5)
            x0, x1, x2, x3, x4 = x 
            x_d3 = self.up1(x4, x3)
        else:
            assert(len(x) == 4)
            x0, x1, x2, x3 = x 
            x_d3 = x3
        x_d2 = self.up2(x_d3, x2)
        x_d1 = self.up3(x_d2, x1)
        x_d0 = self.up4(x_d1, x0)
        output = self.out_conv(x_d0)
        if(self.mul_pred and self.stage == 'train'):
            output1 = self.out_conv1(x_d1)
            output2 = self.out_conv2(x_d2)
            output3 = self.out_conv3(x_d3)
            output = [output, output1, output2, output3]
        return output

class UNet2D(nn.Module):
    """
    An implementation of 2D U-Net.

    * Reference: Olaf Ronneberger, Philipp Fischer, Thomas Brox:
      U-Net: Convolutional Networks for Biomedical Image Segmentation. 
      MICCAI (3) 2015: 234-241
    
    Note that there are some modifications from the original paper, such as
    the use of batch normalization, dropout, leaky relu and deep supervision.

    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param in_chns: (int) Input channel number.
    :param class_num: (int) The class number for segmentation task. 

    Optional parameters:

    :param feature_chns: (list) Feature channel for each resolution level. 
      The length should be 4 or 5, such as [16, 32, 64, 128, 256].
    :param dropout: (list) The dropout ratio for each resolution level. 
      The length should be the same as that of `feature_chns`.
    :param up_mode: (string or int) The mode for upsampling. The allowed values are:
        0 (or `TransConv`), 1 (or `Nearest`), 2 (or `Bilinear`), 3 (or `Bicubic`). 
        The default value is 2 (or `Bilinear`).
    :param multiscale_pred: (bool) Get multiscale prediction.
    """
    def __init__(self, params):
        super(UNet2D, self).__init__()
        params = self.get_default_parameters(params)
        for p in params:
            print(p, params[p])
        self.encoder  = Encoder(params)
        self.decoder  = Decoder(params)    
      
    def get_default_parameters(self, params):
        default_param = {
            'feature_chns': [32, 64, 128, 256, 512],
            'dropout': [0.0, 0.0, 0.2, 0.3, 0.4],
            'up_mode': 2,
            'multiscale_pred': False
        }
        for key in default_param:
            params[key] = params.get(key, default_param[key])
        for key in params:
                logging.info("{0:}  = {1:}".format(key, params[key]))
        return params

    def set_stage(self, stage):
        self.stage = stage
        self.decoder.set_stage(stage)

    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
          [N, C, D, H, W] = x_shape
          new_shape = [N*D, C, H, W]
          x = torch.transpose(x, 1, 2)
          x = torch.reshape(x, new_shape)

        f = self.encoder(x)
        output = self.decoder(f)
        if(len(x_shape) == 5):
            if(isinstance(output, (list,tuple))):
                for i in range(len(output)):
                    new_shape = [N, D] + list(output[i].shape)[1:]
                    output[i] = torch.transpose(torch.reshape(output[i], new_shape), 1, 2)
            else:
                new_shape = [N, D] + list(output.shape)[1:]
                output = torch.transpose(torch.reshape(output, new_shape), 1, 2)
            
        return output