# -*- coding: utf-8 -*-
from __future__ import print_function, division

import logging
import torch
import torch.nn as nn
import numpy as np

ConvND      = {2: nn.Conv2d, 3: nn.Conv3d}
BatchNormND = {2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
MaxPoolND   = {2: nn.MaxPool2d, 3: nn.MaxPool3d}  
ConvTransND = {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}

class ConvBlockND(nn.Module):
    """
    2D or 3D convolutional block
    
    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dim: (int) Should be 2 or 3, for 2D and 3D convolution, respectively.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dim = 2, dropout_p = 0.0):
        super(ConvBlockND, self).__init__()
        assert(dim == 2 or dim == 3)
        self.dim = dim 
        self.conv_conv = nn.Sequential(
                ConvND[dim](in_channels, out_channels, kernel_size=3, padding=1),
                BatchNormND[dim](out_channels),
                nn.PReLU(),
                nn.Dropout(dropout_p),
                ConvND[dim](out_channels, out_channels, kernel_size=3, padding=1),
                BatchNormND[dim](out_channels),
                nn.PReLU()
            )

    def forward(self, x):
        output = self.conv_conv(x)
        return output 

class DownBlock(nn.Module):
    """`ConvBlockND` block followed by downsampling.

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dim: (int) Should be 2 or 3, for 2D and 3D convolution, respectively.
    :param dropout_p: (int) Dropout probability.
    :param downsample: (bool) Use downsample or not after convolution. 
    """
    def __init__(self, in_channels, out_channels, dim = 2, dropout_p = 0.0, downsample = True):
        super(DownBlock, self).__init__()
        self.downsample = downsample 
        self.dim = dim
        self.conv = ConvBlockND(in_channels, out_channels, dim, dropout_p)
        self.down_layer = MaxPoolND[dim](kernel_size = 2, stride = 2)
    
    def forward(self, x):
        x_shape = list(x.shape)
        if(self.dim == 2 and len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        output = self.conv(x)
        if(self.downsample):
            output_d = self.down_layer(output)
        else:
            output_d = None 
        if(self.dim == 2 and len(x_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)
            if(self.downsample):
                new_shape = [N, D] + list(output_d.shape)[1:]
                output_d = torch.reshape(output_d, new_shape)
                output_d = torch.transpose(output_d, 1, 2)

        return output, output_d

class UpBlock(nn.Module):
    """Upsampling followed by `ConvBlockND` block
    
    :param in_channels1: (int) Input channel number for low-resolution feature map.
    :param in_channels2: (int) Input channel number for high-resolution feature map.
    :param out_channels: (int) Output channel number.
    :param dim: (int) Should be 2 or 3, for 2D and 3D convolution, respectively.
    :param dropout_p: (int) Dropout probability.
    :param up_mode: (string or int) The mode for upsampling. The allowed values are:
        0 (or `TransConv`), 1 (`Nearest`), 2 (`Trilinear` for 3D and `Bilinear` for 2D). 
        The default value is 2.
    """
    def __init__(self, in_channels1, in_channels2, out_channels, 
                 dim = 2, dropout_p = 0.0, up_mode= 2):
        super(UpBlock, self).__init__()
        if(isinstance(up_mode, int)):
            up_mode_values = ["transconv", "nearest", "trilinear"]
            if(up_mode > 2):
                raise ValueError("The upsample mode should be 0-2, but {0:} is given.".format(up_mode))
            self.up_mode = up_mode_values[up_mode]
        else:
            self.up_mode = up_mode.lower()

        self.dim = dim
        if (self.up_mode == "transconv"):
            self.up = ConvTransND[dim](in_channels1, in_channels2, kernel_size=2, stride=2)
        else:    
            self.conv1x1 = ConvND[dim](in_channels1, in_channels2, kernel_size = 1)
            if(self.up_mode == "nearest"):
                self.up = nn.Upsample(scale_factor=2, mode=self.up_mode)
            else:
                mode = "trilinear" if dim == 3 else "bilinear"
                self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        self.conv = ConvBlockND(in_channels2 * 2, out_channels, dim, dropout_p)

    def forward(self, x1, x2):
        x1_shape = list(x1.shape)
        x2_shape = list(x2.shape)
        if(self.dim == 2 and len(x1_shape) == 5):
            [N, C, D, H, W] = x1_shape
            new_shape = [N*D, C, H, W]
            x1 = torch.transpose(x1, 1, 2)
            x1 = torch.reshape(x1, new_shape)
            [N, C, D, H, W] = x2_shape
            new_shape = [N*D, C, H, W]
            x2 = torch.transpose(x2, 1, 2)
            x2 = torch.reshape(x2, new_shape)

        if self.up_mode != "transconv":
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        output = torch.cat([x2, x1], dim=1)
        output = self.conv(output)
        if(self.dim == 2 and len(x1_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)
        return output  

class Encoder(nn.Module):
    """
    A modification of the encoder of 3D UNet by using ConvScSEBlock3D

    Parameters are given in the `params` dictionary.
    See :mod:`pymic.net.net3d.unet3d.Encoder` for details. 
    """
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.n_class   = self.params['class_num']
        self.ft_chns   = self.params['feature_chns']
        self.dropout   = self.params['dropout']
        self.dims      = self.params['conv_dims']
        
        self.block0 = DownBlock(self.in_chns, self.ft_chns[0], self.dims[0], self.dropout[0], True)
        self.block1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dims[1], self.dropout[1], True)
        self.block2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dims[2], self.dropout[2], True)
        self.block3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dims[3], self.dropout[3], True)
        self.block4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dims[4], self.dropout[4], False)

    def forward(self, x):
        x0, x0_d = self.block0(x)
        x1, x1_d = self.block1(x0_d)
        x2, x2_d = self.block2(x1_d)
        x3, x3_d = self.block3(x2_d)
        x4, x4_d = self.block4(x3_d)
        return [x0, x1, x2, x3, x4]

class Decoder(nn.Module):
    """
    Decoder of 3D UNet.

    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param in_chns: (int) Input channel number.
    :param feature_chns: (list) Feature channel for each resolution level. 
      The length should be 4 or 5, such as [16, 32, 64, 128, 256].
    :param dropout: (list) The dropout ratio for each resolution level. 
      The length should be the same as that of `feature_chns`.
    :param class_num: (int) The class number for segmentation task. 
    :param up_mode: (string or int) The mode for upsampling. The allowed values are:
        0 (or `TransConv`), 1 (`Nearest`), 2 (`Trilinear` for 3D and `Bilinear` for 2D). 
        The default value is 2.
    :param multiscale_pred: (bool) Get multi-scale prediction. 
    """
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.n_class   = self.params['class_num']
        self.ft_chns   = self.params['feature_chns']
        self.dropout   = self.params['dropout']
        self.dims      = self.params['conv_dims']
        self.up_mode   = self.params.get('up_mode', 2)
        self.mul_pred  = self.params.get('multiscale_pred', False)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], 
                    self.dims[3], dropout_p = self.dropout[3], up_mode=self.up_mode) 
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], 
                    self.dims[2], dropout_p = self.dropout[2], up_mode=self.up_mode) 
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], 
                    self.dims[1], dropout_p = self.dropout[1], up_mode=self.up_mode) 
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], 
                    self.dims[0], dropout_p = self.dropout[0], up_mode=self.up_mode) 

        self.out_conv = nn.Conv3d(self.ft_chns[0], self.n_class, kernel_size = 1)
        if(self.mul_pred):
            self.out_conv1 = nn.Conv3d(self.ft_chns[1], self.n_class, kernel_size = 1)
            self.out_conv2 = nn.Conv3d(self.ft_chns[2], self.n_class, kernel_size = 1)
            self.out_conv3 = nn.Conv3d(self.ft_chns[3], self.n_class, kernel_size = 1)
        self.stage = 'train'

    def set_stage(self, stage):
        self.stage = stage
    
    def forward(self, x):
        x0, x1, x2, x3, x4 = x
        x_d3 = self.up1(x4, x3)
        x_d2 = self.up2(x_d3, x2)
        x_d1 = self.up3(x_d2, x1)
        x_d0 = self.up4(x_d1, x0)
        output = self.out_conv(x_d0)
        if(self.mul_pred and self.stage == 'train'):
            output1 = self.out_conv1(x_d1)
            output2 = self.out_conv2(x_d2)
            output3 = self.out_conv3(x_d3)
            output  = [output, output1, output2, output3]
        return output

class UNet2D5(nn.Module):
    """
    A 2.5D network combining 3D convolutions with 2D convolutions.

    * Reference: Guotai Wang, Jonathan Shapey, Wenqi Li, Reuben Dorent, Alex Demitriadis, 
      Sotirios Bisdas, Ian Paddick, Robert Bradford, Shaoting Zhang, Sébastien Ourselin, 
      Tom Vercauteren: Automatic Segmentation of Vestibular Schwannoma from T2-Weighted 
      MRI by Deep Spatial Attention with Hardness-Weighted Loss. 
      `MICCAI (2) 2019: 264-272. <https://link.springer.com/chapter/10.1007/978-3-030-32245-8_30>`_
    
    Note that the attention module in the orininal paper is not used here.

    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param in_chns: (int) Input channel number.
    :param feature_chns: (list) Feature channel for each resolution level. 
      The length should be 5, such as [16, 32, 64, 128, 256].
    :param dropout: (list) The dropout ratio for each resolution level. 
      The length should be the same as that of `feature_chns`.
    :param conv_dims: (list) The convolution dimension (2 or 3) for each resolution level. 
      The length should be the same as that of `feature_chns`.
    :param class_num: (int) The class number for segmentation task. 
    :param up_mode: (string or int) The mode for upsampling. The allowed values are:
        0 (or `TransConv`), 1 (`Nearest`), 2 (`Trilinear`). The default value
        is 2 (`Trilinear`).
    :param multiscale_pred: (bool) Get multi-scale prediction. 
    """
    def __init__(self, params):
        super(UNet2D5, self).__init__()
        params = self.get_default_parameters(params)
        for p in params:
            print(p, params[p])
        self.stage    = 'train'
        self.encoder  = Encoder(params)
        self.decoder  = Decoder(params)    
      
    def get_default_parameters(self, params):
        default_param = {
            'feature_chns': [32, 64, 128, 256, 512],
            'dropout':  [0.0, 0.0, 0.2, 0.3, 0.4],
            'conv_dims':[2, 2, 3, 3, 3],
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
        f = self.encoder(x)
        output = self.decoder(f)
        return output
