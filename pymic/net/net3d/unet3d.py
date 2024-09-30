# -*- coding: utf-8 -*-
from __future__ import print_function, division

import logging
import torch
import torch.nn as nn
import numpy as np
from pymic.net.net_init import Initialization_He, Initialization_XavierUniform


class ConvBlock(nn.Module):
    """
    Two 3D convolution layers with batch norm and leaky relu.
    Droput is used between the two convolution layers.
    
    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dropout_p, norm_type = 'batch_norm'):
        super(ConvBlock, self).__init__()
        if(norm_type == 'batch_norm'):
            norm1 = nn.BatchNorm3d(out_channels, affine = True)
            norm2 = nn.BatchNorm3d(out_channels, affine = True)
        elif(norm_type == 'instance_norm'):
            norm1 = nn.InstanceNorm3d(out_channels, affine = True)
            norm2 = nn.InstanceNorm3d(out_channels, affine = True)
        else:
            raise ValueError("norm_type {0:} not supported, it should be batch_norm or instance_norm".format(norm_type))
        self.conv_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            norm1,
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            norm2,
            nn.LeakyReLU()
        )
       
    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """
    3D downsampling followed by ConvBlock

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dropout_p, norm_type = 'batch_norm'):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock(in_channels, out_channels, dropout_p, norm_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """
    3D upsampling followed by ConvBlock
    
    :param in_channels1: (int) Channel number of high-level features.
    :param in_channels2: (int) Channel number of low-level features.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    :param up_mode: (string or int) The mode for upsampling. The allowed values are:
        0 (or `TransConv`), 1 (`Nearest`), 2 (`Trilinear`). The default value
        is 2 (`Trilinear`).
    """
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, 
            up_mode=2, norm_type = 'batch_norm'):
        super(UpBlock, self).__init__()
        if(isinstance(up_mode, int)):
            up_mode_values = ["transconv", "nearest", "trilinear"]
            if(up_mode > 2):
                raise ValueError("The upsample mode should be 0-2, but {0:} is given.".format(up_mode))
            self.up_mode = up_mode_values[up_mode]
        else:
            self.up_mode = up_mode.lower()

        if (self.up_mode == "transconv"):
            self.up = nn.ConvTranspose3d(in_channels1, in_channels2, kernel_size=2, stride=2)
        else:    
            self.conv1x1 = nn.Conv3d(in_channels1, in_channels2, kernel_size = 1)
            if(self.up_mode == "nearest"):
                self.up = nn.Upsample(scale_factor=2, mode=self.up_mode)
            else:
                self.up = nn.Upsample(scale_factor=2, mode=self.up_mode, align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p, norm_type)

    def forward(self, x1, x2):
        if self.up_mode != "transconv":
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Encoder(nn.Module):
    """
    Encoder of 3D UNet.

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
        in_chns   = self.params['in_chns']
        ft_chns   = self.params['feature_chns']
        dropout   = self.params['dropout']
        norm_type = self.params['norm_type']
        assert(len(ft_chns) == 5 or len(ft_chns) == 4)

        self.ft_chns= ft_chns
        self.in_conv= ConvBlock(in_chns,    ft_chns[0], dropout[0], norm_type)
        self.down1  = DownBlock(ft_chns[0], ft_chns[1], dropout[1], norm_type)
        self.down2  = DownBlock(ft_chns[1], ft_chns[2], dropout[2], norm_type)
        self.down3  = DownBlock(ft_chns[2], ft_chns[3], dropout[3], norm_type)
        if(len(ft_chns) == 5):
            self.down4  = DownBlock(ft_chns[3], ft_chns[4], dropout[4])

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
        0 (or `TransConv`), 1 (`Nearest`), 2 (`Trilinear`). The default value
        is 2 (`Trilinear`).
    :param multiscale_pred: (bool) Get multi-scale prediction. 
    """
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params    = params
        ft_chns   = self.params['feature_chns']
        dropout   = self.params['dropout']
        n_class   = self.params['class_num']
        norm_type = self.params['norm_type']
        up_mode   = self.params.get('up_mode', 2)
        self.ft_chns  = ft_chns 
        self.mul_pred = self.params.get('multiscale_pred', False)
        assert(len(ft_chns) == 5 or len(ft_chns) == 4)

        if(len(ft_chns) == 5):
            self.up1 = UpBlock(ft_chns[4], ft_chns[3], ft_chns[3], dropout[3], up_mode, norm_type) 
        self.up2 = UpBlock(ft_chns[3], ft_chns[2], ft_chns[2], dropout[2], up_mode, norm_type) 
        self.up3 = UpBlock(ft_chns[2], ft_chns[1], ft_chns[1], dropout[1], up_mode, norm_type) 
        self.up4 = UpBlock(ft_chns[1], ft_chns[0], ft_chns[0], dropout[0], up_mode, norm_type) 
        self.out_conv = nn.Conv3d(ft_chns[0], n_class, kernel_size = 1)

        if(self.mul_pred):
            self.out_conv1 = nn.Conv3d(ft_chns[1], n_class, kernel_size = 1)
            self.out_conv2 = nn.Conv3d(ft_chns[2], n_class, kernel_size = 1)
            self.out_conv3 = nn.Conv3d(ft_chns[3], n_class, kernel_size = 1)
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

class UNet3D(nn.Module):
    """
    An implementation of the U-Net.
        
    * Reference: Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
      3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. 
      `MICCAI (2) 2016: 424-432. <https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49>`_
    
    Note that there are some modifications from the original paper, such as
    the use of batch normalization, dropout, leaky relu and deep supervision.

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
        super(UNet3D, self).__init__()
        params = self.get_default_parameters(params)
        for p in params:
            print(p, params[p])
        self.stage    = 'train'
        self.tune_mode= params.get('finetune_mode', 'all')
        self.load_mode= params.get('weights_load_mode', 'all')
        self.encoder  = Encoder(params)
        self.decoder  = Decoder(params) 

        init = params['initialization'].lower()
        weightInitializer =  Initialization_He(1e-2) if init == 'he' else Initialization_XavierUniform()
        self.apply(weightInitializer)

    def get_default_parameters(self, params):
        default_param = {
            'feature_chns': [32, 64, 128, 256, 512],
            'dropout': [0.0, 0.0, 0.2, 0.3, 0.4],
            'up_mode': 2,
            'initialization': 'he',
            'norm_type': 'batch_norm',
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

    def get_parameters_to_update(self):
        if(self.tune_mode == "all"):
            return self.parameters()
        elif(self.tune_mode == "decoder"):
            print("only update parameters in decoder")
            params = self.decoder.parameters()
            return params
        else:
            raise(ValueError("update_mode can only be 'all' or 'decoder'."))

    def get_parameters_to_load(self):
        state_dict = self.state_dict()
        if(self.load_mode == 'encoder'): 
            state_dict = {k:v for k, v in state_dict.items() if "encoder" in k }
        return state_dict
