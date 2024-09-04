# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
from pymic.net.net3d.unet3d import UpBlock, Encoder, Decoder, UNet3D
from pymic.net.net3d.scse3d import *

class ConvScSEBlock3D(nn.Module):
    """
    Two 3D convolutional blocks followed by `ChannelSpatialSELayer3D`.
    Each block consists of `Conv3d` + `BatchNorm3d` + `LeakyReLU`.
    A dropout layer is used between the wo blocks.

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvScSEBlock3D, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            ChannelSpatialSELayer3D(out_channels)
        )
       
    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """3D Downsampling followed by `ConvScSEBlock3D`.

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ConvScSEBlock3D(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlockScSE(UpBlock):
    """3D Up-sampling followed by `ConvScSEBlock3D` in UNet3D_ScSE.
    
    :param in_channels1: (int) Input channel number for low-resolution feature map.
    :param in_channels2: (int) Input channel number for high-resolution feature map.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    :param up_mode: (string or int) The mode for upsampling. The allowed values are:
        0 (or `TransConv`), 1 (`Nearest`), 2 (`Trilinear`). The default value
        is 2 (`Trilinear`).
    """
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,  up_mode=2):
        super(UpBlockScSE, self).__init__(in_channels1, in_channels2, 
            out_channels, dropout_p, up_mode)
        self.conv = ConvScSEBlock3D(in_channels2 * 2, out_channels, dropout_p)


class EncoderScSE(Encoder):
    """
    A modification of the encoder of 3D UNet by using ConvScSEBlock3D

    Parameters are given in the `params` dictionary.
    See :mod:`pymic.net.net3d.unet3d.Encoder` for details. 
    """
    def __init__(self, params):
        super(EncoderScSE, self).__init__(params)
        
        self.in_conv= ConvScSEBlock3D(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        if(len(self.ft_chns) == 5):
            self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

class DecoderScSE(Decoder):
    """
    A modification of the decoder of 3D UNet by using ConvScSEBlock3D

    Parameters are given in the `params` dictionary.
    See :mod:`pymic.net.net3d.unet3d.Decoder` for details. 
    """
    def __init__(self, params):
        super(DecoderScSE, self).__init__(params)

        if(len(self.ft_chns) == 5):
            self.up1 = UpBlockScSE(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], self.dropout[3], self.up_mode) 
        self.up2 = UpBlockScSE(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], self.dropout[2], self.up_mode) 
        self.up3 = UpBlockScSE(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], self.dropout[1], self.up_mode) 
        self.up4 = UpBlockScSE(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], self.dropout[0], self.up_mode) 


class UNet3D_ScSE(UNet3D):
    """
    Combining 3D U-Net with SCSE module.

    * Reference: Abhijit Guha Roy, Nassir Navab, Christian Wachinger:
      Recalibrating Fully Convolutional Networks With Spatial and Channel 
      "Squeeze and Excitation" Blocks. 
      `IEEE Trans. Med. Imaging 38(2): 540-549 (2019). <https://ieeexplore.ieee.org/document/8447284>`_

    Parameters are given in the `params` dictionary.
    See :mod:`pymic.net.net3d.unet3d.UNet3D` for details. 
    """
    def __init__(self, params):
        super(UNet3D_ScSE, self).__init__(params)
        self.encoder  = EncoderScSE(params)
        self.decoder  = DecoderScSE(params)   
