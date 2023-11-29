# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np 
from pymic.net.net2d.unet2d import UpBlock, Encoder, Decoder, UNet2D
from pymic.net.net2d.scse2d import *

class ConvScSEBlock(nn.Module):
    """
    Two convolutional blocks followed by `ChannelSpatialSELayer`.
    Each block consists of `Conv2d` + `BatchNorm2d` + `LeakyReLU`.
    A dropout layer is used between the wo blocks.

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvScSEBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            ChannelSpatialSELayer(out_channels)
        )
       
    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by `ConvScSEBlock`.

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvScSEBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlockScSE(UpBlock):
    """Up-sampling followed by `ConvScSEBlock` in U-Net structure.
    
    The parameters for the backbone should be given in the `params` dictionary. 
    See :mod:`pymic.net.net2d.unet2d.UpBlock` for details. 
    """
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, up_mode = 2):
        super(UpBlockScSE, self).__init__(in_channels1, in_channels2, out_channels, dropout_p, up_mode)
        self.conv = ConvScSEBlock(in_channels2 * 2, out_channels, dropout_p)

class EncoderScSE(Encoder):
    """
    Encoder of 2D UNet with ScSE.

    The parameters for the backbone should be given in the `params` dictionary. 
    See :mod:`pymic.net.net2d.unet2d.Encoder` for details. 
    """
    def __init__(self, params):
        super(EncoderScSE, self).__init__(params)

        self.in_conv= ConvScSEBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        if(len(self.ft_chns) == 5):
            self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

class DecoderScSE(Decoder):
    """
    Decoder of 2D UNet with ScSE.

    The parameters for the backbone should be given in the `params` dictionary. 
    See :mod:`pymic.net.net2d.unet2d.Decoder` for details. 
    """
    def __init__(self, params):
        super(DecoderScSE, self).__init__(params)
        

        if(len(self.ft_chns) == 5):
            self.up1 = UpBlockScSE(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], self.dropout[3], self.up_mode) 
        self.up2 = UpBlockScSE(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], self.dropout[2], self.up_mode) 
        self.up3 = UpBlockScSE(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], self.dropout[1], self.up_mode) 
        self.up4 = UpBlockScSE(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], self.dropout[0], self.up_mode) 


class UNet2D_ScSE(UNet2D):
    """
    Combining 2D U-Net with SCSE module.

    * Reference: Abhijit Guha Roy, Nassir Navab, Christian Wachinger:
      Recalibrating Fully Convolutional Networks With Spatial and Channel 
      "Squeeze and Excitation" Blocks. 
      `IEEE Trans. Med. Imaging 38(2): 540-549 (2019). <https://ieeexplore.ieee.org/document/8447284>`_

    The parameters for the backbone should be given in the `params` dictionary. 
    See :mod:`pymic.net.net2d.unet2d.unet2d` for details. 
    """
    def __init__(self, params):
        super(UNet2D_ScSE, self).__init__(params)
        self.encoder  = Encoder(params)
        self.decoder  = Decoder(params)    
