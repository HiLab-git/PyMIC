# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np 
from pymic.net.cnn.basic_layer import get_conv_class
from pymic.net.cnn.unet import ConvBlock, DownBlock, UpBlock, Encoder, Decoder, UNet
from pymic.net.net2d.scse2d import *


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block.

    * Reference: Jie Hu, Li Shen, Gang Sun: Squeeze-and-Excitation Networks.
      `CVPR 2018. <https://ieeexplore.ieee.org/document/8578843>`_

    :param num_channels: Number of input channels
    :param reduction_ratio: By how much should the num_channels should be reduced.
    """
    def __init__(self, dim, num_channels, reduction_ratio=2):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """

        batch_size, num_channels = list(input_tensor.size())[:2]
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        if(self.dim == 2):
            output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        else:
            output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1, 1))

        return output_tensor

class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise.

    * Reference: Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in 
      Fully Convolutional Networks, MICCAI 2018.

    :param num_channels: Number of input channels.
    """
    def __init__(self, dim, num_channels):
        super(SpatialSELayer, self).__init__()
        self.dim  = dim
        conv_nd   = get_conv_class(dim) 
        self.conv = conv_nd(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        if(self.dim == 2):
            batch_size, channel, H, W = input_tensor.size()
        else:
            batch_size, channel, D, H, W = input_tensor.size()
        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            if(self.dim == 2):
                out = F.conv2d(input_tensor, weights)
            else:
                out = F.conv3d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        if(self.dim == 2):
            squeeze_tensor = squeeze_tensor.view(batch_size, 1, H, W)
        else:
            squeeze_tensor = squeeze_tensor.view(batch_size, 1, D, H, W)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor

class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation.

    * Reference: Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in 
      Fully Convolutional Networks, MICCAI 2018.
    
    :param num_channels: Number of input channels.
    :param reduction_ratio: By how much should the num_channels should be reduced.
    """
    def __init__(self, dim, num_channels, reduction_ratio=2):
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(dim, num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(dim, num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class ConvScSEBlock(ConvBlock):
    """
    Two convolutional blocks followed by `ChannelSpatialSELayer`.
    Each block consists of `Conv2d` + `BatchNorm2d` + `LeakyReLU`.
    A dropout layer is used between the wo blocks.

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, dim, in_channels, out_channels, dropout_p, kernel_size = 3, padding = 1,
        stride = 1, dilation = 1, norm_type = 'batch_norm'):
        super(ConvScSEBlock, self).__init__(dim, in_channels, out_channels, dropout_p, kernel_size, padding,
            stride, dilation, norm_type)

        self.scse = ChannelSpatialSELayer(dim, out_channels)
       
    def forward(self, x):
        x = self.conv_conv(x)
        x = self.scse(x)
        return x

class DownScSEBlock(DownBlock):
    """Downsampling followed by `ConvScSEBlock`.

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, dim, in_channels, out_channels, dropout_p):
        super(DownScSEBlock, self).__init__(dim, in_channels, out_channels, dropout_p)
        self.scse = ChannelSpatialSELayer(dim, out_channels)

    def forward(self, x):
        x = self.maxpool_conv(x)
        x = self.scse(x)
        return x

class UpScSEBlock(UpBlock):
    """Up-sampling followed by `ConvScSEBlock` in U-Net structure.
    
    The parameters for the backbone should be given in the `params` dictionary. 
    See :mod:`pymic.net.net2d.unet2d.UpBlock` for details. 
    """
    def __init__(self, dim, in_channels1, in_channels2, out_channels, dropout_p, up_mode = 2,
            kernel_size = 3, padding = 1, stride = 1, dilation = 1, norm_type = 'batch_norm'):
        super(UpScSEBlock, self).__init__(dim, in_channels1, in_channels2, out_channels, dropout_p, up_mode)
        self.conv = ConvScSEBlock(dim, in_channels2 * 2, out_channels, dropout_p, kernel_size, padding,
            stride, dilation, norm_type)

class EncoderScSE(Encoder):
    """
    Encoder of 2D UNet with ScSE.

    The parameters for the backbone should be given in the `params` dictionary. 
    See :mod:`pymic.net.net2d.unet2d.Encoder` for details. 
    """
    def __init__(self, params):
        super(EncoderScSE, self).__init__(params)
        
        self.in_conv= ConvScSEBlock(self.dim, self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1  = DownScSEBlock(self.dim, self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2  = DownScSEBlock(self.dim, self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3  = DownScSEBlock(self.dim, self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4  = DownScSEBlock(self.dim, self.ft_chns[3], self.ft_chns[4], self.dropout[4])

class DecoderScSE(Decoder):
    """
    Decoder of 2D UNet with ScSE.

    The parameters for the backbone should be given in the `params` dictionary. 
    See :mod:`pymic.net.net2d.unet2d.Decoder` for details. 
    """
    def __init__(self, params):
        super(DecoderScSE, self).__init__(params)
        self.up1 = UpScSEBlock(self.dim, self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], self.dropout[3], self.up_mode) 
        self.up2 = UpScSEBlock(self.dim, self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], self.dropout[2], self.up_mode) 
        self.up3 = UpScSEBlock(self.dim, self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], self.dropout[1], self.up_mode) 
        self.up4 = UpScSEBlock(self.dim, self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], self.dropout[0], self.up_mode) 


class UNet_ScSE(UNet):
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
        super(UNet_ScSE, self).__init__(params)
        self.encoder  = EncoderScSE(params)
        self.decoder  = DecoderScSE(params)    
