# -*- coding: utf-8 -*-
from __future__ import print_function, division

import itertools
import logging
import torch
import torch.nn as nn
from pymic.net.net_init import Initialization_He, Initialization_XavierUniform

'''
A copy of fmunetv3, and rename the class as FMUNet.
'''
dim0 = {0:3, 1:2, 2:2}
dim1 = {0:3, 1:3, 2:2}
conv_knl = {2: (1, 3, 3), 3: 3}
conv_pad = {2: (0, 1, 1), 3: 1}
pool_knl    = {2: (1, 2, 2), 3: 2}
down_stride = {2: (1, 2, 2), 3: 2}

class ResConv(nn.Module):
    def __init__(self, out_channels, dim = 3, dropout_p = 0.0, depth = 2):
        super(ResConv, self).__init__()
        assert(dim == 2 or dim == 3)
        self.out_channels = out_channels
        self.conv_list = nn.ModuleList([nn.Sequential(
            nn.InstanceNorm3d(out_channels, affine = True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size=conv_knl[dim], padding=conv_pad[dim]))
            for i in range(depth)])

    def forward(self, x):
        for conv in self.conv_list:
            x = conv(x)  + x
        return x

class DownSample(nn.Module):
    """downsampling based on convolution

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dim: (int) Should be 2 or 3, for 2D and 3D convolution, respectively.
    :param dropout_p: (int) Dropout probability.
    :param downsample: (bool) Use downsample or not after convolution. 
    """
    def __init__(self, in_channels, out_channels, dim = 3):
        super(DownSample, self).__init__()        
        self.down   = nn.Sequential(
            nn.InstanceNorm3d(in_channels, affine = True),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=conv_knl[dim], 
                padding=conv_pad[dim], stride = down_stride[dim])
        )

    def forward(self, x):
        return self.down(x)

class UpCatConv(nn.Module):
    """Upsampling followed by `ResConv` block
    
    :param in_channels1: (int) Input channel number for low-resolution feature map.
    :param in_channels2: (int) Input channel number for high-resolution feature map.
    :param out_channels: (int) Output channel number.
    :param dim: (int) Should be 2 or 3, for 2D and 3D convolution, respectively.
    :param dropout_p: (int) Dropout probability.
    :param up_mode: (string or int) The mode for upsampling. The allowed values are:
        0 (or `TransConv`), 1 (`Nearest`), 2 (`Trilinear` for 3D and `Bilinear` for 2D). 
        The default value is 2.
    """
    def __init__(self, in_channels1, in_channels2, out_channels, dim = 3):
        super(UpCatConv, self).__init__()
 
        self.up = nn.Sequential(
                nn.InstanceNorm3d(in_channels1, affine = True),
                nn.LeakyReLU(),
                nn.Conv3d(in_channels1, in_channels2, kernel_size=1, padding=0),
                nn.Upsample(scale_factor=pool_knl[dim], mode='trilinear', align_corners=True)
            )

        self.conv = nn.Sequential(
            nn.InstanceNorm3d(in_channels2*2, affine = True),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels2 * 2, out_channels, kernel_size=conv_knl[dim], padding=conv_pad[dim])
            )     

    def forward(self, x_l, x_h):
        """
        x_l: low-resolution feature map.
        x_h: high-resolution feature map.
        """
        y = torch.cat([x_h, self.up(x_l)], dim=1)
        return self.conv(y)

class Encoder(nn.Module):
    """
    A modification of the encoder of 3D UNet by using ConvScSEBlock3D

    Parameters are given in the `params` dictionary.
    See :mod:`pymic.net.net3d.unet3d.Encoder` for details. 

    res_mode: resolution mode: 0-- isotrpic, 1-- near isotrpic, 2-- isotropic
    """
    def __init__(self, ft_chns, res_mode = 0, dropout_p = 0, depth = 2):
        super(Encoder, self).__init__()
        d0, d1 = dim0[res_mode], dim1[res_mode]

        self.en_conv0 = ResConv(ft_chns[0], d0, 0, depth)
        self.en_conv1 = ResConv(ft_chns[1], d1, 0, depth)
        self.en_conv2 = ResConv(ft_chns[2], 3, dropout_p, depth)
        self.en_conv3 = ResConv(ft_chns[3], 3, dropout_p, depth)
        self.en_conv4 = ResConv(ft_chns[4], 3, dropout_p, depth)

        self.down0 = DownSample(ft_chns[0], ft_chns[1], d0)
        self.down1 = DownSample(ft_chns[1], ft_chns[2], d1)
        self.down2 = DownSample(ft_chns[2], ft_chns[3], 3)
        self.down3 = DownSample(ft_chns[3], ft_chns[4], 3)
    
    def forward(self, x):
        x0 = self.en_conv0(x)
        x1 = self.en_conv1(self.down0(x0))
        x2 = self.en_conv2(self.down1(x1))
        x3 = self.en_conv3(self.down2(x2))
        x4 = self.en_conv4(self.down3(x3))
        return [x0, x1, x2, x3, x4]
        
class Decoder(nn.Module):
    """
    A modification of the encoder of 3D UNet by using ConvScSEBlock3D

    Parameters are given in the `params` dictionary.
    See :mod:`pymic.net.net3d.unet3d.Encoder` for details. 
    """
    def __init__(self, ft_chns, res_mode = 0, dropout_p = 0, depth = 2):
        super(Decoder, self).__init__()
        d0, d1 = dim0[res_mode], dim1[res_mode]
        
        self.upcat0 = UpCatConv(ft_chns[1], ft_chns[0], ft_chns[0], d0) 
        self.upcat1 = UpCatConv(ft_chns[2], ft_chns[1], ft_chns[1], d1) 
        self.upcat2 = UpCatConv(ft_chns[3], ft_chns[2], ft_chns[2], 3) 
        self.upcat3 = UpCatConv(ft_chns[4], ft_chns[3], ft_chns[3], 3) 

        self.de_conv0 = ResConv(ft_chns[0], d0, 0, depth)
        self.de_conv1 = ResConv(ft_chns[1], d1, 0, depth)
        self.de_conv2 = ResConv(ft_chns[2], 3, dropout_p, depth)
        self.de_conv3 = ResConv(ft_chns[3], 3, dropout_p, depth)
        self.de_conv4 = ResConv(ft_chns[4], 3, dropout_p, depth)

    def forward(self, x):
        x0, x1, x2, x3, x4 = x
        x4_de = self.de_conv4(x4)
        x3_de = self.de_conv3(self.upcat3(x4_de, x3))
        x2_de = self.de_conv2(self.upcat2(x3_de, x2))
        x1_de = self.de_conv1(self.upcat1(x2_de, x1))
        x0_de = self.de_conv0(self.upcat0(x1_de, x0))
        return [x0_de, x1_de, x2_de, x3_de]

class FMUNet(nn.Module):
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
        super(FMUNet, self).__init__()
        params = self.get_default_parameters(params)

        self.stage  = 'train'
        in_chns     = params['in_chns'] 
        ft_chns     = params['feature_chns']
        res_mode    = params['res_mode']
        dropout     = params['dropout']
        depth       = params['depth']
        cls_num     = params['class_num']
        self.mul_pred = params.get('multiscale_pred', True)
        self.tune_mode= params.get('finetune_mode', 'all')
        self.load_mode= params.get('weights_load_mode', 'all')

        d0 = dim0[res_mode]
        self.project = nn.Conv3d(in_chns, ft_chns[0], kernel_size=conv_knl[d0], padding=conv_pad[d0])
        self.encoder = Encoder(ft_chns, res_mode, dropout, depth)
        # self.decoder = Decoder(ft_chns, res_mode, dropout, depth = 2)
        self.decoder = Decoder(ft_chns, res_mode, dropout, depth)
        
        self.out_layers = nn.ModuleList()
        dims = [dim0[res_mode], dim1[res_mode], 3, 3]
        for i in range(4):
            out_layer = nn.Sequential(
                nn.InstanceNorm3d(ft_chns[i], affine = True),
                nn.LeakyReLU(),
                nn.Conv3d(ft_chns[i], cls_num, kernel_size=conv_knl[dims[i]], padding=conv_pad[dims[i]]))
            self.out_layers.append(out_layer)

        init = params['initialization'].lower()
        weightInitializer =  Initialization_He(1e-2) if init == 'he' else Initialization_XavierUniform()
        self.apply(weightInitializer)
      
    def get_default_parameters(self, params):
        default_param = {
            'finetune_mode': 'all',
            'initialization': 'he',
            'feature_chns':  [32, 64, 128, 256, 512],
            'dropout':  0.2,
            'res_mode': 0,
            'depth': 2,
            'multiscale_pred': True
        }
        for key in default_param:
            params[key] = params.get(key, default_param[key])
        for key in params:
                logging.info("{0:}  = {1:}".format(key, params[key]))
        return params

    def set_stage(self, stage):
        self.stage = stage

    def forward(self, x):
        x_en = self.encoder(self.project(x))
        x_de = self.decoder(x_en)
        output = self.out_layers[0](x_de[0])
        if(self.mul_pred and self.stage == 'train'):
            output = [output]
            for i in range(1, len(x_de)):
                output.append(self.out_layers[i](x_de[i]))
        return output
    
    def get_parameters_to_update(self):
        if(self.tune_mode == 'all'): 
            return self.parameters()

        up_params = itertools.chain()
        if(self.tune_mode == 'decoder'):
            up_blocks = [self.decoder, self.out_layers]
        else:
            raise ValueError("undefined fine-tune mode for FMUNet: {0:}".format(self.tune_mode))
        for block in up_blocks:
            up_params = itertools.chain(up_params, block.parameters())
        return up_params
    
    def get_parameters_to_load(self):
        state_dict = self.state_dict()
        if(self.load_mode == 'encoder'): 
            state_dict = {k:v for k, v in state_dict.items() if "project" in k or "encoder" in k }
        return state_dict