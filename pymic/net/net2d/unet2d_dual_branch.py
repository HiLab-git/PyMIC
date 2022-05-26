# -*- coding: utf-8 -*-
"""
Extention of U-Net with two decoders. The network was introduced in
the following paper:
    Xiangde Luo, Minhao Hu, Wenjun Liao, Shuwei Zhai, Tao Song, Guotai Wang,
    Shaoting Zhang. ScribblScribble-Supervised Medical Image Segmentation via 
    Dual-Branch Network and Dynamically Mixed Pseudo Labels Supervision.
    MICCAI 2022. 
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np 
from torch.nn.functional import interpolate
from pymic.net.net2d.unet2d import *

class DualBranchUNet2D(UNet2D):
    def __init__(self, params):
        params['deep_supervise'] = False
        super(DualBranchUNet2D, self).__init__(params)
        if(len(self.ft_chns) == 5):
            self.up1_aux = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], 0.0, self.bilinear) 
        self.up2_aux = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], 0.0, self.bilinear) 
        self.up3_aux = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], 0.0, self.bilinear) 
        self.up4_aux = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], 0.0, self.bilinear) 
    
        self.out_conv_aux = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size = 1)

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
          x_d3, x_d3_aux = self.up1(x4, x3), self.up1_aux(x4, x3)
        else:
          x_d3, x_d3_aux = x3, x3

        x_d2, x_d2_aux = self.up2(x_d3, x2), self.up2_aux(x_d3_aux, x2)
        x_d1, x_d1_aux = self.up3(x_d2, x1), self.up3_aux(x_d2_aux, x1)
        x_d0, x_d0_aux = self.up4(x_d1, x0), self.up4_aux(x_d1_aux, x0)
        output, output_aux = self.out_conv(x_d0), self.out_conv_aux(x_d0_aux)
        
        if(len(x_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)
            output_aux = torch.reshape(output_aux, new_shape)
            output_aux = torch.transpose(output_aux, 1, 2)
        return output, output_aux