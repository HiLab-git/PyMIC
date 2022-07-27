# -*- coding: utf-8 -*-
"""
An modification the U-Net to obtain multi-scale prediction according to 
the URPC paper (MICCAI 2021):
    Xiangde Luo, Wenjun Liao, Jienneg Chen, Tao Song, Yinan Chen, 
    Shichuan Zhang, Nianyong Chen, Guotai Wang, Shaoting Zhang:
    Efficient Semi-Supervised Gross Target Volume of Nasopharyngeal Carcinoma 
    Segmentation via Uncertainty Rectified Pyramid Consistency. 
    MICCAI 2021: 318-329
    https://link.springer.com/chapter/10.1007/978-3-030-87196-3_30 
Also see: https://github.com/HiLab-git/SSL4MIS/blob/master/code/networks/unet.py
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np 
from torch.distributions.uniform import Uniform
from pymic.net.net2d.unet2d import ConvBlock, DownBlock, UpBlock

def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x

class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x

class UNet2D_URPC(nn.Module):
    def __init__(self, params):
        super(UNet2D_URPC, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.n_class   = self.params['class_num']
        self.bilinear  = self.params['bilinear']
        self.dropout   = self.params['dropout']
        assert(len(self.ft_chns) == 5)

        self.in_conv= ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])
        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], 0.0, self.bilinear) 
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], 0.0, self.bilinear) 
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], 0.0, self.bilinear) 
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], 0.0, self.bilinear) 
    
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,  
                                  kernel_size = 3, padding = 1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

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
        x4 = self.down4(x3)
        
        x = self.up1(x4, x3)
        if self.training:
            x = nn.functional.dropout(x, p=0.5)
        dp3_out = self.out_conv_dp3(x)

        x = self.up2(x, x2)
        if self.training:
            x = FeatureDropout(x)
        dp2_out = self.out_conv_dp2(x)

        x = self.up3(x, x1)
        if self.training:
            x = self.feature_noise(x)
        dp1_out = self.out_conv_dp1(x)

        x = self.up4(x, x0)
        dp0_out = self.out_conv(x)

        out_shape = list(dp0_out.shape)[2:]
        dp3_out = nn.functional.interpolate(dp3_out, out_shape)
        dp2_out = nn.functional.interpolate(dp2_out, out_shape)
        dp1_out = nn.functional.interpolate(dp1_out, out_shape)
        out = [dp0_out, dp1_out, dp2_out, dp3_out]

        if(len(x_shape) == 5):
            new_shape = [N, D] + list(dp0_out.shape)[1:]
            for i in range(len(out)):
                out[i] = torch.transpose(torch.reshape(out[i], new_shape), 1, 2)
        return out