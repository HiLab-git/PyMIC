# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
from pymic.layer.activation import get_acti_func
from pymic.layer.convolution import ConvolutionLayer
from pymic.layer.deconvolution import DeconvolutionLayer

class UNetBlock(nn.Module):
    def __init__(self,in_channels, out_channels, acti_func, acti_func_param):
        super(UNetBlock, self).__init__()
        
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func

        self.conv1 = ConvolutionLayer(in_channels,  out_channels, 3, 
                dim = 2, padding = 1, acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = ConvolutionLayer(out_channels, out_channels, 3, 
                dim = 2, padding = 1, acti_func=get_acti_func(acti_func, acti_func_param))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNet2D(nn.Module):
    def __init__(self, params):
        super(UNet2D, self).__init__()
        self.params = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.n_class   = self.params['class_num']
        self.acti_func = self.params['acti_func']
        assert(len(self.ft_chns) == 5)

        self.block1 = UNetBlock(self.in_chns, self.ft_chns[0], 
             self.acti_func, self.params)

        self.block2 = UNetBlock(self.ft_chns[0], self.ft_chns[1], 
             self.acti_func, self.params)

        self.block3 = UNetBlock(self.ft_chns[1], self.ft_chns[2], 
             self.acti_func, self.params)

        self.block4 = UNetBlock(self.ft_chns[2], self.ft_chns[3], 
             self.acti_func, self.params)

        self.block5 = UNetBlock(self.ft_chns[3], self.ft_chns[4], 
             self.acti_func, self.params)

        self.block6 = UNetBlock(self.ft_chns[3] * 2, self.ft_chns[3], 
             self.acti_func, self.params)

        self.block7 = UNetBlock(self.ft_chns[2] * 2, self.ft_chns[2], 
             self.acti_func, self.params)

        self.block8 = UNetBlock(self.ft_chns[1] * 2, self.ft_chns[1], 
             self.acti_func, self.params)

        self.block9 = UNetBlock(self.ft_chns[0] * 2, self.ft_chns[0], 
             self.acti_func, self.params)

        self.down1 = nn.MaxPool2d(kernel_size = 2)
        self.down2 = nn.MaxPool2d(kernel_size = 2)
        self.down3 = nn.MaxPool2d(kernel_size = 2)
        self.down4 = nn.MaxPool2d(kernel_size = 2)

        self.up1 = DeconvolutionLayer(self.ft_chns[4], self.ft_chns[3], kernel_size = 2,
            dim = 2, stride = 2, acti_func = get_acti_func(self.acti_func, self.params))
        self.up2 = DeconvolutionLayer(self.ft_chns[3], self.ft_chns[2], kernel_size = 2,
            dim = 2, stride = 2, acti_func = get_acti_func(self.acti_func, self.params))
        self.up3 = DeconvolutionLayer(self.ft_chns[2], self.ft_chns[1], kernel_size = 2,
            dim = 2, stride = 2, acti_func = get_acti_func(self.acti_func, self.params))
        self.up4 = DeconvolutionLayer(self.ft_chns[1], self.ft_chns[0], kernel_size = 2,
            dim = 2, stride = 2, acti_func = get_acti_func(self.acti_func, self.params))

        self.conv = nn.Conv2d(self.ft_chns[0], self.n_class, 
            kernel_size = 3, padding = 1)

    def forward(self, x):
        f1 = self.block1(x);  d1 = self.down1(f1)
        f2 = self.block2(d1); d2 = self.down2(f2)
        f3 = self.block3(d2); d3 = self.down3(f3)
        f4 = self.block4(d3); d4 = self.down4(f4)
        f5 = self.block5(d4)

        f5up  = self.up1(f5)
        f4cat = torch.cat((f4, f5up), dim = 1)
        f6    = self.block6(f4cat)

        f6up  = self.up2(f6)
        f3cat = torch.cat((f3, f6up), dim = 1)
        f7    = self.block7(f3cat)

        f7up  = self.up3(f7)
        f2cat = torch.cat((f2, f7up), dim = 1)
        f8    = self.block8(f2cat)

        f8up  = self.up4(f8)
        f1cat = torch.cat((f1, f8up), dim = 1)
        f9    = self.block9(f1cat)

        output = self.conv(f9)
        return output

if __name__ == "__main__":
    params = {'in_chns':4,
              'feature_chns':[2, 8, 32, 48, 64],
              'class_num': 2,
              'acti_func': 'relu'}
    Net = UNet2D(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    print(len(y.size()))
    y = y.detach().numpy()
    print(y.shape)
