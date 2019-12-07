# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
from pymic.layer.activation import get_acti_func
from pymic.layer.convolution import ConvolutionLayer
from pymic.layer.deconvolution import DeconvolutionLayer

class MyUNetBlock(nn.Module):
    def __init__(self,in_channels, out_channels, acti_func, acti_func_param):
        super(MyUNetBlock, self).__init__()
        
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func

        self.conv1 = ConvolutionLayer(in_channels,  out_channels, 3, 
                dim = 2, padding = 1, acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = ConvolutionLayer(out_channels, out_channels, 3, 
                dim = 2, padding = 1, acti_func=get_acti_func(acti_func, acti_func_param))

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        return f1 + f2

class MyUNet2D(nn.Module):
    def __init__(self, params):
        super(MyUNet2D, self).__init__()
        self.params = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.n_class   = self.params['class_num']
        self.acti_func = self.params['acti_func']
        self.dropout   = self.params['dropout']
        self.resolution_level = len(self.ft_chns)
        assert(self.resolution_level == 4)

        self.block1 = MyUNetBlock(self.in_chns, self.ft_chns[0], 
             self.acti_func, self.params)

        self.block2 = MyUNetBlock(self.ft_chns[0], self.ft_chns[1], 
             self.acti_func, self.params)

        self.block3 = MyUNetBlock(self.ft_chns[1], self.ft_chns[2], 
             self.acti_func, self.params)

        self.block4 = MyUNetBlock(self.ft_chns[2], self.ft_chns[3], 
             self.acti_func, self.params)

        self.block5 = MyUNetBlock(self.ft_chns[2] * 2, self.ft_chns[2], 
             self.acti_func, self.params)

        self.block6 = MyUNetBlock(self.ft_chns[1] * 2, self.ft_chns[1], 
             self.acti_func, self.params)

        self.block7 = MyUNetBlock(self.ft_chns[0] * 2, self.ft_chns[0], 
             self.acti_func, self.params)

        self.down1 = nn.MaxPool2d(kernel_size = 2)
        self.down2 = nn.MaxPool2d(kernel_size = 2)
        self.down3 = nn.MaxPool2d(kernel_size = 2)
        
        self.up1 = DeconvolutionLayer(self.ft_chns[3], self.ft_chns[2], kernel_size = 2,
            dim = 2, stride = 2, acti_func = get_acti_func(self.acti_func, self.params))
        self.up2 = DeconvolutionLayer(self.ft_chns[2], self.ft_chns[1], kernel_size = 2,
            dim = 2, stride = 2, acti_func = get_acti_func(self.acti_func, self.params))
        self.up3 = DeconvolutionLayer(self.ft_chns[1], self.ft_chns[0], kernel_size = 2,
            dim = 2, stride = 2, acti_func = get_acti_func(self.acti_func, self.params))

        if(self.dropout):
            self.drop3 = nn.Dropout(0.3)
            self.drop4 = nn.Dropout(0.5)
            
        self.conv = nn.Conv2d(self.ft_chns[0], self.n_class, 
            kernel_size = 3, padding = 1)

    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        f1 = self.block1(x)
        d1 = self.down1(f1)
        
        f2 = self.block2(d1)
        d2 = self.down2(f2)

        f3 = self.block3(d2)
        if(self.dropout > 0):
             f3 = self.drop3(f3)
        d3 = self.down3(f3)

        f4 = self.block4(d3)
        if(self.dropout > 0):
             f4 = self.drop4(f4)
        
        f4up  = self.up1(f4)
        f3cat = torch.cat((f3, f4up), dim = 1)
        f5    = self.block5(f3cat)

        f5up  = self.up2(f5)
        f2cat = torch.cat((f2, f5up), dim = 1)
        f6    = self.block6(f2cat)

        f6up  = self.up3(f6)
        f1cat = torch.cat((f1, f6up), dim = 1)
        f7    = self.block7(f1cat)

        output = self.conv(f7)
        if(len(x_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)
        return output

if __name__ == "__main__":
    params = {'in_chns':4,
              'feature_chns':[8, 32, 48, 64],
              'class_num': 2,
              'acti_func': 'relu', 
              'dropout': True}
    Net = MyUNet2D(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 10, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    print(len(y.size()))
    y = y.detach().numpy()
    print(y.shape)
