# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
from pymic.layer.activation import get_acti_func
from pymic.layer.convolution import ConvolutionLayer
from pymic.layer.deconvolution import DeconvolutionLayer

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_chnannels,
            dim, resample, acti_func, acti_func_param):
        super(UNetBlock, self).__init__()
        
        self.in_chns   = in_channels
        self.out_chns  = out_chnannels
        self.dim       = dim
        self.resample  = resample  # resample should be 'down', 'up', or None
        self.acti_func = acti_func

        self.conv1 = ConvolutionLayer(in_channels,  out_chnannels, kernel_size = 3, padding=1,
                dim = self.dim, acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = ConvolutionLayer(out_chnannels, out_chnannels, kernel_size = 3, padding=1,
                dim = self.dim, acti_func=get_acti_func(acti_func, acti_func_param))
        if(self.resample == 'down'):
            if(self.dim == 2):
                self.resample_layer = nn.MaxPool2d(kernel_size = 2, stride = 2)
            else:
                self.resample_layer = nn.MaxPool3d(kernel_size = 2, stride = 2)
        elif(self.resample == 'up'):
            self.resample_layer = DeconvolutionLayer(out_chnannels, out_chnannels, kernel_size = 2,
                    dim = self.dim, stride = 2, acti_func = get_acti_func(acti_func, acti_func_param))
        else:
            assert(self.resample == None)

    def forward(self, x):
        x_shape = list(x.shape)
        if(self.dim == 2 and len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        output = self.conv1(x)
        output = self.conv2(output)
        resample = None
        if(self.resample is not None):
            resample =  self.resample_layer(output)

        if(self.dim == 2 and len(x_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)
            if(resample is not None):
                resample_shape = list(resample.shape)
                new_shape = [N, D] + resample_shape[1:]
                resample = torch.reshape(resample, new_shape)
                resample = torch.transpose(resample, 1, 2)
        return output, resample

class UNet2D5(nn.Module):
    def __init__(self, params):
        super(UNet2D5, self).__init__()
        self.params = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.n_class   = self.params['class_num']
        self.acti_func = self.params['acti_func']
        self.dropout   = self.params['dropout']
        assert(len(self.ft_chns) == 5)

        self.block1 = UNetBlock(self.in_chns, self.ft_chns[0], 
            2, 'down', self.acti_func, self.params)

        self.block2 = UNetBlock(self.ft_chns[0], self.ft_chns[1], 
            2, 'down', self.acti_func, self.params)

        self.block3 = UNetBlock(self.ft_chns[1], self.ft_chns[2], 
            2, 'down', self.acti_func, self.params)

        self.block4 = UNetBlock(self.ft_chns[2], self.ft_chns[3], 
            3, 'down', self.acti_func, self.params)

        self.block5 = UNetBlock(self.ft_chns[3], self.ft_chns[4], 
            3, 'up', self.acti_func, self.params)

        self.block6 = UNetBlock(self.ft_chns[3] + self.ft_chns[4], self.ft_chns[3], 
            2, 'up', self.acti_func, self.params)

        self.block7 = UNetBlock(self.ft_chns[2] + self.ft_chns[3], self.ft_chns[2], 
            2, 'up', self.acti_func, self.params)

        self.block8 = UNetBlock(self.ft_chns[1] + self.ft_chns[2], self.ft_chns[1], 
            2, 'up', self.acti_func, self.params)

        self.block9 = UNetBlock(self.ft_chns[0] + self.ft_chns[1], self.ft_chns[0], 
            2, None, self.acti_func, self.params)

        self.conv = nn.Conv3d(self.ft_chns[0], self.n_class, 
            kernel_size = (1, 3, 3), padding = (0, 1, 1))

        if(self.dropout):
            self.drop1 = nn.Dropout(p=0.1)
            self.drop2 = nn.Dropout(p=0.1)
            self.drop3 = nn.Dropout(p=0.2)
            self.drop4 = nn.Dropout(p=0.2)
            self.drop5 = nn.Dropout(p=0.2)
            
    def forward(self, x):
        f1, d1 = self.block1(x)
        if(self.dropout):
            f1 = self.drop1(f1)
            d1 = self.drop1(d1)
        f2, d2 = self.block2(d1)
        if(self.dropout):
            f2 = self.drop2(f2)
            d2 = self.drop2(d2)
        f3, d3 = self.block3(d2)
        if(self.dropout):
            f3 = self.drop3(f3)
            d3 = self.drop3(d3)
        f4, d4 = self.block4(d3)
        if(self.dropout):
            f4 = self.drop4(f4)
            d4 = self.drop4(d4)
        f5, f5up = self.block5(d4)
        if(self.dropout):
            f5 = self.drop5(f5)
            f5up = self.drop5(f5up)
        
        f4cat = torch.cat((f4, f5up), dim = 1)
        f6, f6up = self.block6(f4cat)

        f3cat = torch.cat((f3, f6up), dim = 1)
        f7, f7up = self.block7(f3cat)

        f2cat = torch.cat((f2, f7up), dim = 1)
        f8, f8up = self.block8(f2cat)

        f1cat = torch.cat((f1, f8up), dim = 1)
        f9, _ = self.block9(f1cat)

        output = self.conv(f9)
        return output

if __name__ == "__main__":
    params = {'in_chns':4,
              'feature_chns':[2, 8, 32, 48, 64],
              'class_num': 2,
              'dropout': True,
              'acti_func': 'leakyrelu',
              'leakyrelu_alpha': 0.01}
    Net = UNet2D5(params)
    Net = Net.double()

    x  = np.random.rand(1, 4, 16, 128, 128)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    y = y.detach().numpy()
    print(y.shape)
