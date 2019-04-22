# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
from pyMIC.layer.convolution import ConvolutionLayer
from pyMIC.layer.deconvolution import DeconvolutionLayer

class UNetBlock(nn.Module):
    def __init__(self,in_channels, out_channels, acti_func):
        super(UNetBlock, self).__init__()
        
        self.in_chns  = in_channels
        self.out_chns = out_channels
        self.acti_func = acti_func

        self.conv1 = ConvolutionLayer(in_channels,  out_channels, 3, 
                padding = 1, acti_func=acti_func, keep_prob=1.0)
        self.conv2 = ConvolutionLayer(out_channels, out_channels, 3, 
                padding = 1, acti_func=acti_func, keep_prob=1.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DemoNet(nn.Module):
    def __init__(self, params):
        super(DemoNet, self).__init__()
        self.params = params
        self.in_chns = self.params['input_chn_num']
        self.ft_chns = self.params['feature_chn_nums']
        self.n_class = self.params['class_num']
        assert(len(self.ft_chns) == 2)

        # self.block1 = UNetBlock(self.in_chns, self.ft_chns[0], 
        #      acti_func=self.get_acti_func())

        # self.block2 = UNetBlock(self.ft_chns[0], self.ft_chns[1], 
        #      acti_func=self.get_acti_func())

        # self.block3 = UNetBlock(self.ft_chns[0] * 2, self.ft_chns[0], 
        #      acti_func=self.get_acti_func())

        # self.down1 = nn.MaxPool3d(kernel_size = 2)


        # self.up1 = DeconvolutionLayer(self.ft_chns[1], self.ft_chns[0], kernel_size = 2,
        #     stride = 2, acti_func = self.get_acti_func())
        self.conv1 = nn.Conv3d(self.in_chns, self.ft_chns[0], 
            kernel_size = (1, 3, 3), padding = (0, 1, 1))
        self.conv2 = nn.Conv3d(self.ft_chns[0], self.ft_chns[0], 
            kernel_size = (1, 3, 3), padding = (0, 1, 1))
        self.conv = nn.Conv3d(self.ft_chns[0], self.n_class, 
            kernel_size = (1, 3, 3), padding = (0, 1, 1))

    def get_acti_func(self):
        return nn.ReLU()
        # return  nn.LeakyReLU(0.1)

    def forward(self, x):
        f1 = self.conv1(x)
        f1 = self.conv2(f1)
        # ;  d1 = self.down1(f1)
        # f2 = self.block2(d1)

        # f2up  = self.up1(f2)
        # f1cat = torch.cat((f1, f2up), dim = 1)
        # f3    = self.block3(f1cat)

        output = self.conv(f1)
        return output

if __name__ == "__main__":
    params = {'input_chn_num':4,
              'feature_chn_nums':[2, 8],
              'class_num': 2}
    Net = DemoNet(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 96, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    y = y.detach().numpy()
    print(y.shape)
