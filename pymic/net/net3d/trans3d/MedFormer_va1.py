# -*- coding: utf-8 -*-
from __future__ import print_function, division

import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from pymic.net.net3d.unet3d import Decoder

class EmbeddingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(EmbeddingBlock, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv3d(in_channels, out_channels//2, kernel_size=kernel_size, padding=padding, stride = stride)
        self.conv2 = nn.Conv3d(out_channels//2, out_channels, kernel_size=1)
        self.act  = nn.GELU()
        self.norm1 = nn.LayerNorm(out_channels//2)
        self.norm2 = nn.LayerNorm(out_channels)
        
       
    def forward(self, x):
        x = self.act(self.conv1(x))
        # norm 1
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_channels // 2, Ws, Wh, Ww)

        x = self.act(self.conv2(x))
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm2(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_channels, Ws, Wh, Ww)

        return x

class Encoder(nn.Module):
    """
    Encoder of 3D UNet.

    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param in_chns: (int) Input channel number.
    :param feature_chns: (list) Feature channel for each resolution level. 
      The length should be 4 or 5, such as [16, 32, 64, 128, 256].
    :param dropout: (list) The dropout ratio for each resolution level. 
      The length should be the same as that of `feature_chns`.
    """
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        assert(len(self.ft_chns) == 4)

        self.down0  = EmbeddingBlock(self.in_chns, self.ft_chns[0], 3, 1, 1)
        self.down1  = EmbeddingBlock(self.in_chns, self.ft_chns[1], 2, 0, 2)
        self.down2  = EmbeddingBlock(self.in_chns, self.ft_chns[2], 4, 0, 4)
        self.down3  = EmbeddingBlock(self.in_chns, self.ft_chns[3], 8, 0, 8)
        
    def forward(self, x):
        x0 = self.down0(x)
        x1 = self.down1(x)
        x2 = self.down2(x)
        x3 = self.down3(x)
        output = [x0, x1, x2, x3]
        return output

class MedFormerVA1(nn.Module):
    def __init__(self, params):
        super(MedFormerVA1, self).__init__()
        self.params   = params
        self.encoder  = Encoder(params)
        self.decoder  = Decoder(params)  

    def forward(self, x):
        f = self.encoder(x)
        output = self.decoder(f)
        return output


if __name__ == "__main__":
    params = {'in_chns':1,
              'class_num': 8,
              'feature_chns':[16, 32, 64, 128],
              'dropout' : [0, 0, 0, 0.5],
              'trilinear': True,
              'deep_supervise': True,
              'attention_hidden_size': 128,
              'attention_num_heads': 4,
              'attention_mlp_dim': 256,
              'attention_dropout_rate': 0.2}
    Net = MedFormerVA1(params)
    Net = Net.double()

    x  = np.random.rand(1, 1, 96, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    print("output length", len(y))
    for yi in y:
        yi = yi.detach().numpy()
        print(yi.shape)