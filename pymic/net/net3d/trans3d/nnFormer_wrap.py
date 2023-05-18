# -*- coding: utf-8 -*-
from __future__ import print_function, division

import math
import torch
import torch.nn as nn
import numpy as np
from nnformer.network_architecture.nnFormer_tumor import nnFormer

class nnFormer_wrap(nn.Module):
    def __init__(self, params):
        super(nnFormer_wrap, self).__init__()
        patch_size = params["patch_size"] # 96x96x96
        n_class    = params['class_num']
        in_chns    = params['in_chns']
        # https://github.com/282857341/nnFormer/blob/main/nnformer/network_architecture/nnFormer_tumor.py
        self.nnformer = nnFormer(crop_size = patch_size,
                embedding_dim=192,
                input_channels = in_chns, 
                num_classes = n_class, 
                conv_op=nn.Conv3d, 
                depths =[2,2,2,2],
                num_heads = [6, 12, 24, 48],
                patch_size = [4,4,4],
                window_size= [4,4,8,4],
                deep_supervision=False)

    def forward(self, x):
        return self.nnformer(x)

if __name__ == "__main__":
    params = {"patch_size": [96, 96, 96],
              "in_chns": 1, 
              "class_num": 5}
    Net = nnFormer_wrap(params)
    Net = Net.double()

    x  = np.random.rand(1, 1, 96, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    print(y.shape)
