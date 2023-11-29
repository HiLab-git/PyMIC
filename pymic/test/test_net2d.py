# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import numpy as np 
from pymic.net.net2d.unet2d import UNet2D
from pymic.net.net2d.unet2d_scse import UNet2D_ScSE

def test_unet2d():
    params = {'in_chns':4,
              'feature_chns':[16, 32, 64, 128, 256],
              'dropout':  [0, 0, 0.3, 0.4, 0.5],
              'class_num': 2,
              'up_mode': 0,
              'multiscale_pred': True}
    Net = UNet2D(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 10, 256, 256)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    out = Net(xt)
    if params['multiscale_pred']:
        for y in out:
            print(len(y.size()))
            y = y.detach().numpy()
            print(y.shape)
    else:
        print(out.shape)

def test_unet2d_scse():
    params = {'in_chns':4,
              'feature_chns':[16, 32, 64, 128, 256],
              'dropout':  [0, 0, 0.3, 0.4, 0.5],
              'class_num': 2,
              'up_mode': 0,
              'multiscale_pred': True}
    Net = UNet2D_ScSE(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 10, 256, 256)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    out = Net(xt)
    if params['multiscale_pred']:
        for y in out:
            print(len(y.size()))
            y = y.detach().numpy()
            print(y.shape)
    else:
        print(out.shape)

if __name__ == "__main__":
    # test_unet2d()
    test_unet2d_scse()