# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import numpy as np 
from pymic.net.net3d.unet3d import UNet3D
from pymic.net.net3d.unet3d_scse import UNet3D_ScSE
from pymic.net.net3d.unet2d5 import UNet2D5

def test_unet3d():
    params = {'in_chns':4,
              'class_num': 2,
              'feature_chns':[2, 8, 32, 64],
              'dropout' : [0, 0, 0, 0.5],
              'up_mode': 2,
              'multiscale_pred': False}
    Net = UNet3D(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 96, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    y = y.detach().numpy()
    print(y.shape)

    params = {'in_chns':4,
              'class_num': 2,
              'feature_chns':[2, 8, 32, 64, 128],
              'dropout' : [0, 0, 0, 0.4, 0.5],
              'up_mode': 3,
              'multiscale_pred': True}
    Net = UNet3D(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 96, 128, 128)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    out = Net(xt)
    for y in out: 
        y = y.detach().numpy()
        print(y.shape)

def test_unet3d_scse():
    params = {'in_chns':4,
              'feature_chns':[2, 8, 32, 48, 64],
              'dropout':  [0, 0, 0.3, 0.4, 0.5],
              'class_num': 2,
              'up_mode': 2}
    Net = UNet3D_ScSE(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 96, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    print(len(y.size()))
    y = y.detach().numpy()
    print(y.shape)

def test_unet2d5():
    params = {'in_chns':4,
              'feature_chns':[8, 16, 32, 64, 128],
              'conv_dims': [2, 2, 3, 3, 3],
              'dropout':  [0, 0, 0.3, 0.4, 0.5],
              'class_num': 2,
              'up_mode': 2,
              'multiscale_pred': True}
    Net = UNet2D5(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 32, 128, 128)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    out = Net(xt)
    for y in out:
        y = y.detach().numpy()
        print(y.shape)

    params = {'in_chns':4,
              'feature_chns':[8, 16, 32, 64, 128],
              'conv_dims': [2, 3, 3, 3, 3],
              'dropout':  [0, 0, 0.3, 0.4, 0.5],
              'class_num': 2,
              'up_mode': 0,
              'multiscale_pred': True}
    Net = UNet2D5(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 64, 128, 128)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    out = Net(xt)
    for y in out:
        y = y.detach().numpy()
        print(y.shape)

if __name__ == "__main__":
    # test_unet3d()
    # test_unet3d_scse()
    test_unet2d5()

    