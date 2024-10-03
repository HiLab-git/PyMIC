# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import numpy as np 
from pymic.net.net3d.unet3d import UNet3D
from pymic.net.net3d.unet3d_scse import UNet3D_ScSE
from pymic.net.net3d.unet2d5 import UNet2D5
from pymic.net.net3d.grunet import GRUNet
from pymic.net.net3d.lcovnet import LCOVNet
from pymic.net.net3d.trans3d.unetr_pp import UNETR_PP

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

def test_lcovnet():
    params = {'in_chns':4,
              'feature_chns':[16, 32, 64, 128, 256],
              'class_num': 2}
    Net = LCOVNet(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 96, 96, 96)
    xt = torch.from_numpy(x)
    xt = xt.clone().detach()
    
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

def test_mystunet():
    in_chns   = 4
    num_class = 4 
            # input_channels, num_classes, depth=[1,1,1,1,1,1], dims=[32, 64, 128, 256, 512, 512],
            #      pool_op_kernel_sizes=None, conv_kernel_sizes=None)
    dims=[16, 32, 64, 128, 256, 512]
    Net = MySTUNet(in_chns, num_class, dims = dims, pool_op_kernel_sizes = [[2, 2, 2], [2,2,2], [2,2,2], [2,2,2], [1, 1, 1]],
        conv_kernel_sizes = [[3, 3, 3], [3,3,3], [3,3,3], [3,3,3], [3,3,3], [3, 3, 3]])
    Net = Net.double()

    x  = np.random.rand(4, 4, 96, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    out = Net(xt)
    for y in out:
        y = y.detach().numpy()
        print(y.shape)

def test_grunet():
    params = {'in_chns':4,
              'feature_chns':[8, 16, 32, 64, 128],
              'dims': [2, 3, 3, 3, 3],
              'dropout':  [0, 0, 0.3, 0.4, 0.5],
              'class_num': 2,
              'depth': 2,
              'multiscale_pred': True}
    x  = np.random.rand(4, 4, 64, 128, 128)

    # params = {'in_chns':4,
    #           'feature_chns':[8, 16, 32, 64, 128],
    #           'dims': [3, 3, 3, 3, 3],
    #           'dropout':  [0, 0, 0.3, 0.4, 0.5],
    #           'class_num': 2,
    #           'depth': 4,
    #           'multiscale_pred': True}
    # x  = np.random.rand(4, 4, 96, 96, 96)

    Net = GRUNet(params)
    Net = Net.double()

    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    out = Net(xt)
    for y in out:
        y = y.detach().numpy()
        print(y.shape)

def test_unetr_pp():
    depths    = [128, 64, 32]
    for i in range(3):
        params = {'in_chns': 4,
                'class_num': 2,
                'img_size': [depths[i], 128, 128],
                'resolution_mode': i
                }
        net = UNETR_PP(params)
        net.double()

        x  = np.random.rand(2, 4, depths[i], 128, 128)
        xt = torch.from_numpy(x)
        xt = torch.tensor(xt)
        
        y = net(xt)
        print(len(y))
        for yi in y:
            yi = yi.detach().numpy()
            print(yi.shape)



if __name__ == "__main__":
    # test_unet3d()
    # test_unet3d_scse()
    test_lcovnet()
    # test_unetr_pp()
    # test_unet2d5()
    # test_mystunet()
    # test_fmunetv2()

    