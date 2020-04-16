# -*- coding: utf-8 -*-
from __future__ import print_function, division
from pymic.net2d.unet2d import UNet2D
from pymic.net2d.unet2d_scse import UNet2D_ScSE
from pymic.net3d.unet2d5 import UNet2D5
from pymic.net3d.unet3d import UNet3D

net_dict = {
	'UNet2D': UNet2D,
	'UNet2D_ScSE': UNet2D_ScSE,
	'UNet2D5': UNet2D5,
	'UNet3D': UNet3D
	}
def get_network(params):
    net_type = params['net_type']
    net =  net_dict[net_type](params)
    return net

