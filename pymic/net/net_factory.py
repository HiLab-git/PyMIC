# -*- coding: utf-8 -*-
from __future__ import print_function, division
from pymic.net.net2d.unet2d import UNet2D
from pymic.net.net2d.cople_net import COPLENet
from pymic.net.net2d.unet2d_scse import UNet2D_ScSE
from pymic.net.net3d.unet2d5 import UNet2D5
from pymic.net.net3d.unet3d import UNet3D

net_dict = {
	'UNet2D': UNet2D,
	'COPLENet': COPLENet,
	'UNet2D_ScSE': UNet2D_ScSE,
	'UNet2D5': UNet2D5,
	'UNet3D': UNet3D
	}
	
def get_network(params):
	net_type = params['net_type']
	if(net_type in net_dict):
		net_obj = net_dict[net_type](params)
	else:
		raise ValueError("Undefined network type {0:}".format(net_type))
	return net_obj
