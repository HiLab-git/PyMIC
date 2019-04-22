# -*- coding: utf-8 -*-
from __future__ import print_function, division

from pymic.net2d.unet2d import UNet2D
from pymic.net3d.unet2d5 import UNet2D5
from pymic.net3d.unet3d import UNet3D

def get_network(params):
    net_type = params['net_type']
    if(net_type == 'UNet2D'):
        return UNet2D(params)
    if(net_type == 'UNet2D5'):
        return UNet2D5(params)
    elif(net_type == 'UNet3D'):
        return UNet3D(params)
    else:
        raise ValueError("undefined network {0:}".format(net_type))