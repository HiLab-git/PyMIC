# -*- coding: utf-8 -*-
from __future__ import print_function, division
from pymic.net.net2d.unet2d import UNet2D
from pymic.net.net2d.cople_net import COPLENet
from pymic.net.net2d.unet2d_attention import AttentionUNet2D
from pymic.net.net2d.unet2d_nest import NestedUNet2D
from pymic.net.net2d.unet2d_scse import UNet2D_ScSE
from pymic.net.net3d.unet2d5 import UNet2D5
from pymic.net.net3d.unet3d import UNet3D
from pymic.net.net3d.unet3d_scse import UNet3D_ScSE

SegNetDict = {
	'UNet2D': UNet2D,
	'COPLENet': COPLENet,
	'AttentionUNet2D': AttentionUNet2D,
	'NestedUNet2D': NestedUNet2D,
	'UNet2D_ScSE': UNet2D_ScSE,
	'UNet2D5': UNet2D5,
	'UNet3D': UNet3D,
	'UNet3D_ScSE': UNet3D_ScSE
	}
