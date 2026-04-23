# -*- coding: utf-8 -*-
"""
Built-in networks for segmentation.

* UNet2D :mod:`pymic.net.net2d.unet2d.UNet2D`
* UNet2D_DualBranch :mod:`pymic.net.net2d.unet2d_dual_branch.UNet2D_DualBranch`
* UNet2D_CCT  :mod:`pymic.net.net2d.unet2d_cct.UNet2D_CCT`
* UNet2D_ScSE  :mod:`pymic.net.net2d.unet2d_scse.UNet2D_ScSE`
* AttentionUNet2D  :mod:`pymic.net.net2d.unet2d_attention.AttentionUNet2D`
* MCNet2D      :mod:`pymic.net.net2d.unet2d_mcnet.MCNet2D`
* NestedUNet2D :mod:`pymic.net.net2d.unet2d_nest.NestedUNet2D`
* COPLENet  :mod:`pymic.net.net2d.cople_net.COPLENet`
* UNet2D5 :mod:`pymic.net.net3d.unet2d5.UNet2D5`
* UNet3D :mod:`pymic.net.net3d.unet3d.UNet3D`
* UNet3D_ScSE :mod:`pymic.net.net3d.unet3d_scse.UNet3D_ScSE`
"""
from __future__ import print_function, division
# from pymic.net.net2d.unet2d import UNet2D
# from pymic.net.net2d.unet2d_multi_decoder import UNet2D_DualBranch, MCNet2D
# from pymic.net.net2d.unet2d_mtnet import MTNet2D
from pymic.net.cnn.unet import UNet
from pymic.net.cnn.unet2d_canet import CANet
from pymic.net.cnn.unet_scse import UNet_ScSE
from pymic.net.cnn.coplenet2d import COPLENet
from pymic.net.cnn.unet_attention import AttentionUNet
from pymic.net.cnn.unet_pp import UNetpp
from pymic.net.cnn.unet2d5 import UNet2D5

# from pymic.net.net2d.unet2d_scse import UNet2D_ScSE
from pymic.net.transformer.transunet import TransUNet
from pymic.net.transformer.swinunet import SwinUNet
# from pymic.net.net2d.umamba import UMambaBot, UMambaEnc
# from pymic.net.net2d.unet2d_vm import VMUNet
# from pymic.net.net2d.unet2d_vm_light import UltraLight_VM_UNet

from pymic.net.net3d.fmunetv3 import FMUNetV3
from pymic.net.net3d.fmunet import FMUNet
from pymic.net.cnn.lcovnet3d import LCOVNet
from pymic.net.net3d.unet3d_dual_branch import UNet3D_DualBranch

# from pymic.net.net3d.mystunet import MySTUNet
from pymic.net.transformer.unetr import UNETR
from pymic.net.transformer.unetr_pp import UNETR_PP

from pymic.net.specific.cctnet import CCTNet
from pymic.net.specific.dbnet import DBNet
from pymic.net.specific.tdnet import TDNet
from pymic.net.specific.tdnet3d import TDNet3D

# from pymic.net.third_party.nnFormer_wrap import nnFormer_wrap
# from pymic.net.third_party.stunet_wrap import STUNet_wrap
# from pymic.net.third_party.cotr_wrap import CoTr_wrap


SegNetDict = {
	#
	# ---- networks for both 2D and 3D ---- #
	'UNet': UNet,
	'UNet_ScSE': UNet_ScSE,
	'AttentionUNet': AttentionUNet,
	'UNetpp': UNetpp,
	'CANet': CANet,
	#
	# ---- networks for  2D ---- #
	'COPLENet': COPLENet,
	'TransUNet': TransUNet,
	'SwinUNet': SwinUNet,
	# 'MCNet2D': MCNet2D,
	# 'MTNet2D': MTNet2D,
	# 'UNet2D_DualBranch': UNet2D_DualBranch,
    # 'UMambaBot': UMambaBot,
    # 'UMambaEnc': UMambaEnc,
	# 'VMUNet':VMUNet,
    # 'UltraLight_VM_UNet': UltraLight_VM_UNet,
	#
	# ---- networks for 3D ---- #
	'UNet2D5': UNet2D5,
    'LCOVNet': LCOVNet,
    'FMUNet': FMUNet,
	'FMUNetV3': FMUNetV3,
	'UNETR': UNETR,
	'UNETR_PP': UNETR_PP,
	# 'UNet3D_ScSE': UNet3D_ScSE,
	# 'UNet3D_DualBranch': UNet3D_DualBranch, 
	# 'MySTUNet': MySTUNet,
	#
	# ---- special networks for weakly/semi-supervised segmentation ---- #
	'CCTNet': CCTNet,
	'DBNet': DBNet,
	'TDNet': TDNet,
	#
	# ---- thirdy part networks ---- #
	# 'nnFormer': nnFormer_wrap,
	# 'STUNet': STUNet_wrap,
	# 'CoTr': CoTr_wrap
	}
