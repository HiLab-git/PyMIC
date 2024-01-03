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
from pymic.net.net2d.unet2d import UNet2D
from pymic.net.net2d.unet2d_dual_branch import UNet2D_DualBranch
from pymic.net.net2d.unet2d_canet import CANet
from pymic.net.net2d.unet2d_cct import UNet2D_CCT
from pymic.net.net2d.unet2d_mcnet import MCNet2D
from pymic.net.net2d.cople_net import COPLENet
from pymic.net.net2d.unet2d_attention import AttentionUNet2D
from pymic.net.net2d.unet2d_nest import NestedUNet2D
from pymic.net.net2d.unet2d_scse import UNet2D_ScSE
from pymic.net.net2d.trans2d.transunet import TransUNet
from pymic.net.net2d.trans2d.swinunet import SwinUNet
from pymic.net.net3d.unet2d5 import UNet2D5
from pymic.net.net3d.unet3d import UNet3D
from pymic.net.net3d.unet3d_scse import UNet3D_ScSE
from pymic.net.net3d.unet3d_dual_branch import UNet3D_DualBranch
# from pymic.net.net3d.trans3d.nnFormer_wrap import nnFormer_wrap
# from pymic.net.net3d.trans3d.unetr import UNETR
# from pymic.net.net3d.trans3d.unetr_pp import UNETR_PP
# from pymic.net.net3d.trans3d.MedFormer_v1 import MedFormerV1
# from pymic.net.net3d.trans3d.MedFormer_v2 import MedFormerV2
# from pymic.net.net3d.trans3d.MedFormer_v3 import MedFormerV3
# from pymic.net.net3d.trans3d.MedFormer_va1 import MedFormerVA1
# from pymic.net.net3d.trans3d.HiFormer_v1 import HiFormer_v1
# from pymic.net.net3d.trans3d.HiFormer_v2 import HiFormer_v2
# from pymic.net.net3d.trans3d.HiFormer_v3 import HiFormer_v3
# from pymic.net.net3d.trans3d.HiFormer_v4 import HiFormer_v4
# from pymic.net.net3d.trans3d.HiFormer_v5 import HiFormer_v5
# from pymic.net.net3d.trans3d.SwitchNet import SwitchNet

SegNetDict = {
	'UNet2D': UNet2D,
	'UNet2D_DualBranch': UNet2D_DualBranch,
	'UNet2D_CCT': UNet2D_CCT,
	'MCNet2D': MCNet2D,
	'CANet': CANet,
	'COPLENet': COPLENet,
	'AttentionUNet2D': AttentionUNet2D,
	'NestedUNet2D': NestedUNet2D,
	'UNet2D_ScSE': UNet2D_ScSE,
	'TransUNet': TransUNet,
	'SwinUNet': SwinUNet,
	'UNet2D5': UNet2D5,
	'UNet3D': UNet3D,
	'UNet3D_ScSE': UNet3D_ScSE,
	'UNet3D_DualBranch': UNet3D_DualBranch, 
    # 'nnFormer': nnFormer_wrap,
	# 'UNETR': UNETR,
	# 'UNETR_PP': UNETR_PP,
	# 'MedFormerV1': MedFormerV1,
	# 'MedFormerV2': MedFormerV2,
	# 'MedFormerV3': MedFormerV3,
	# 'MedFormerVA1':MedFormerVA1,
	# 'HiFormer_v1': HiFormer_v1,
	# 'HiFormer_v2': HiFormer_v2,
	# 'HiFormer_v3': HiFormer_v3,
    # 'HiFormer_v4': HiFormer_v4, 
    # 'HiFormer_v5': HiFormer_v5
	# 'SwitchNet': SwitchNet
	}
