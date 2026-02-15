# -*- coding: utf-8 -*-
'''`
Implementation of DBNet (dual branch network) used in DMSPS for scribble-supervised segmentation. 
  https://www.sciencedirect.com/science/article/abs/pii/S1361841524001993?dgcid
    @article{han2024dmsps,
    title={DMSPS: Dynamically mixed soft pseudo-label supervision for scribble-supervised medical image segmentation},
    author={Han, Meng and Luo, Xiangde and Xie, Xiangjiang and Liao, Wenjun and Zhang, Shichuan and Song, Tao and Wang, Guotai and Zhang, Shaoting},
    journal={Medical Image Analysis},
    pages={103274},
    year={2024},
    publisher={Elsevier}
}

'''

from __future__ import print_function, division
import copy 
import torch
import torch.nn as nn
from pymic.util.parse_config import *
from pymic.net.cnn.unet import Encoder, Decoder

class MCNet(nn.Nodule):
    """
    A tri-branch network using UNet2D as backbone.

    * Reference: Yicheng Wu, Zongyuan Ge et al. Mutual consistency learning for 
      semi-supervised medical image segmentation.
      `Medical Image Analysis 2022. <https://doi.org/10.1016/j.media.2022.102530>`_ 

    The original code is at: https://github.com/ycwu1997/MC-Net
    
    The parameters for the backbone should be given in the `params` dictionary. 
    See :mod:`pymic.net.net2d.unet2d.UNet2D` for details. 
    """
    def __init__(self, params):
        super(MCNet, self).__init__()
        params    = self.get_default_parameters(params)
        self.dim  = params['dimension']
        params1   = copy.deepcopy(params)
        params2   = copy.deepcopy(params)
        params3   = copy.deepcopy(params)
        params1['up_mode'] = 0
        params2['up_mode'] = 1
        params3['up_mode'] = 2

        self.encoder  = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        self.decoder3 = Decoder(params3)

    def get_default_parameters(self, params):
        default_param = {
            'feature_chns': [32, 64, 128, 256, 512],
            'dropout': [0.0, 0.0, 0.2, 0.3, 0.4],
            'up_mode': 2,
            'multiscale_pred': False,
        }
        for key in default_param:
            params[key] = params.get(key, default_param[key])
        for key in params:
                logging.info("{0:}  = {1:}".format(key, params[key]))
        return params

    def forward(self, x):
        x_shape = list(x.shape)
        if(self.dim == 2 and len(x_shape) == 5):
          [N, C, D, H, W] = x_shape
          new_shape = [N*D, C, H, W]
          x = torch.transpose(x, 1, 2)
          x = torch.reshape(x, new_shape)

        feature = self.encoder(x)
        output1 = self.decoder1(feature)

        if(self.dim == 2 and len(x_shape) == 5):
            new_shape = [N, D] + list(output1.shape)[1:]
            output1   = torch.transpose(torch.reshape(output1, new_shape), 1, 2)
        if(not self.training):
            return output1
        else:
            output2 = self.decoder2(feature)
            output3 = self.decoder3(feature)
            if(self.dim == 2 and len(x_shape) == 5):
                output2 = torch.transpose(torch.reshape(output2, new_shape), 1, 2)
                output3 = torch.transpose(torch.reshape(output3, new_shape), 1, 2)
            return output1, output2, output3