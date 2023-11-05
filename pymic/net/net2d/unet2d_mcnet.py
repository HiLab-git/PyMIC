# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch.nn as nn
from pymic.net.net2d.unet2d import *

class MCNet2D(nn.Module):
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
        super(MCNet2D, self).__init__()
        in_chns   = params['in_chns']
        class_num = params['class_num'] 
        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_mode': 0,
                  'multiscale_pred': False }
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_mode': 1,
                  'multiscale_pred': False}
        params3 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_mode': 2,
                  'multiscale_pred': False}
        self.encoder  = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        self.decoder3 = Decoder(params3)
        
    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
          [N, C, D, H, W] = x_shape
          new_shape = [N*D, C, H, W]
          x = torch.transpose(x, 1, 2)
          x = torch.reshape(x, new_shape)

        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        new_shape = [N, D] + list(output1.shape)[1:]
        output1   = torch.transpose(torch.reshape(output1, new_shape), 1, 2)
        if(not self.training):
          return output1
        output2 = self.decoder2(feature)
        output3 = self.decoder3(feature)
        if(len(x_shape) == 5):
          output2 = torch.transpose(torch.reshape(output2, new_shape), 1, 2)
          output3 = torch.transpose(torch.reshape(output3, new_shape), 1, 2)
        return output1, output2, output3
