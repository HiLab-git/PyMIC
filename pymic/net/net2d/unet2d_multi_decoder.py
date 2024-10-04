# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.net.net2d.unet2d import *

class UNet2D_DualBranch(nn.Module):
    """
    A dual branch network using UNet2D as backbone.

    * Reference: Xiangde Luo, Minhao Hu, Wenjun Liao, Shuwei Zhai, Tao Song, Guotai Wang,
      Shaoting Zhang. ScribblScribble-Supervised Medical Image Segmentation via 
      Dual-Branch Network and Dynamically Mixed Pseudo Labels Supervision.
      `MICCAI 2022. <https://arxiv.org/abs/2203.02106>`_ 

    The parameters for the backbone should be given in the `params` dictionary. 
    See :mod:`pymic.net.net2d.unet2d.UNet2D` for details. 
    In addition, the following field should be included:

    :param output_mode: (str) How to obtain the result during the inference. 
      `average`: taking average of the two branches. 
      `first`: takeing the result in the first branch. 
      `second`: taking the result in the second branch.
    """
    def __init__(self, params):
        super(UNet2D_DualBranch, self).__init__()
        params = self.get_default_parameters(params)
        self.output_mode = params["output_mode"]
        self.encoder  = Encoder(params)
        self.decoder1 = Decoder(params)    
        self.decoder2 = Decoder(params)        

    def get_default_parameters(self, params):
        default_param = {
            'feature_chns': [32, 64, 128, 256, 512],
            'dropout': [0.0, 0.0, 0.2, 0.3, 0.4],
            'up_mode': 2,
            'multiscale_pred': False,
            'output_mode': "average"
        }
        for key in default_param:
            params[key] = params.get(key, default_param[key])
        for key in params:
                logging.info("{0:}  = {1:}".format(key, params[key]))
        return params

    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
          [N, C, D, H, W] = x_shape
          new_shape = [N*D, C, H, W]
          x = torch.transpose(x, 1, 2)
          x = torch.reshape(x, new_shape)

        f = self.encoder(x)
        output1 = self.decoder1(f)
        output2 = self.decoder2(f)
        if(len(x_shape) == 5):
            new_shape = [N, D] + list(output1.shape)[1:]
            output1 = torch.reshape(output1, new_shape)
            output1 = torch.transpose(output1, 1, 2)
            output2 = torch.reshape(output2, new_shape)
            output2 = torch.transpose(output2, 1, 2)

        if(self.training):
          return output1, output2
        else:
          if(self.output_mode == "average"):
            return (output1 + output2)/2
          elif(self.output_mode == "first"):
            return output1
          else:
            return output2

class UNet2D_TriBranch(nn.Module):
    """
    A tri-branch network using UNet2D as backbone. The super class for MCNet2D and MTNet2D.

    The parameters for the backbone should be given in the `params` dictionary. 
    See :mod:`pymic.net.net2d.unet2d.UNet2D` for details. 
    """
    def __init__(self, params):
        super(UNet2D_TriBranch, self).__init__()
        params = self.get_default_parameters(params)
        self.encoder  = Encoder(params)
        self.decoder1 = Decoder(params)    
        self.decoder2 = Decoder(params) 
        self.decoder3 = Decoder(params)        

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

class MCNet2D(UNet2D_TriBranch):
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
        super(MCNet2D, self).__init__(params)
        in_chns   = params['in_chns']
        class_num = params['class_num'] 
        ft_chns   = params['feature_chns']
        dropout   = params['dropout']
        params1 = {'in_chns': in_chns,
                  'feature_chns': ft_chns,
                  'dropout': dropout,
                  'class_num': class_num,
                  'up_mode': 0 }
        params2 = {'in_chns': in_chns,
                  'feature_chns': ft_chns,
                  'dropout': dropout,
                  'class_num': class_num,
                  'up_mode': 1 }
        params3 = {'in_chns': in_chns,
                  'feature_chns': ft_chns,
                  'dropout': dropout,
                  'class_num': class_num,
                  'up_mode': 2 }
        self.encoder  = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        self.decoder3 = Decoder(params3)