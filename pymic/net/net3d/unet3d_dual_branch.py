# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.net.net3d.unet3d import *

class UNet3D_DualBranch(nn.Module):
    """
    A dual branch network using UNet3D as backbone.

    * Reference: Xiangde Luo, Minhao Hu, Wenjun Liao, Shuwei Zhai, Tao Song, Guotai Wang,
      Shaoting Zhang. ScribblScribble-Supervised Medical Image Segmentation via 
      Dual-Branch Network and Dynamically Mixed Pseudo Labels Supervision.
      `MICCAI 2022. <https://arxiv.org/abs/2203.02106>`_ 

    The parameters for the backbone should be given in the `params` dictionary. 
    See :mod:`pymic.net.net3d.unet3d.UNet3D` for details. 
    In addition, the following field should be included:

    :param output_mode: (str) How to obtain the result during the inference. 
      `average`: taking average of the two branches. 
      `first`: takeing the result in the first branch. 
      `second`: taking the result in the second branch.
    """
    def __init__(self, params):
        super(UNet3D_DualBranch, self).__init__()
        self.output_mode = params.get("output_mode", "average")
        self.encoder  = Encoder(params)
        self.decoder1 = Decoder(params)    
        self.decoder2 = Decoder(params)        

    def forward(self, x):
        f = self.encoder(x)
        output1 = self.decoder1(f)
        output2 = self.decoder2(f)

        if(self.training):
          return output1, output2
        else:
          if(self.output_mode == "average"):
            return (output1 + output2)/2
          elif(self.output_mode == "first"):
            return output1
          else:
            return output2
