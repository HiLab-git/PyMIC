# -*- coding: utf-8 -*-
from __future__ import print_function, division

from torch import nn


class Initialization_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class Initialization_XavierUniform(object):
    def __init__(self, gain=1):
        self.gain = gain

    def __call__(self, module):
        if isinstance(module, (nn.Conv3d ,nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            module.weight = nn.init.xavier_uniform_(module.weight, self.gain)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
