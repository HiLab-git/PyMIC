# -*- coding: utf-8 -*-
"""
Built-in networks for classification.

* resnet18 :mod:`pymic.net.cls.torch_pretrained_net.ResNet18`
* vgg16 :mod:`pymic.net.cls.torch_pretrained_net.VGG16` 
* mobilenetv2 :mod:`pymic.net.cls.torch_pretrained_net.MobileNetV2`
"""

from __future__ import print_function, division
from pymic.net.cls.torch_pretrained_net import *

TorchClsNetDict = {
    'resnet18':   ResNet18,
    'vgg16':      VGG16,
    'mobilenetv2':MobileNetV2
}
