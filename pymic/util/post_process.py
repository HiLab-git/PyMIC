# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import numpy as np 
import SimpleITK as sitk 
from pymic.util.image_process import get_largest_k_components

class PostProcess(object):
    def __init__(self, params):
        self.params = params

    def __call__(self, seg):
        return seg

class PostKeepLargestComponent(PostProcess):
    def __init__(self, params):
        super(PostKeepLargestComponent, self).__init__(params)
        self.mode = params.get("KeepLargestComponent_mode".lower(), 1)
        """
        mode = 1: keep the largest component of the union of foreground classes. 
        mode = 2: keep the largest component for each foreground class.
        """

    def __call__(self, seg):
        if(self.mode == 1):
            mask = np.asarray(seg > 0, np.uint8)
            mask = get_largest_k_components(mask)
            seg = seg * mask
        elif(self.mode == 2):
            class_num = seg.max()
            output    = np.zeros_like(seg)
            for c in range(1, class_num + 1):
                seg_c  = np.asarray(seg == c, np.uint8)
                seg_c  = get_largest_k_components(seg_c) 
                output = output + seg_c * c
        return seg

PostProcessDict = {
    'KeepLargestComponent': PostKeepLargestComponent}