
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch.nn as nn
from pymic.loss.seg.abstract import AbstractSegLoss

class ARSTverskyLoss(AbstractSegLoss):
    """
    The Adaptive Region-Specific Loss in this paper: 
        
    * Y. Chen et al.: Adaptive Region-Specific Loss for Improved Medical Image Segmentation.
      `IEEE TPAMI 2023. <https://ieeexplore.ieee.org/document/10163830>`_

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `ARSTversky_patch_size`: (list) the patch size.
    :param `A`: the lowest weight for FP or FN (default 0.3)
    :param `B`: the gap between lowest and highest weight (default 0.4)
    """
    def __init__(self, params):
        super(ARSTverskyLoss, self).__init__(params)
        self.patch_size = params['ARSTversky_patch_size'.lower()]
        self.a = params.get('ARSTversky_a'.lower(), 0.3)
        self.b = params.get('ARSTversky_b'.lower(), 0.4)

        self.dim = len(self.patch_size)
        assert self.dim in [2, 3], "The num of dim must be 2 or 3."
        if self.dim == 3:
            self.pool = nn.AvgPool3d(kernel_size=self.patch_size, stride=self.patch_size)
        elif self.dim == 2:
            self.pool = nn.AvgPool2d(kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        
        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.acti_func is not None):
            predict = self.get_activated_prediction(predict, self.acti_func)
        
        smooth = 1e-5
        if self.dim == 2:
            assert predict.shape[-2] % self.patch_size[0] == 0, "image size % patch size must be 0 in dimension y"
            assert predict.shape[-1] % self.patch_size[1] == 0, "image size % patch size must be 0 in dimension x"
        elif self.dim == 3:
            assert predict.shape[-3] % self.patch_size[0] == 0, "image size % patch size must be 0 in dimension z"
            assert predict.shape[-2] % self.patch_size[1] == 0, "image size % patch size must be 0 in dimension y"
            assert predict.shape[-1] % self.patch_size[2] == 0, "image size % patch size must be 0 in dimension x"

        tp = predict * soft_y
        fp = predict * (1 - soft_y)
        fn = (1 - predict) * soft_y

        region_tp = self.pool(tp)
        region_fp = self.pool(fp)
        region_fn = self.pool(fn)

        alpha = self.a + self.b * (region_fp + smooth) / (region_fp + region_fn + smooth)
        beta = self.a + self.b * (region_fn + smooth) / (region_fp + region_fn + smooth)

        region_tversky = (region_tp + smooth) / (region_tp + alpha * region_fp + beta * region_fn + smooth)
        region_tversky = 1 - region_tversky
        loss = region_tversky.mean()
        return loss