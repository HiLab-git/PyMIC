# -*- coding: utf-8 -*-
"""
Spatial Label Smoothing Regularization (SLSR) loss for learning from
noisy annotatins according to the following paper:
    Minqing Zhang, Jiantao Gao et al., Characterizing Label Errors: 
    Confident Learning for Noisy-Labeled Image Segmentation, MICCAI 2020.
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.loss.seg.util import reshape_tensor_to_2D

class SLSRLoss(nn.Module):
    def __init__(self, params):
        super(SLSRLoss, self).__init__()
        if(params is None):
            params = {}
        self.softmax = params.get('loss_softmax', True)
        self.epsilon = params.get('slsrloss_softmax', 0.25)
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        pix_w   = loss_input_dict.get('pixel_weight', None)
        # the pixel wight here is actually the confidence mask
        # i.e., if the value is one, it means the label of corresponding 
        # pixel is noisy and should be replaced with smoothed label.

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y)
        if(pix_w is not None):
            pix_w   = reshape_tensor_to_2D(pix_w > 0).float()

            # smooth labels for pixels in the unconfident mask 
            smooth_y = (soft_y - 0.5) * (0.5 - self.epsilon) / 0.5 + 0.5
            smooth_y = pix_w * smooth_y + (1 - pix_w) * soft_y
        else:
            smooth_y = soft_y

        # for numeric stability
        predict = predict * 0.999 + 5e-4
        ce = - smooth_y* torch.log(predict)
        ce = torch.sum(ce, dim = 1) # shape is [N]
        ce = torch.mean(ce)  
        return ce
