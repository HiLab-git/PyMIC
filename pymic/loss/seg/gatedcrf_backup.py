# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np 
from pymic.loss.seg.ce import CrossEntropyLoss
from pymic.loss.seg.gatedcrf_util import ModelLossSemsegGatedCRF

class GatedCRFLoss(nn.Module):
    def __init__(self, params):
        super(GatedCRFLoss, self).__init__()
        self.gcrf_loss = ModelLossSemsegGatedCRF()
        self.softmax = params.get('loss_softmax', True)
        w0 = params['GatedCRFLoss_W0'.lower()]
        xy0= params['GatedCRFLoss_XY0'.lower()]
        rgb= params['GatedCRFLoss_rgb'.lower()]
        w1 = params['GatedCRFLoss_W1'.lower()]
        xy1= params['GatedCRFLoss_XY1'.lower()]
        kernel0 = {'weight': w0, 'xy': xy0, 'rgb': rgb}
        kernel1 = {'weight': w1, 'xy': xy1}
        self.kernels = [kernel0, kernel1]
        self.radius  = params['GatedCRFLoss_Radius'.lower()]
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        image   = loss_input_dict['image']  # should be normalized by mean, std
        scribble= loss_input_dict['scribbles']
        validity_mask = loss_input_dict['validity_mask']

        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)

        batch_dict = {'rgb': image, 
            'semseg_scribbles': scribble}
        x_shape = list(predict.shape)
        l_crf = {'loss': 0}
        if(self.gcrf_w > 0):
            l_crf = self.gcrf_loss(predict,
                    self.kernels,
                    self.radius,
                    batch_dict,
                    x_shape[-2],
                    x_shape[-1],
                    mask_src=validity_mask,
                    out_kernels_vis=False,
                )
        return l_crf['loss']