# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.loss.seg.abstract import AbstractSegLoss
from pymic.loss.seg.util import reshape_tensor_to_2D

class CrossEntropyLoss(AbstractSegLoss):
    """
    Cross entropy loss for segmentation tasks.

    The parameters should be written in the `params` dictionary, and it has the
    following fields:

    :param `loss_softmax`: (optional, bool) 
        Apply softmax to the prediction of network or not. Default is True.
    """
    def __init__(self, params = None):
        super(CrossEntropyLoss, self).__init__(params)
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        pix_w   = loss_input_dict.get('pixel_weight', None)

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y)

        # for numeric stability
        predict = predict * 0.999 + 5e-4
        ce = - soft_y* torch.log(predict)
        ce = torch.sum(ce, dim = 1) # shape is [N]
        if(pix_w is None):
            ce = torch.mean(ce)  
        else:
            pix_w = torch.squeeze(reshape_tensor_to_2D(pix_w))
            ce = torch.sum(pix_w * ce) / (pix_w.sum() + 1e-5) 
        return ce

class GeneralizedCELoss(AbstractSegLoss):
    """
    Generalized cross entropy loss to deal with noisy labels. 

    * Reference: Z. Zhang et al. Generalized Cross Entropy Loss for Training Deep Neural Networks 
      with Noisy Labels, NeurIPS 2018.

    The parameters should be written in the `params` dictionary, and it has the
    following fields:

    :param `loss_softmax`: (bool) Apply softmax to the prediction of network or not.
    :param `loss_gce_q`: (float): hyper-parameter in the range of (0, 1).  
    :param `loss_with_pixel_weight`: (optional, bool): Use pixel weighting or not. 
    :param `loss_class_weight`: (optional, list or none): If not none, a list of weight for each class.
         
    """
    def __init__(self, params):
        super(GeneralizedCELoss, self).__init__(params)
        self.q = params.get('loss_gce_q', 0.5)
        self.enable_pix_weight = params.get('loss_with_pixel_weight', False)
        self.cls_weight = params.get('loss_class_weight', None)
        
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']        

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y)
        gce     = (1.0 - torch.pow(predict, self.q)) / self.q * soft_y
        
        if(self.cls_weight is not None):
            gce = torch.sum(gce * self.cls_w, dim = 1)
        else:
            gce = torch.sum(gce, dim = 1)
        
        if(self.enable_pix_weight):
            pix_w   = loss_input_dict.get('pixel_weight', None)
            if(pix_w is None):
                raise ValueError("Pixel weight is enabled but not defined")
            pix_w = reshape_tensor_to_2D(pix_w)
            gce    = torch.sum(gce * pix_w) / torch.sum(pix_w)
        else:
            gce = torch.mean(gce)
        return gce 
