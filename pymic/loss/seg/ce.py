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

    :param `loss_acti_func`: (optional, string) 
        Apply an activation function to the prediction of network or not, for example,
        'softmax' for image segmentation tasks, 'tanh' for reconstruction tasks, and None
        means no activation is used. 
    """
    def __init__(self, params = None):
        super(CrossEntropyLoss, self).__init__(params)
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        pix_w   = loss_input_dict.get('pixel_weight', None)
        cls_w   = loss_input_dict.get('class_weight', None)

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.acti_func is not None):
            predict = self.get_activated_prediction(predict, self.acti_func)

        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y)

        # for numeric stability
        predict = predict * 0.999 + 5e-4
        ce = - soft_y* torch.log(predict)
        if(cls_w is not None):
            ce = torch.sum(ce*cls_w, dim = 1)
        else:
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
        
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']  
        pix_w   = loss_input_dict.get('pixel_weight', None)
        cls_w   = loss_input_dict.get('class_weight', None)      

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.acti_func is not None):
            predict = self.get_activated_prediction(predict, self.acti_func)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y)
        gce     = (1.0 - torch.pow(predict, self.q)) / self.q * soft_y
        
        if(cls_w is not None):
            gce = torch.sum(gce * cls_w, dim = 1)
        else:
            gce = torch.sum(gce, dim = 1)
        
        if(pix_w is not None):
            pix_w = torch.squeeze(reshape_tensor_to_2D(pix_w))
            gce   = torch.sum(gce * pix_w) / torch.sum(pix_w)
        else:
            gce = torch.mean(gce)
        return gce 
