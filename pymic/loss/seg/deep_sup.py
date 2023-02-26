# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch.nn as nn
from torch.nn.functional import interpolate
from pymic.loss.seg.abstract import AbstractSegLoss

def match_prediction_and_gt_shape(pred, gt, mode = 0):
    pred_shape = list(pred.shape)
    gt_shape   = list(gt.shape)
    dim = len(pred_shape) - 2
    shape_match = False 
    if(dim == 2):
        if(pred_shape[-1] == gt_shape[-1] and pred_shape[-2] == gt_shape[-2]):
            shape_match = True
    else:
        if(pred_shape[-1] == gt_shape[-1] and pred_shape[-2] == gt_shape[-2]
           and pred_shape[-3] == gt_shape[-3]):
            shape_match = True
    if(shape_match):
        return pred, gt 
    
    interp_mode = 'bilinear' if dim == 2 else 'trilinear'
    if(mode == 0):
        pred_new = interpolate(pred, gt_shape[2:], mode = interp_mode)
        gt_new   = gt  
    elif(mode == 1):
        pred_new = pred
        gt_new   = interpolate(gt, pred_shape[2:], mode = interp_mode)
    elif(mode == 2):
        pred_new = pred
        if(dim == 2):
            avg_pool = nn.AdaptiveAvgPool2d(pred_shape[-2:])
        else:
            avg_pool = nn.AdaptiveAvgPool3d(pred_shape[-3:])
        gt_new = avg_pool(gt)
    else:
        raise ValueError("mode shoud be 0, 1 or 2, but {0:} was given".format(mode))
    return pred_new, gt_new
            

class DeepSuperviseLoss(AbstractSegLoss):
    '''
    Combine deep supervision with a basic loss function.  
    Arguments should be provided in the `params` dictionary, and it has the 
    following fields:

    :param `loss_softmax`: (optional, bool) 
        Apply softmax to the prediction of network or not. Default is True.
    :param `base_loss`: (nn.Module) The basic function used for each scale.
    :param `deep_supervise_weight`: (list) A list of weight for each deep supervision scale. 
    :param `deep_supervise_model`: (int) Mode for deep supervision when the prediction
        has a smaller shape than the ground truth. 0: upsample the prediction to the size 
        of the ground truth. 1: downsample the ground truth to the size of the prediction
        via interpolation. 2: downsample the ground truth via adaptive average pooling.

    '''
    def __init__(self, params):
        super(DeepSuperviseLoss, self).__init__(params)
        self.base_loss       = params['base_loss']
        self.deep_sup_weight = params.get('deep_supervise_weight', None)
        self.deep_sup_mode   = params.get('deep_supervise_mode', 0)

    def forward(self, loss_input_dict):
        pred = loss_input_dict['prediction']
        gt   = loss_input_dict['ground_truth']
        if(not isinstance(pred, (list,tuple))):
            raise ValueError("""For deep supervision, the prediction should
                be a list or a tuple""")
        pred_num = len(pred)
        if(self.deep_sup_weight is None):
            self.deep_sup_weight = [1.0] * pred_num
        else:
            assert(pred_num == len(self.deep_sup_weight))
        loss_sum, weight_sum  = 0.0, 0.0
        for i in range(pred_num):
            pred_i, gt_i = match_prediction_and_gt_shape(pred[i], gt, self.deep_sup_mode)
            loss_input_dict['prediction']   = pred_i
            loss_input_dict['ground_truth'] = gt_i
            temp_loss   = self.base_loss(loss_input_dict)
            loss_sum   += temp_loss * self.deep_sup_weight[i]
            weight_sum += self.deep_sup_weight[i]
        loss = loss_sum/weight_sum
        return loss