
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np 
from pymic.loss.seg.util import reshape_tensor_to_2D

class EntropyLoss(nn.Module):
    """
    Minimize the entropy for each pixel
    """
    def __init__(self, params = None):
        super(EntropyLoss, self).__init__()
        if(params is None):
            self.softmax = True
        else:
            self.softmax = params.get('loss_softmax', True)
        
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)

        # for numeric stability
        predict = predict * 0.999 + 5e-4
        C = list(predict.shape)[1]
        entropy = torch.sum(-predict*torch.log(predict), dim=1) / np.log(C)
        avg_ent = torch.mean(entropy)
        return avg_ent
    
class TotalVariationLoss(nn.Module):
    """
    Minimize the total variation of a segmentation
    """
    def __init__(self, params = None):
        super(TotalVariationLoss, self).__init__()
        if(params is None):
            self.softmax = True
        else:
            self.softmax = params.get('loss_softmax', True)
        
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)

        # for numeric stability
        predict = predict * 0.999 + 5e-4
        dim = list(predict.shape)[2:]
        if(dim == 2):
            pred_min = -1 * nn.functional.max_pool2d(-1*predict, (3, 3), 1, 1)
            pred_max = nn.functional.max_pool2d(pred_min, (3, 3), 1, 1)
        else:
            pred_min = -1 * nn.functional.max_pool3d(-1*predict, (3, 3, 3), 1, 1)
            pred_max = nn.functional.max_pool3d(pred_min, (3, 3, 3), 1, 1)            
        contour = torch.relu(pred_max - pred_min)
        length  = torch.mean(contour)
        return length