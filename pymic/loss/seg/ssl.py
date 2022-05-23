
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