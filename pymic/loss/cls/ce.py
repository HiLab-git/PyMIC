# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    """
    Standard Softmax-based  CE loss
    Args:
    predict has a shape of [N, C] where C is the class number
    labels  has a shape of [N]

    note that predict is the digit output of a network, before using softmax
    """
    def __init__(self, params):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        labels  = loss_input_dict['ground_truth']
        loss = self.ce_loss(predict, labels)
        return loss

class SigmoidCELoss(nn.Module):
    """
    Sigmoid-based CE loss
    Args:
    predict has a shape of [N, C] where C is the class number
    labels  has a shape of [N, C] with binary values
    note that predict is the digit output of a network, before using sigmoid."""
    def __init__(self, params):
        super(SigmoidCELoss, self).__init__()
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        labels  = loss_input_dict['ground_truth']
        predict = nn.Sigmoid()(predict) * 0.999 + 5e-4
        loss = - labels * torch.log(predict) - (1 - labels) * torch.log( 1 - predict)
        loss = loss.mean()
        return loss