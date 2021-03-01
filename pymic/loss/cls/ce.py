# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, params):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        labels  = loss_input_dict['ground_truth']
        loss = self.ce_loss(predict, labels)
        return loss
