# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn

class NLLLoss(nn.Module):
    def __init__(self, params):
        super(NLLLoss, self).__init__()
        self.nll_loss = nn.NLLLoss()
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        labels  = loss_input_dict['ground_truth']
        logsoft = nn.LogSoftmax(dim = 1)
        predict = logsoft(predict)
        loss = self.nll_loss(predict, labels)
        return loss
