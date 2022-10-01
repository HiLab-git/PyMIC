# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn

class AbstractClassificationLoss(nn.Module):
    """
    Abstract Classification Loss.
    """
    def __init__(self, params = None):
        super(AbstractClassificationLoss, self).__init__()
    
    def forward(self, loss_input_dict):
        """
        The arguments should be written in the `loss_input_dict` dictionary, and it has the
        following fields. 
        
        :param prediction: A prediction with shape of [N, C] where C is the class number.
        :param ground_truth: The corresponding ground truth, with shape of [N, 1].

        Note that `prediction` is the digit output of a network, before using softmax.
        """
        pass

class CrossEntropyLoss(AbstractClassificationLoss):
    """
    Standard Softmax-based  CE loss.
    """
    def __init__(self, params = None):
        super(CrossEntropyLoss, self).__init__(params)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        labels  = loss_input_dict['ground_truth']
        loss = self.ce_loss(predict, labels)
        return loss

class SigmoidCELoss(AbstractClassificationLoss):
    """
    Sigmoid-based CE loss.
    """
    def __init__(self, params = None):
        super(SigmoidCELoss, self).__init__(params)
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        labels  = loss_input_dict['ground_truth']
        predict = nn.Sigmoid()(predict) * 0.999 + 5e-4
        loss = - labels * torch.log(predict) - (1 - labels) * torch.log( 1 - predict)
        loss = loss.mean()
        return loss

class L1Loss(AbstractClassificationLoss):
    """
    L1 (MAE) loss for classification
    """
    def __init__(self, params = None):
        super(L1Loss, self).__init__(params)
        self.l1_loss = nn.L1Loss()
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        labels  = loss_input_dict['ground_truth'][:, None] # reshape to N, 1
        softmax = nn.Softmax(dim = 1)
        predict = softmax(predict)
        loss = self.l1_loss(predict, labels)
        return loss

class MSELoss(AbstractClassificationLoss):
    """
    Mean Square Error loss for classification.
    """
    def __init__(self, params = None):
        super(MSELoss, self).__init__(params)
        self.mse_loss = nn.MSELoss()
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        labels  = loss_input_dict['ground_truth'][:, None] # reshape to N, 1
        softmax = nn.Softmax(dim = 1)
        predict = softmax(predict)
        loss = self.mse_loss(predict, labels)
        return loss

class NLLLoss(AbstractClassificationLoss):
    """
    The negative log likelihood loss for classification.
    """
    def __init__(self, params = None):
        super(NLLLoss, self).__init__(params)
        self.nll_loss = nn.NLLLoss()
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        labels  = loss_input_dict['ground_truth']
        logsoft = nn.LogSoftmax(dim = 1)
        predict = logsoft(predict)
        loss = self.nll_loss(predict, labels)
        return loss