# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn

class AbstractSegLoss(nn.Module):
    """
    Abstract class for loss function of segmentation tasks.
    The parameters should be written in the `params` dictionary, and it has the
    following fields:

    :param `loss_softmax`: (optional, bool) 
        Apply softmax to the prediction of network or not. Default is True.
    """
    def __init__(self, params = None):
        super(AbstractSegLoss, self).__init__()
        if(params is None):
            self.acti_func = 'softmax'
        else:
            self.acti_func = params.get('loss_acti_func', 'softmax')

    def get_activated_prediction(self, p, acti_func = 'softmax'):
        if(acti_func == "softmax"):
            p = nn.Softmax(dim = 1)(p)
        elif(acti_func == "tanh"):
            p = nn.Tanh()(p)
        elif(acti_func == "sigmoid"):
            p = nn.Sigmoid()(p)
        else:
            raise ValueError("activation for output is not supported: {0:}".format(acti_func))
        return p 

    def forward(self, loss_input_dict):
        """
        Forward pass for calculating the loss.
        The arguments should be written in the `loss_input_dict` dictionary, 
        and it has the following fields:

        :param `prediction`: (tensor) Prediction of a network, with the 
            shape of [N, C, D, H, W] or [N, C, H, W].
        :param `ground_truth`: (tensor) Ground truth, with the 
            shape of [N, C, D, H, W] or [N, C, H, W]. 
        :param `pixel_weight`: (optional) Pixel-wise weight map, with the
            shape of [N, 1, D, H, W] or [N, 1, H, W]. Default is None.
        :return: Loss function value.
        """
        pass
