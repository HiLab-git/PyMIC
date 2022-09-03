# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn

class AbstractSegLoss(nn.Module):
    """
    Cross entropy loss for segmentation tasks.
    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    Args:
        `loss_softmax` (bool): Apply softmax to the prediction of network or not. \n
    """
    def __init__(self, params = None):
        super(AbstractSegLoss, self).__init__()
    
    def forward(self, loss_input_dict):
        """
        Forward pass for calculating the loss.
        The arguments should be written in the `loss_input_dict` dictionary, 
        and it has the following fields:

        :param `prediction`: (tensor) Prediction of a network, with the 
            shape of [N, C, D, H, W] or [N, C, H, W].
        :param `ground_truth`: (tensor) Ground truth, with the 
            shape of [N, C, D, H, W] or [N, C, H, W]. 
        
        :return: Loss function value.
        """
        pass
