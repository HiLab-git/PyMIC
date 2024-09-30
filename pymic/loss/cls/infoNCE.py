# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn

class InfoNCELoss(nn.Module):
    """
    Abstract Classification Loss.
    """
    def __init__(self, params = None):
        super(InfoNCELoss, self).__init__()
        self.temp = params.get("temperature", 0.1)
    
    def forward(self, input_1, input_2):
        """
        The arguments should be written in the `loss_input_dict` dictionary, and it has the
        following fields. 
        
        :param prediction: A prediction with shape of [N, C] where C is the class number.
        :param ground_truth: The corresponding ground truth, with shape of [N, 1].

        Note that `prediction` is the digit output of a network, before using softmax.
        """
        B = list(input_1.shape)[0]
        loss = 0.0
        for b in range(B):
            embeds_1 = input_1[b]
            embeds_2 = input_2[b]
            logits_11 = torch.matmul(embeds_1, embeds_1.T) / self.temp
            logits_11.fill_diagonal_(float('-inf'))
            logits_12 = torch.matmul(embeds_1, embeds_2.T) / self.temp
            logits_22 = torch.matmul(embeds_2, embeds_2.T) / self.temp
            logits_22.fill_diagonal_(float('-inf'))
            loss_1 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_11, logits_12], dim=1), dim=1))
            loss_2 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_12.T, logits_22], dim=1), dim=1))
            loss = loss + (loss_1 + loss_2) / 2
        loss = loss / B 
        return loss