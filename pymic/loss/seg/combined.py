# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, params, loss_dict):
        super(CombinedLoss, self).__init__()
        loss_names  = params['loss_type']
        self.loss_weight = params['loss_weight']
        assert (len(loss_names) == len(self.loss_weight))
        self.loss_list = []
        for loss_name in loss_names:
            if(loss_name in loss_dict):
                one_loss = loss_dict[loss_name](params)
                self.loss_list.append(one_loss)
            else:
                raise ValueError("{0:} is not defined, or has not been added to the \
                    loss dictionary".format(loss_name))

    def forward(self, loss_input_dict):
        loss_value = 0.0
        for i in range(len(self.loss_list)):
            loss_value += self.loss_weight[i]*self.loss_list[i](loss_input_dict)
        return loss_value
