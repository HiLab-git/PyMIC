# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch.nn as nn
from pymic.loss.seg.abstract import AbstractSegLoss

class DeepSuperviseLoss(AbstractSegLoss):
    '''
    Combine deep supervision with a basic loss function.  
    Arguments should be provided in the `params` dictionary, and it has the 
    following fields:

    :param `loss_softmax`: (optional, bool) 
        Apply softmax to the prediction of network or not. Default is True.
    :param `deep_suervise_weight`: (list) A list of weight for each deep supervision scale. \n
    :param `base_loss`: (nn.Module) The basic function used for each scale.

    '''
    def __init__(self, params):
        super(DeepSuperviseLoss, self).__init__(params)
        self.deep_sup_weight = params.get('deep_suervise_weight', None)
        self.base_loss = params['base_loss']

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        if(not isinstance(predict, (list,tuple))):
            raise ValueError("""For deep supervision, the prediction should
                be a list or a tuple""")
        predict_num = len(predict)
        if(self.deep_sup_weight is None):
            self.deep_sup_weight = [1.0] * predict_num
        else:
            assert(predict_num == len(self.deep_sup_weight))
        loss_sum, weight_sum  = 0.0, 0.0
        for i in range(predict_num):
            loss_input_dict['prediction'] =  predict[i]
            temp_loss   = self.base_loss(loss_input_dict)
            loss_sum   += temp_loss * self.deep_sup_weight[i]
            weight_sum += self.deep_sup_weight[i]
        loss = loss_sum/weight_sum
        return loss