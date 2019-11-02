# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn


def get_acti_func(acti_func, params):
    acti_func = acti_func.lower()
    if(acti_func == 'relu'):
        inplace = params.get('relu_inplace', False)
        return nn.ReLU(inplace)

    elif(acti_func == 'leakyrelu'):
        slope   = params.get('leakyrelu_negative_slope', 1e-2)
        inplace = params.get('leakyrelu_inplace', False)
        return nn.LeakyReLU(slope, inplace)

    elif(acti_func == 'prelu'):
        num_params = params.get('prelu_num_parameters', 1)
        init_value = params.get('prelu_init', 0.25)
        return nn.PReLU(num_params, init_value)

    elif(acti_func == 'rrelu'):
        lower   = params.get('rrelu_lower', 1.0 /8)
        upper   = params.get('rrelu_upper', 1.0 /3)
        inplace = params.get('rrelu_inplace', False)
        return nn.RReLU(lower, upper, inplace)

    elif(acti_func == 'elu'):
        alpha   = params.get('elu_alpha', 1.0)
        inplace = params.get('elu_inplace', False)
        return nn.ELU(alpha, inplace)

    elif(acti_func == 'celu'):
        alpha   = params.get('celu_alpha', 1.0)
        inplace = params.get('celu_inplace', False)
        return nn.CELU(alpha, inplace)

    elif(acti_func == 'selu'):
        inplace = params.get('selu_inplace', False)
        return nn.SELU(inplace)

    elif(acti_func == 'glu'):
        dim = params.get('glu_dim', -1)
        return nn.GLU(dim)

    elif(acti_func == 'sigmoid'):
        return nn.Sigmoid()

    elif(acti_func == 'logsigmoid'):
        return nn.LogSigmoid()

    elif(acti_func == 'tanh'):
        return nn.Tanh()

    elif(acti_func == 'hardtanh'):
        min_val = params.get('hardtanh_min_val', -1.0)
        max_val = params.get('hardtanh_max_val',  1.0)
        inplace = params.get('hardtanh_inplace', False)
        return nn.Hardtanh(min_val, max_val, inplace)
    
    elif(acti_func == 'softplus'):
        beta      = params.get('softplus_beta', 1.0)
        threshold = params.get('softplus_threshold', 20)
        return nn.Softplus(beta, threshold)
    
    elif(acti_func == 'softshrink'):
        lambd = params.get('softshrink_lambda', 0.5)
        return nn.Softshrink(lambd)
    
    elif(acti_func == 'softsign'):
        return nn.Softsign()
    
    else:
        raise ValueError("Not implemented: {0:}".format(acti_func))