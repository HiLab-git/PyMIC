# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np

def get_soft_label(input_tensor, num_class, data_type = 'float'):
    """
        convert a label tensor to one-hot soft label 
        input_tensor: tensor with shae [B, 1]
        output_tensor: shape [B, num_class]
    """
    tensor_list = []
    for i in range(num_class):
        temp_prob = input_tensor == i*torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim = 1)
    if(data_type == 'float'):
        output_tensor = output_tensor.float()
    elif(data_type == 'double'):
        output_tensor = output_tensor.double()
    else:
        raise ValueError("data type can only be float and double: {0:}".format(data_type))

    return output_tensor