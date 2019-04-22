# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn

def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label 
        input_tensor: tensor with shae [B, 1, D, H, W]
        output_tensor: shape [B, num_class, D, H, W]
    """
    tensor_list = []
    for i in range(num_class):
        temp_prob = input_tensor == i*torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim = 1)
    output_tensor = output_tensor.double()

    return output_tensor

def get_classwise_dice(predict, soft_y):
    """
    get dice scores for each class in predict and soft_y
    """
    tensor_dim = len(predict.size())
    num_class  = list(predict.size())[1]
    if(tensor_dim == 5):
        soft_y  = soft_y.permute(0, 2, 3, 4, 1)
        predict = predict.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        soft_y  = soft_y.permute(0, 2, 3, 1)
        predict = predict.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))
    
    soft_y  = torch.reshape(soft_y,  (-1, num_class))
    predict = torch.reshape(predict, (-1, num_class))

    y_vol = torch.sum(soft_y,  dim = 0)
    p_vol = torch.sum(predict, dim = 0)
    intersect = torch.sum(soft_y * predict, dim = 0)
    dice_score = (2.0 * intersect + 1e-5)/ (y_vol + p_vol + 1e-5)
    return dice_score 

def soft_dice_loss(predict, soft_y, softmax = True):
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    dice_score = get_classwise_dice(predict, soft_y)
    dice_loss  = 1.0 - torch.mean(dice_score)   
    return dice_loss

def exponentialized_dice_loss(predict, soft_y, softmax = True):
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    dice_score = get_classwise_dice(predict, soft_y)
    exp_dice = - torch.log(dice_score)
    exp_dice = torch.mean(exp_dice)
    return exp_dice

def generalized_dice_loss(predict, soft_y, softmax = True):
    tensor_dim = len(predict.size())
    num_class  = list(predict.size())[1]
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    if(tensor_dim == 5):
        soft_y  = soft_y.permute(0, 2, 3, 4, 1)
        predict = predict.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        soft_y  = soft_y.permute(0, 2, 3, 1)
        predict = predict.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))
    
    soft_y  = torch.reshape(soft_y,  (-1, num_class))
    predict = torch.reshape(predict, (-1, num_class))
    num_voxel = list(soft_y.size())[0]
    vol = torch.sum(soft_y, dim = 0)
    weight = (num_voxel - vol)/num_voxel
    intersect = torch.sum(predict * soft_y, dim = 0)
    intersect = torch.sum(weight * intersect)
    vol_sum = torch.sum(soft_y, dim = 0) + torch.sum(predict, dim = 0)
    vol_sum = torch.sum(weight * vol_sum)
    dice_score = (2.0 * intersect + 1e-5) / (vol_sum + 1e-5)
    dice_loss = 1.0 - dice_score
    return dice_loss

segmentation_loss_dict = {'dice_loss': soft_dice_loss,
        'generalized_dice_loss':generalized_dice_loss,
        'exponentialized_dice_loss':exponentialized_dice_loss}

class SegmentationLossCalculator():
    def __init__(self, loss_name = 'dice_loss'):
        self.loss_name = loss_name
        if(self.loss_name not in segmentation_loss_dict):
            raise ValueError("Undefined loss function: {0:}".format(self.loss_name))
    
    def get_loss(self, predict, softy, softmax = True):
        return segmentation_loss_dict[self.loss_name](predict,
            softy, softmax)