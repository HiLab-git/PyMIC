# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np

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

def reshape_prediction_and_ground_truth(predict, soft_y):
    """
    reshape input variables of shape [B, C, D, H, W] to [voxel_n, C]
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
    
    predict = torch.reshape(predict, (-1, num_class)) 
    soft_y  = torch.reshape(soft_y,  (-1, num_class))
      
    return predict, soft_y

def cross_entropy_loss(predict, soft_y, softmax = True):
    """
    get cross entropy scores for each class in predict and soft_y
    """
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    predict, soft_y = reshape_prediction_and_ground_truth(predict, soft_y) 

    ce = - soft_y * torch.log(predict)
    ce = torch.mean(ce, dim = 0)
    ce = torch.sum(ce)
    return ce

def get_classwise_dice(predict, soft_y):
    """
    get dice scores for each class in predict (after softmax) and soft_y
    """
    y_vol = torch.sum(soft_y,  dim = 0)
    p_vol = torch.sum(predict, dim = 0)
    intersect = torch.sum(soft_y * predict, dim = 0)
    dice_score = (2.0 * intersect + 1e-5)/ (y_vol + p_vol + 1e-5)
    return dice_score 

def soft_dice_loss(predict, soft_y, softmax = True):
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    predict, soft_y = reshape_prediction_and_ground_truth(predict, soft_y) 
    dice_score = get_classwise_dice(predict, soft_y)
    dice_loss  = 1.0 - torch.mean(dice_score)   
    return dice_loss

def ce_dice_loss(predict, soft_y, softmax = True):
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    predict, soft_y = reshape_prediction_and_ground_truth(predict, soft_y) 

    ce = - soft_y * torch.log(predict)
    ce = torch.mean(ce, dim = 0)
    ce = torch.sum(ce) 

    dice_score = get_classwise_dice(predict, soft_y)
    dice_loss  = 1.0 - torch.mean(dice_score)

    loss = ce + dice_loss
    return loss

def volume_dice_loss(predict, soft_y, softmax = True):
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    predict, soft_y = reshape_prediction_and_ground_truth(predict, soft_y)
    dice_score = get_classwise_dice(predict, soft_y)
    dice_loss  = 1.0 - torch.mean(dice_score)

    vp = torch.sum(predict, dim = 0)
    vy = torch.sum(predict, dim = 0)
    v_loss = (vp - vy)/vy
    v_loss = v_loss * v_loss
    v_loss = torch.mean(v_loss)

    loss = dice_loss + v_loss * 0.2
    return loss


def hardness_weight_dice_loss(predict, soft_y, softmax = True):
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    predict, soft_y = reshape_prediction_and_ground_truth(predict, soft_y) 

    weight = torch.abs(predict - soft_y)
    lamb   = 0.6
    weight = lamb + weight*(1 - lamb)

    y_vol = torch.sum(soft_y*weight,  dim = 0)
    p_vol = torch.sum(predict*weight, dim = 0)
    intersect = torch.sum(soft_y * predict * weight, dim = 0)
    dice_score = (2.0 * intersect + 1e-5)/ (y_vol + p_vol + 1e-5)
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

def distance_loss(predict, soft_y, lab_distance, softmax = True):
    """
    get distance loss function
    lab_distance is unsigned distance transform of foreground contour
    """
    tensor_dim = len(predict.size())
    num_class  = list(predict.size())[1]
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    if(tensor_dim == 5):
        lab_distance  = lab_distance.permute(0, 2, 3, 4, 1)
        predict = predict.permute(0, 2, 3, 4, 1)
        soft_y  = soft_y.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        lab_distance  = lab_distance.permute(0, 2, 3, 1)
        predict = predict.permute(0, 2, 3, 1)
        soft_y  = soft_y.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))

    lab_distance  = torch.reshape(lab_distance,  (-1, num_class))
    predict = torch.reshape(predict, (-1, num_class))
    soft_y  = torch.reshape(soft_y, (-1, num_class))

    # mis_seg  = torch.abs(predict - soft_y)
    dis_sum  = torch.sum(lab_distance * predict, dim = 0)
    vox_sum  = torch.sum(predict, dim = 0)
    avg_dis  = (dis_sum + 1e-5)/(vox_sum + 1e-5)
    avg_dis  = torch.mean(avg_dis)
    return avg_dis  

def dice_distance_loss(predict, soft_y, lab_distance, softmax = True):
    dice_loss = soft_dice_loss(predict, soft_y, softmax)
    dis_loss  = distance_loss(predict, soft_y, lab_distance, softmax)
    loss = dice_loss + 0.2 * dis_loss
    return loss

segmentation_loss_dict = {'dice_loss': soft_dice_loss,
        'hardness_weight_dice_loss':hardness_weight_dice_loss,
        'volume_dice_loss':volume_dice_loss,
        'generalized_dice_loss':generalized_dice_loss,
        'exponentialized_dice_loss':exponentialized_dice_loss,
        'cross_entropy_loss': cross_entropy_loss,
        'ce_dice_loss': ce_dice_loss,
        'distance_loss':distance_loss,
        'dice_distance_loss': dice_distance_loss}

class SegmentationLossCalculator():
    def __init__(self, loss_name = 'dice_loss', deep_supervision = True):
        self.loss_name = loss_name
        self.loss_func = segmentation_loss_dict[loss_name]
        self.deep_spv  = deep_supervision
        if(self.loss_name not in segmentation_loss_dict):
            raise ValueError("Undefined loss function: {0:}".format(self.loss_name))
    
    def get_loss(self, loss_input_dict, softmax = True):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        lab_dis = None
        if("label_distance" in loss_input_dict):
            lab_dis = loss_input_dict["label_distance"]
        if(isinstance(predict, tuple) or isinstance(predict, list)):
            if(self.deep_spv):
                predict_num = len(predict)
                loss = 0.0
                for i in range(predict_num):
                    temp_loss = self.get_loss_of_single_prediction(predict[i], soft_y, lab_dis, softmax)
                    loss = loss + temp_loss
                loss = loss/predict_num
            else:
                loss = self.get_loss_of_single_prediction(predict[0], soft_y, lab_dis, softmax)
        else:
            loss = self.get_loss_of_single_prediction(predict, soft_y, lab_dis, softmax)
        return loss
    
    def get_loss_of_single_prediction(self, predict, soft_y, lab_dis = None, softmax = True):
        if(self.loss_name == 'distance_loss'):
            loss = self.loss_func(predict, lab_dis, softmax)
        elif(self.loss_name == "dice_distance_loss"):
            loss = self.loss_func(predict, soft_y, lab_dis, softmax)
        else:
            loss = self.loss_func(predict, soft_y, softmax)
        return loss
