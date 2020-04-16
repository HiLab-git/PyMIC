# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np

def get_soft_label(input_tensor, num_class, data_type = 'float'):
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
    if(data_type == 'float'):
        output_tensor = output_tensor.float()
    elif(data_type == 'double'):
        output_tensor = output_tensor.double()
    else:
        raise ValueError("data type can only be float and double: {0:}".format(data_type))

    return output_tensor

def reshape_tensor_to_2D(x):
    """
    reshape input variables of shape [B, C, D, H, W] to [voxel_n, C]
    """
    tensor_dim = len(x.size())
    num_class  = list(x.size())[1]
    if(tensor_dim == 5):
        x_perm  = x.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        x_perm  = x.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))
    
    y = torch.reshape(x_perm, (-1, num_class)) 
    return y 

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
    predict = reshape_tensor_to_2D(predict)
    soft_y  = reshape_tensor_to_2D(soft_y)

    ce = - soft_y * torch.log(predict)
    ce = torch.sum(ce, dim = 1)
    ce = torch.mean(ce)
    return ce

def generalized_ce_loss(predict, soft_y, softmax = True):
    """
    get generalized cross entropy loss to deal with noisy labels. 
    Reference: Generalized Cross Entropy Loss for Training Deep Neural Networks 
               with Noisy Labels, NeurIPS 2018.
    """
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    predict = reshape_tensor_to_2D(predict)
    soft_y  = reshape_tensor_to_2D(soft_y)
    n_voxel = list(predict.size())[0]
    q       = 0.7
    gce     = (1.0 - torch.pow(predict, q)) / q * soft_y
    gce     = gce.sum() / n_voxel
    return gce

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
    predict = reshape_tensor_to_2D(predict)
    soft_y  = reshape_tensor_to_2D(soft_y) 
    dice_score = get_classwise_dice(predict, soft_y)
    dice_loss  = 1.0 - torch.mean(dice_score)   
    return dice_loss

def noise_robust_dice_loss(predict, soft_y, softmax, p):
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    predict = reshape_tensor_to_2D(predict)
    soft_y  = reshape_tensor_to_2D(soft_y)
    
    numerator = torch.abs(predict - soft_y)
    numerator = torch.pow(numerator, p)
    numerator = torch.sum(numerator, dim = 0)
    y_vol = torch.sum(soft_y,  dim = 0)
    p_vol = torch.sum(predict, dim = 0)
    loss = (numerator + 1e-5) / (y_vol + p_vol + 1e-5)
    return torch.mean(loss) 

def mae_loss(predict, soft_y, softmax = True):
    """
    loss based on mean absolute value of error. 
    """
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    diff = predict - soft_y
    mae  = diff.abs().mean()
    return mae

def mse_loss(predict, soft_y, softmax = True):
    """
    loss based on mean absolute value of error. 
    """
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    diff = predict - soft_y
    mse  = diff*diff 
    mse  = mse.mean()
    return mse

def exp_log_loss(predict, soft_y, softmax = True):
    """
    The exponential logarithmic loss in this paper: 
    Ken C. L. Wong, Mehdi Moradi, Hui Tang, Tanveer F. Syeda-Mahmood: 3D Segmentation with 
    Exponential Logarithmic Loss for Highly Unbalanced Object Sizes. MICCAI (3) 2018: 612-619.
    """
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    predict = reshape_tensor_to_2D(predict)
    soft_y  = reshape_tensor_to_2D(soft_y)
    gamma   = 0.3
    w_dice  = 0.8
    dice_score = get_classwise_dice(predict, soft_y)
    dice_score = 0.01 + dice_score * 0.98
    exp_dice   = -torch.log(dice_score)
    exp_dice   = torch.pow(exp_dice, gamma)
    exp_dice   = torch.mean(exp_dice)

    predict= 0.01 + predict * 0.98
    wc     = torch.mean(soft_y, dim = 0)
    wc     = 1.0 / (wc + 0.1)
    wc     = torch.pow(wc, 0.5)
    ce     = - torch.log(predict)
    exp_ce = wc * torch.pow(ce, gamma)
    exp_ce = torch.sum(soft_y * exp_ce, dim = 1)
    exp_ce = torch.mean(exp_ce)

    loss = exp_dice * w_dice + exp_ce * (1.0 - w_dice)
    return loss


def volume_weighted_dice(predict, soft_y, softmax = True):
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    predict = reshape_tensor_to_2D(predict)
    soft_y  = reshape_tensor_to_2D(soft_y)
    dice_score = get_classwise_dice(predict, soft_y)
    vol = torch.sum(soft_y, dim = 0)
    wht = 1.0 - nn.Softmax()(vol)
    dice_loss  = 1.0 - torch.sum(dice_score * wht)   
    return dice_loss

def ce_dice_loss(predict, soft_y, softmax = True):
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    predict = reshape_tensor_to_2D(predict)
    soft_y  = reshape_tensor_to_2D(soft_y) 

    ce = - soft_y * torch.log(predict)
    ce = torch.sum(ce, dim = 1)
    ce = torch.mean(ce)

    dice_score = get_classwise_dice(predict, soft_y)
    dice_loss  = 1.0 - torch.mean(dice_score)

    loss = ce + dice_loss
    return loss

def uncertainty_dice_loss(predict, soft_y, gumb, softmax = True):
    predict_g = predict + gumb
    predict_g = nn.Softmax(dim = 1)(predict_g)
    predict_g, soft_y = reshape_prediction_and_ground_truth(predict_g, soft_y) 
    dice_score = get_classwise_dice(predict_g, soft_y)
    dice_loss  = 1.0 - torch.mean(dice_score)  
    return dice_loss

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
    """
    voxel-wise hardness weighted dice loss, proposed in the following paper:
    Guotai Wang, Jonathan Shapey, Wenqi Li, et al. Automatic Segmentation of Vestibular Schwannoma from 
    T2-Weighted MRI by Deep Spatial Attention with Hardness-Weighted Loss. MICCAI (2) 2019: 264-272
    """
    if(softmax):
        predict = nn.Softmax(dim = 1)(predict)
    predict = reshape_tensor_to_2D(predict)
    soft_y  = reshape_tensor_to_2D(soft_y) 

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
    predict = reshape_tensor_to_2D(predict)
    soft_y  = reshape_tensor_to_2D(soft_y)  
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
        'noise_robust_dice_loss': noise_robust_dice_loss,
        'mae_loss': mae_loss,
        'mse_loss': mse_loss, 
        'exp_log_loss': exp_log_loss,
        'volume_weighted_dice': volume_weighted_dice,
        'hardness_weight_dice_loss':hardness_weight_dice_loss,
        'volume_dice_loss':volume_dice_loss,
        'generalized_dice_loss':generalized_dice_loss,
        'exponentialized_dice_loss':exponentialized_dice_loss,
        'cross_entropy_loss': cross_entropy_loss,
        'generalized_ce_loss': generalized_ce_loss,
        'ce_dice_loss': ce_dice_loss,
        'distance_loss':distance_loss,
        'dice_distance_loss': dice_distance_loss, 
        'uncertainty_dice_loss':uncertainty_dice_loss}

class SegmentationLossCalculator():
    def __init__(self, loss_name = 'dice_loss', multi_pred_weight = None):
        self.loss_name = loss_name
        self.loss_func = segmentation_loss_dict[loss_name]
        self.multi_pred_weight  = multi_pred_weight
        if(self.loss_name not in segmentation_loss_dict):
            raise ValueError("Undefined loss function: {0:}".format(self.loss_name))
        self.uncertain_reg_wht = None
        self.noise_robust_dice_loss_p = None
    
    def set_uncertainty_dice_loss_reg_weight(self, wht = 0.08):
        self.uncertain_reg_wht = wht 
    
    def set_noise_robust_dice_loss_p(self, p):
        self.noise_robust_dice_loss_p = p

    def get_loss(self, loss_input_dict, softmax = True):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        lab_dis = None
        if("label_distance" in loss_input_dict):
            lab_dis = loss_input_dict["label_distance"]
        if(isinstance(predict, tuple) or isinstance(predict, list)):
            if(self.multi_pred_weight is not None):
                predict_num = len(predict)
                loss   = 0.0
                weight = 0.0
                for i in range(predict_num):
                    temp_lab_dis = lab_dis[i] if lab_dis is not None else None
                    temp_loss = self.get_loss_of_single_prediction(predict[i], soft_y, temp_lab_dis, softmax)
                    loss      = loss + temp_loss * self.multi_pred_weight[i]
                    weight    = weight + self.multi_pred_weight[i]
                loss = loss/weight
            else:
                loss = self.get_loss_of_single_prediction(predict[0], soft_y, lab_dis, softmax)
        else:
            if(isinstance(lab_dis, tuple) or isinstance(lab_dis, list)):
                lab_dis = lab_dis[0]
            loss = self.get_loss_of_single_prediction(predict, soft_y, lab_dis, softmax)
        if(self.loss_name == "uncertainty_dice_loss"):
            if(isinstance(predict, tuple) or isinstance(predict, list)):
                predict_num = len(predict)
                reg    = 0.0
                weight = 0.0
                for i in range(predict_num):
                    reg    = reg + torch.mean(predict[i] * predict[i]) * self.multi_pred_weight[i]
                    weight = weight + self.multi_pred_weight[i]
                reg = reg / weight
            else:
                reg = torch.mean(predict * predict)
            loss = loss + self.uncertain_reg_wht*reg
        return loss
    
    def get_loss_of_single_prediction(self, predict, soft_y, lab_dis = None, softmax = True):
        if(self.loss_name == 'noise_robust_dice_loss'):
            loss = self.loss_func(predict, soft_y, softmax, self.noise_robust_dice_loss_p)
        elif(self.loss_name == 'distance_loss'):
            loss = self.loss_func(predict, lab_dis, softmax)
        elif(self.loss_name == "dice_distance_loss"):
            loss = self.loss_func(predict, soft_y, lab_dis, softmax)
        elif(self.loss_name ==  "uncertainty_dice_loss"):
            loss = self.loss_func(predict, soft_y, lab_dis, softmax)
        else:
            loss = self.loss_func(predict, soft_y, softmax)
        return loss
