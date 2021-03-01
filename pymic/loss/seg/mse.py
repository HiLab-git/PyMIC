import torch
import torch.nn as nn
from pymic.loss.seg.util import reshape_tensor_to_2D

class MSELoss(nn.Module):
    def __init__(self, params):
        super(MSELoss, self).__init__()
        self.enable_pix_weight = params['MSELoss_Enable_Pixel_Weight'.lower()]
        self.enable_cls_weight = params['MSELoss_Enable_Class_Weight'.lower()]
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        pix_w   = loss_input_dict['pixel_weight']
        cls_w   = loss_input_dict['class_weight']
        softmax = loss_input_dict['softmax']

        if(softmax):
            predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y)
        se   = self.get_prediction_error(predict, soft_y)
        if(self.enable_cls_weight):
            if(cls_w is None):
                raise ValueError("Class weight is enabled but not defined")
            mse = torch.sum(se * cls_w, dim = 1) / torch.sum(cls_w)
        else:
            mse = torch.mean(se, dim = 1)
        if(self.enable_pix_weight):
            if(pix_w is None):
                raise ValueError("Pixel weight is enabled but not defined")
            pix_w = reshape_tensor_to_2D(pix_w)
            mse   = torch.sum(mse * pix_w) / torch.sum(pix_w)
        else:
            mse = torch.mean(mse)  
        return mse 
    
    def get_prediction_error(self, predict, soft_y):
        diff = predict - soft_y
        error = diff * diff 
        return error 
    
class MAELoss(nn.MSELoss):
    def __init__(self, params):
        super(MAELoss, self).__init__(params)
    
    def get_prediction_error(self, predict, soft_y):
        diff  = predict - soft_y
        error = torch.abs(diff)
        return error 
