import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self, params):
        super(MSELoss, self).__init__()
        if(params is None):
            self.softmax = True
        else:
            self.softmax = params.get('loss_softmax', True)
            
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        
        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
        mse  = torch.square(predict - soft_y)
        mse  = torch.mean(mse) 
        return mse 
    
    
class MAELoss(nn.MSELoss):
    def __init__(self, params):
        super(MAELoss, self).__init__(params)
        if(params is None):
            self.softmax = True
        else:
            self.softmax = params.get('loss_softmax', True)
    
    def get_prediction_error(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        
        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
        mae = torch.abs(predict - soft_y)
        mae = torch.mean(mae)
        return mae 
