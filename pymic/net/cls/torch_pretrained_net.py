# pretrained models from pytorch: https://pytorch.org/vision/0.8/models.html
from __future__ import print_function, division

import torch
import torch.nn as nn
import torchvision.models as models

# buildin_net_dict={
#     'resnet18': models.resnet18,
#     'alexnet':  models.alexnet,
#     'vgg16':    models.vgg16,
#     'squeezenet': models.squeezenet1_0,
#     'densenet': models.densenet161,
#     'inception':models.inception_v3,
#     'googlenet':models.googlenet,
#     'shufflenet': models.shufflenet_v2_x1_0,
#     'mobilenet':models.mobilenet,
#     'resnext50':models.resnext50_32x4d,
#     'wide_resnet': models.wide_resnet50_2,
#     'mnasnet': models.mnasnet1_0
# }

class ResNet18(nn.Module):
    def __init__(self, params):
        super(ResNet18, self).__init__()
        self.params    = params
        net_name = params['net_type']
        cls_num  = params['class_num']
        in_chns  = params['input_chns']
        self.pretrain = params['pretrain']
        self.update_layers = params.get('update_layers', 0)
        self.net = models.resnet18(pretrained = self.pretrain)
        
        # replace the last layer 
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, cls_num)
    
    def forward(self, x):
        return self.net(x)
    
    def get_parameters_to_update(self):
        if(self.pretrain == False or self.update_layers == 0):
            return self.net.parameters()
        elif(self.update_layers == -1):
            return self.net.fc.parameters()
        else:
            raise(ValueError("update_layers can only be 0 (all layers) " +
                "or -1 (the last layer)"))

class VGG16(nn.Module):
    def __init__(self, params):
        super(VGG16, self).__init__()
        self.params    = params
        net_name = params['net_type']
        cls_num  = params['class_num']
        in_chns  = params['input_chns']
        self.pretrain = params['pretrain']
        self.update_layers = params.get('update_layers', 0)
        self.net = models.vgg16(pretrained = self.pretrain)
        
        # replace the last layer 
        num_ftrs = self.net.classifier[-1].in_features
        self.net.classifier[-1] = nn.Linear(num_ftrs, cls_num)
    
    def forward(self, x):
        return self.net(x)
    
    def get_parameters_to_update(self):
        if(self.pretrain == False or self.update_layers == 0):
            return self.net.parameters()
        elif(self.update_layers == -1):
            return self.net.classifier[-1].parameters()
        else:
            raise(ValueError("update_layers can only be 0 (all layers) " +
                "or -1 (the last layer)"))
