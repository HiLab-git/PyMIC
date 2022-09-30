# pretrained models from pytorch: https://pytorch.org/vision/0.8/models.html
from __future__ import print_function, division

import itertools
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

class BuiltInNet(nn.Module):
    """
    Built-in Network in Pytorch for classification.
    Parameters should be set in the `params` dictionary that contains the 
    following fields:

    :param input_chns: (int) Input channel number, default is 3.
    :param pretrain: (bool) Using pretrained model or not, default is True. 
    :param update_mode: (str) The strategy for updating layers: "`all`" means updating
        all the layers, and "`last`" (by default) means updating the last layer, 
        as well as the first layer when `input_chns` is not 3.
    """
    def __init__(self, params):
        super(BuiltInNet, self).__init__()
        self.params   = params
        self.in_chns  = params.get('input_chns', 3)
        self.pretrain = params.get('pretrain', True)
        self.update_mode = params.get('update_mode', "last")
        self.net = None 
    
    def forward(self, x):
        return self.net(x)

    def get_parameters_to_update(self):
        pass

class ResNet18(BuiltInNet):
    """
    ResNet18 for classification.
    Parameters should be set in the `params` dictionary that contains the 
    following fields:

    :param input_chns: (int) Input channel number, default is 3.
    :param pretrain: (bool) Using pretrained model or not, default is True. 
    :param update_mode: (str) The strategy for updating layers: "`all`" means updating
        all the layers, and "`last`" (by default) means updating the last layer, 
        as well as the first layer when `input_chns` is not 3.
    """
    def __init__(self, params):
        super(ResNet18, self).__init__(params)
        self.net = models.resnet18(pretrained = self.pretrain)
        
        # replace the last layer 
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, params['class_num'])

        # replace the first layer when in_chns is not 3
        if(self.in_chns != 3):
            self.net.conv1 = nn.Conv2d(self.in_chns, 64, kernel_size=(7, 7), 
                stride=(2, 2), padding=(3, 3), bias=False)
    
    def get_parameters_to_update(self):
        if(self.update_mode == "all"):
            return self.net.parameters()
        elif(self.update_layers == "last"):
            params = self.net.fc.parameters()
            if(self.in_chns !=3):
                # combining the two iterables into a single one 
                # see: https://dzone.com/articles/python-joining-multiple
                params = itertools.chain()
                for pram in [self.net.fc.parameters(), self.net.conv1.parameters()]:
                    params = itertools.chain(params, pram)
            return  params
        else:
            raise(ValueError("update_mode can only be 'all' or 'last'."))

class VGG16(BuiltInNet):
    """
    VGG16 for classification.
    Parameters should be set in the `params` dictionary that contains the 
    following fields:

    :param input_chns: (int) Input channel number, default is 3.
    :param pretrain: (bool) Using pretrained model or not, default is True. 
    :param update_mode: (str) The strategy for updating layers: "`all`" means updating
        all the layers, and "`last`" (by default) means updating the last layer, 
        as well as the first layer when `input_chns` is not 3.
    """
    def __init__(self, params):
        super(VGG16, self).__init__(params)
        self.net = models.vgg16(pretrained = self.pretrain)
        
        # replace the last layer 
        num_ftrs = self.net.classifier[-1].in_features
        self.net.classifier[-1] = nn.Linear(num_ftrs, params['class_num'])

        # replace the first layer when in_chns is not 3
        if(self.in_chns != 3):
            self.net.features[0] = nn.Conv2d(self.in_chns, 64, kernel_size=(3, 3), 
                stride=(1, 1), padding=(1, 1), bias=False)
    
    def get_parameters_to_update(self):
        if(self.update_mode == "all"):
            return self.net.parameters()
        elif(self.update_mode == "last"):
            params = self.net.classifier[-1].parameters()
            if(self.in_chns !=3):
                params = itertools.chain()
                for pram in [self.net.classifier[-1].parameters(), self.net.net.features[0].parameters()]:
                    params = itertools.chain(params, pram)
            return  params
        else:
            raise(ValueError("update_mode can only be 'all' or 'last'."))

class MobileNetV2(BuiltInNet):
    """
    MobileNetV2 for classification.
    Parameters should be set in the `params` dictionary that contains the 
    following fields:

    :param input_chns: (int) Input channel number, default is 3.
    :param pretrain: (bool) Using pretrained model or not, default is True. 
    :param update_mode: (str) The strategy for updating layers: "`all`" means updating
        all the layers, and "`last`" (by default) means updating the last layer, 
        as well as the first layer when `input_chns` is not 3.
    """
    def __init__(self, params):
        super(MobileNetV2, self).__init__()
        self.net = models.mobilenet_v2(pretrained = self.pretrain)
        
        # replace the last layer 
        num_ftrs = self.net.last_channel
        self.net.classifier[-1] = nn.Linear(num_ftrs, params['class_num'])

        # replace the first layer when in_chns is not 3
        if(self.in_chns != 3):
            self.net.features[0][0] = nn.Conv2d(self.in_chns, 32, kernel_size=(3, 3), 
                stride=(2, 2), padding=(1, 1), bias=False)
    
    def get_parameters_to_update(self):
        if(self.update_mode == "all"):
            return self.net.parameters()
        elif(self.update_mode == "last"):
            params = self.net.classifier[-1].parameters()
            if(self.in_chns !=3):
                params = itertools.chain()
                for pram in [self.net.classifier[-1].parameters(), self.net.net.features[0][0].parameters()]:
                    params = itertools.chain(params, pram)
            return  params
        else:
            raise(ValueError("update_mode can only be 'all' or 'last'."))

if __name__ == "__main__":
    params = {"class_num": 2, "pretrain": False, "input_chns": 3}
    net = ResNet18(params)
    print(net)