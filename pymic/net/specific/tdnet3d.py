# -*- coding: utf-8 -*-
from __future__ import print_function, division

import logging
import copy
import torch
import torch.nn as nn
import numpy as np
import random

class ConvBlock3D(nn.Module):
    """
    Two 3D convolution layers with batch norm and leaky relu.
    Droput is used between the two convolution layers.
    
    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dropout_p, kernel_size=3,  
        padding_size=1, dilation_size=1):
        super(ConvBlock3D, self).__init__()
        norm1 = nn.BatchNorm3d(out_channels, affine = True)
        norm2 = nn.BatchNorm3d(out_channels, affine = True)
        self.conv_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, 
                padding = padding_size, dilation = dilation_size),
            norm1,
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size = kernel_size, 
                padding = padding_size, dilation = dilation_size),
            norm2,
            nn.LeakyReLU()
        )
       
    def forward(self, x):
        return self.conv_conv(x)


class UpBlock3D(nn.Module):
    """upsample +'compare shape' + cat + conv(含droupout); out = conv"""
    def __init__(self, in_channels, out_channels, dropout_p, kernel_size = 3, 
        padding_size = 1, dilation_size = 1):
        super(UpBlock3D, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear') 
        self.conv = ConvBlock3D(in_channels + out_channels, out_channels, dropout_p, kernel_size,  
            padding_size, dilation_size) 

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class Encoder(nn.Module): 
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        in_chns   = self.params['in_chns']
        ft_chns   = self.params['feature_chns']
        dropout   = self.params['dropout']
        # init_type = self.params['init_type']

        # self.in_channels = self.params['in_chns'] 
        # self.feature_scale = self.params['feature_scale']        
        # self.is_batchnorm = self.params['is_batchnorm'] # 默认True,IN
        # self.dropout_p = self.params['dropout'] #[0.05, 0.1, 0.2, 0.3, 0.5]
        # self.init_type = self.params['init_type']  #select mode: kaiming (default), xavier, normal, orthogonal
    

        # filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]
        self.conv1 = ConvBlock3D(in_chns, ft_chns[0], dropout[0])
        self.conv2 = ConvBlock3D(ft_chns[0], ft_chns[1], dropout[1])
        self.conv3 = ConvBlock3D(ft_chns[1], ft_chns[2], dropout[2])
        self.conv4 = ConvBlock3D(ft_chns[2], ft_chns[3], dropout[3])
        self.conv5 = ConvBlock3D(ft_chns[3], ft_chns[4], dropout[4])
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)      
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)        
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)        
        self.maxpool4 = nn.MaxPool3d(kernel_size=2)

        
        # self.conv1 = ConvBlock3d(self.in_channels, filters[0], self.dropout_p[0], self.is_batchnorm, kernel_size=(
        #     3, 3, 3), padding_size=(1, 1, 1))
        # self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # self.conv2 = ConvBlock3d(filters[0], filters[1], self.dropout_p[1], self.is_batchnorm, kernel_size=(
        #     3, 3, 3), padding_size=(1, 1, 1))
        # self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # self.conv3 = ConvBlock3d(filters[1], filters[2], self.dropout_p[2], self.is_batchnorm, kernel_size=(
        #     3, 3, 3), padding_size=(1, 1, 1))
        # self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # self.conv4 = ConvBlock3d(filters[2], filters[3], self.dropout_p[3], self.is_batchnorm, kernel_size=(
        #     3, 3, 3), padding_size=(1, 1, 1))
        # self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # self.center = ConvBlock3d(filters[3], filters[4], self.dropout_p[4], self.is_batchnorm, kernel_size=(
        #     3, 3, 3), padding_size=(1, 1, 1))

        # initialise weights 
        # for m in self.modules(): #kaiming initialization
        #     if isinstance(m, nn.Conv3d):
        #         init_weights(m, init_type = init_type)
        #     elif isinstance(m, nn.BatchNorm3d):
        #         init_weights(m, init_type = init_type)
        self.apply(self.init_weight)

    def init_weight(self, m):
        if(isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

    def forward(self, inputs):
        x0 = self.conv1(inputs)
        x1 = self.conv2(self.maxpool1(x0))
        x2 = self.conv3(self.maxpool2(x1))
        x3 = self.conv4(self.maxpool3(x2))
        x4 = self.conv5(self.maxpool4(x3))
        return [x0, x1, x2, x3, x4]

def Dropout_random(x):
    dropout_p = round(random.uniform(0.2, 0.5),2)
    x = torch.nn.functional.dropout3d(x, dropout_p)
    return x 


def FeatureDropout_3D(x): 
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9) 
    threshold = threshold.view(x.size(0), 1, 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float() 
    x = x.mul(drop_mask)
    return x
     
class Decoder(nn.Module): 
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params  = params
        
        ft_chns = self.params['feature_chns']
        n_class = self.params['class_num']
        # self.feature_scale = self.params['feature_scale']
        # self.is_batchnorm = self.params['is_batchnorm']
        
        padding_size = self.params['dilation'] 
        # self.params['padding_size'] #3,1,6
        dilation = self.params['dilation'] #3,1,6
        dropout_p = self.params['dropout'] #[0.05, 0.1, 0.2, 0.3, 0.5]

        # filters = self.params['feature_chns']
        self.up4 = UpBlock3D(ft_chns[4], ft_chns[3], dropout_p[3], kernel_size=3, 
            padding_size=1, dilation_size=1)
        self.up3 = UpBlock3D(ft_chns[3], ft_chns[2], dropout_p[2], kernel_size=3, 
            padding_size=padding_size, dilation_size=dilation)
        self.up2 = UpBlock3D(ft_chns[2], ft_chns[1], dropout_p[1], kernel_size=3, 
            padding_size=padding_size, dilation_size=dilation)
        self.up1 = UpBlock3D(ft_chns[1], ft_chns[0], dropout_p[0], kernel_size=3, 
            padding_size=padding_size, dilation_size=dilation)
        # self.up_concat4 = Unet_Upblock3d(filters[4], filters[3], self.dropout_p[3], self.is_batchnorm, padding_size = self.padding_size, \
        #     dilation = self.dilation) #UnetUp3_CT: up + cat + conv
        # self.up_concat3 = Unet_Upblock3d(filters[3], filters[2], self.dropout_p[2], self.is_batchnorm, padding_size = self.padding_size, \
        #     dilation = self.dilation)
        # self.up_concat2 = Unet_Upblock3d(filters[2], filters[1], self.dropout_p[1], self.is_batchnorm, padding_size = self.padding_size, \
        #     dilation = self.dilation)
        # self.up_concat1 = Unet_Upblock3d(filters[1], filters[0], self.dropout_p[0], self.is_batchnorm, padding_size = self.padding_size, \
        #     dilation = self.dilation) # filters[1],[0]分别代表in, out,

        self.out_conv  = nn.Conv3d(ft_chns[0], n_class, kernel_size = 1)
        self.out_conv2 = nn.Conv3d(ft_chns[1], n_class, kernel_size = 1)
        self.out_conv3 = nn.Conv3d(ft_chns[2], n_class, kernel_size = 1)
        self.out_conv4 = nn.Conv3d(ft_chns[3], n_class, kernel_size = 1)
        
        # for m in self.modules():
        #     print(m, type(m))
        #     if isinstance(m, nn.Conv3d):
        #         init_weights(m, init_type=init_type)
        #     elif isinstance(m, nn.BatchNorm3d):
        #         init_weights(m, init_type=init_type)
        self.apply(self.init_weight)

    def init_weight(self, m):
        init_type   = self.params['init_type'] #select mode: kaiming, xavier, normal, orthogonal。默认kaiming
        if(isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d)):
            if(init_type == "kaiming"):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                print("this layer is ininialized by kaiming", m)
            elif(init_type == "xavier"):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                print("this layer is ininialized by xavier", m)
            else:
                nn.init.normal_(m.weight, 0.0, 0.02)
                print("this layer is ininialized by normal", m)
            

    def forward(self, x):
        x0, x1, x2, x3, x4 = x
        x3_dec = self.up4(x4, x3)
        x2_dec = self.up3(x3_dec, x2)
        x1_dec = self.up2(x2_dec, x1)
        x0_dec = self.up1(x1_dec, x0)
        output = self.out_conv(x0_dec)
        output_2 = self.out_conv2(x1_dec)
        output_3 = self.out_conv3(x2_dec)
        output_4 = self.out_conv4(x3_dec)

        return output, output_2, output_3, output_4

class TDNet3D(nn.Module): 
    def __init__(self, params):
        super(TDNet3D, self).__init__()
        
        params = self.get_default_parameters(params)
        init_types = params['init_types']
        dilation_rates = params['dilation_rates']
        for p in params:
            print(p, params[p])

        params_decoder = copy.deepcopy(params)
        params_decoder['init_type'] = init_types[0]
        params_decoder['dilation']  = dilation_rates[0]

        params_deaux1 = copy.deepcopy(params)
        params_deaux1['init_type'] = init_types[1]
        params_deaux1['dilation']  = dilation_rates[1]

        params_deaux2 = copy.deepcopy(params)
        params_deaux2['init_type'] = init_types[2]
        params_deaux2['dilation']  = dilation_rates[2]

        self.encoder      = Encoder(params)
        self.main_decoder = Decoder(params_decoder)
        self.aux_decoder1 = Decoder(params_deaux1) 
        self.aux_decoder2 = Decoder(params_deaux2) 

    def get_default_parameters(self, params):
        default_param = { 
                    'feature_chns': [16, 32, 64, 128, 256],
                    'dropout': [0.00, 0.1, 0.2, 0.3, 0.5],
                    'up_mode': 2,
                    'norm_type': 'batch_norm',
                    'init_types': ['kaiming', 'xavier', 'normal'],
                    'dilation_rates': [1,1,1]} 
        for key in default_param:
            params[key] = params.get(key, default_param[key])
        for key in params:
                logging.info("{0:}  = {1:}".format(key, params[key]))
        return params

    def forward(self, x): 
        # pLS论文里提到：
        # We added the dropout layer (ratio=0.5) before each
        # conv-block of the auxiliary decoder to introduce perturbations
        """
        here, we added random dropout before each conv-block of the auxiliary decoder"""
        features = self.encoder(x)
        main_seg, main_embedding1, main_embedding2, main_embedding3 = self.main_decoder(features)

        aux1_feature = [Dropout_random(i) for i in features]
        aux2_feature = [FeatureDropout_3D(i) for i in features]
      
        aux1_seg, aux1_embedding1, aux1_embedding2, aux1_embedding3 = self.aux_decoder1(aux1_feature)      
        aux2_seg, aux2_embedding1, aux2_embedding2, aux2_embedding3 = self.aux_decoder2(aux2_feature)

        if self.training:
            return [main_seg, aux1_seg, aux2_seg], [main_embedding1, aux1_embedding1, aux2_embedding1],\
               [main_embedding2, aux1_embedding2, aux2_embedding2], [main_embedding3, aux1_embedding3, aux2_embedding3]
        else: 
            return main_seg, aux1_seg, aux2_seg
    