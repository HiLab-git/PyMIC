# -*- coding: utf-8 -*-
'''`
Implementation of TDNet (triple branch multi-dilated network) used in PSSeg for scribble-supervised segmentation. 
  https://www.sciencedirect.com/science/article/pii/S0925231226002341
    @article{han2025psseg,
    title={PS-Seg: Learning from partial scribbles for 3D multiple abdominal organ segmentation},
    author={Han, Meng and Ma, Xiaochuan and Luo, Xiangde and Liao, Wenjun and Zhang, Shichuan and Zhang, Shaoting and Wang, Guotai},
    journal={Neurocomputing},
    volume={672},
    pages={132837},
    year={2026},
    publisher={Elsevier}
}

'''
from __future__ import print_function, division

import logging
import copy
import torch
import torch.nn as nn
import numpy as np
import random
from pymic.net.cnn.basic_layer import *
from pymic.net.cnn.unet import Encoder, UpBlock

def Dropout_random(x, dim = 2):
    dropout_p = round(random.uniform(0.2, 0.5), 2)
    if dim == 2:
        x = torch.nn.functional.dropout2d(x, dropout_p)
    else:
        x = torch.nn.functional.dropout3d(x, dropout_p)
    return x


def FeatureDropout(x, dim = 2): 
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9) 
    if(dim == 2):
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    else:
        threshold = threshold.view(x.size(0), 1, 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float() 
    x = x.mul(drop_mask)
    return x
     
class Decoder(nn.Module): 
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params  = params
        self.dim     = self.params['dimension']
        ft_chns = self.params['feature_chns']
        dropout = self.params['dropout'] 
        n_class = self.params['class_num']
        up_mode = self.params.get('up_mode', 2)
        
        padding  = self.params['dilation'] 
        dilation = self.params['dilation'] 
        
        assert(len(ft_chns) == 5)
        self.up4 = UpBlock(self.dim, ft_chns[4], ft_chns[3], ft_chns[3], dropout[3], up_mode, padding = 1, dilation = 1) 
        self.up3 = UpBlock(self.dim, ft_chns[3], ft_chns[2], ft_chns[2], dropout[2], up_mode, padding = padding, dilation = dilation) 
        self.up2 = UpBlock(self.dim, ft_chns[2], ft_chns[1], ft_chns[1], dropout[1], up_mode, padding = padding, dilation = dilation) 
        self.up1 = UpBlock(self.dim, ft_chns[1], ft_chns[0], ft_chns[0], dropout[0], up_mode, padding = padding, dilation = dilation) 
        
        conv_nd = get_conv_class(self.dim)
        self.out_conv  = conv_nd(ft_chns[0], n_class, kernel_size = 1)
        self.out_conv2 = conv_nd(ft_chns[1], n_class, kernel_size = 1)
        self.out_conv3 = conv_nd(ft_chns[2], n_class, kernel_size = 1)
        self.out_conv4 = conv_nd(ft_chns[3], n_class, kernel_size = 1)
        self.stage = 'train'

        self.apply(self.init_weight)

    def init_weight(self, m):
        init_type   = self.params['init_type'] #select mode: kaiming, xavier, normal, orthogonal
        if(isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d)):
            if(init_type == "kaiming"):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif(init_type == "xavier"):
                nn.init.xavier_normal_(m.weight.data, gain=1)
            else:
                nn.init.normal_(m.weight, 0.0, 0.02)
            

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

class TDNet(nn.Module): 
    def __init__(self, params):
        super(TDNet, self).__init__()
        params     = self.get_default_parameters(params)
        self.dim   = params['dimension']
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

    def transpose_decoder_outputs(self, batch_size, slice_num, outputs):
        """
        transpose a 4D tensor to a 5D one. Split the 2D batch size into 3D
        batch size times slice number
        """
        for i in range(len(outputs)):
            new_shape  = [batch_size, slice_num] + list(outputs[i].shape)[1:]
            outputs[i] = torch.transpose(torch.reshape(outputs[i], new_shape), 1, 2) 
        return outputs

    def forward(self, x): 
        x_shape = list(x.shape)
        if (self.dim == 2 and len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)


        features = self.encoder(x)
        main_out, main_feat1, main_feat2, main_feat3 = self.main_decoder(features)

        aux1_feature = [Dropout_random(i, self.dim) for i in features]
        aux2_feature = [FeatureDropout(i, self.dim) for i in features]
        aux1_out, aux1_feat1, aux1_feat2, aux1_feat3 = self.aux_decoder1(aux1_feature)      
        aux2_out, aux2_feat1, aux2_feat2, aux2_feat3 = self.aux_decoder2(aux2_feature)

        if(self.dim == 2 and len(x_shape) == 5):
            main_out, main_feat1, main_feat2, main_feat3 = self.transpose_decoder_outputs(N, D,
                [main_out, main_feat1, main_feat2, main_feat3]
            )
            aux1_out, aux1_feat1, aux1_feat2, aux1_feat3 = self.transpose_decoder_outputs(N, D,
                [aux1_out, aux1_feat1, aux1_feat2, aux1_feat3]
            )
            aux2_out, aux2_feat1, aux2_feat2, aux2_feat3 = self.transpose_decoder_outputs(N, D,
                [aux2_out, aux2_feat1, aux2_feat2, aux2_feat3]
            )

        if self.training:
            return [main_out, aux1_out, aux2_out], [main_feat1, aux1_feat1, aux2_feat1],\
               [main_feat2, aux1_feat2, aux2_feat2], [main_feat3, aux1_feat3, aux2_feat3]
        else: 
            return main_out, aux1_out, aux2_out
    