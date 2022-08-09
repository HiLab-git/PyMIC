# -*- coding: utf-8 -*-
"""
An modification the U-Net with auxiliary decoders according to 
the CCT paper:
    Yassine Ouali, Celine Hudelot and Myriam Tami:
    Semi-Supervised Semantic Segmentation With Cross-Consistency Training. 
    CVPR 2020.
    https://arxiv.org/abs/2003.09005  
Code adapted from: https://github.com/yassouali/CCT
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torch.distributions.uniform import Uniform
from pymic.net.net2d.unet2d import ConvBlock, DownBlock, UpBlock

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.dropout   = self.params['dropout']
        assert(len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

        self.in_conv= ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        if(len(self.ft_chns) == 5):
            self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        output = [x0, x1, x2, x3]
        if(len(self.ft_chns) == 5):
          x4 = self.down4(x3)
          output.append(x4)
        return output

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.dropout   = self.params['dropout']
        self.n_class   = self.params['class_num']
        self.bilinear  = self.params['bilinear']

        assert(len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

        if(len(self.ft_chns) == 5):
            self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], self.dropout[3], self.bilinear) 
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], self.dropout[2], self.bilinear) 
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], self.dropout[1], self.bilinear) 
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], self.dropout[0], self.bilinear) 
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size = 1)

    def forward(self, x):
        if(len(self.ft_chns) == 5):
            assert(len(x) == 5)
            x0, x1, x2, x3, x4 = x 
            x_d3 = self.up1(x4, x3)
        else:
            assert(len(x) == 4)
            x0, x1, x2, x3 = x 
            x_d3 = x3
        x_d2 = self.up2(x_d3, x2)
        x_d1 = self.up3(x_d2, x1)
        x_d0 = self.up4(x_d1, x0)
        output = self.out_conv(x_d0)
        return output

def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d



def get_r_adv(x_list, decoder, it=1, xi=1e-1, eps=10.0):
    """
    Virtual Adversarial Training according to
    https://arxiv.org/abs/1704.03976
    """
    x_detached = [item.detach() for item in x_list]
    xe_detached = x_detached[-1]
    with torch.no_grad():
        pred = F.softmax(decoder(x_detached), dim=1)

    d = torch.rand(x_list[-1].shape).sub(0.5).to(x_list[-1].device)
    d = _l2_normalize(d)

    for _ in range(it):
        d.requires_grad_()
        x_detached[-1] = xe_detached + xi * d
        pred_hat = decoder(x_detached)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder.zero_grad()

    r_adv = d * eps
    return x_list[-1] + r_adv
    

class AuxiliaryDecoder(nn.Module):
    def __init__(self, params, aux_type):
        super(AuxiliaryDecoder, self).__init__()
        self.params   = params
        self.decoder  = Decoder(params)
        self.aux_type = aux_type
        uniform_range = params.get("Uniform_range".lower(), 0.3)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_drop(self, x):
        attention  = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        if(self.aux_type == "DropOut"):
            pass
        elif(self.aux_type == "FeatureDrop"):
            x[-1] = self.feature_drop(x[-1])
        elif(self.aux_type == "FeatureNoise"):
            x[-1] = self.feature_based_noise(x[-1])
        elif(self.aux_type == "VAT"):
            it = self.params.get("VAT_it".lower(), 2)
            xi = self.params.get("VAT_xi".lower(), 1e-6)
            eps= self.params.get("VAT_eps".lower(), 2.0)
            x[-1] = get_r_adv(x, self.decoder, it, xi, eps)
        else:
            raise ValueError("Undefined auxiliary decoder type {0:}".format(self.aux_type))
 
        output = self.decoder(x)
        return output


class UNet2D_CCT(nn.Module):
    def __init__(self, params):
        super(UNet2D_CCT, self).__init__()
        self.params    = params
        self.encoder   = Encoder(params)
        self.decoder   = Decoder(params)
        aux_names = params.get("CCT_aux_decoders".lower(), None)
        if aux_names is None:
            aux_names = ["DropOut", "FeatureDrop", "FeatureNoise", "VAT"]
        aux_decoders = []
        for aux_name in aux_names:
            aux_decoders.append(AuxiliaryDecoder(params, aux_name))
        self.aux_decoders = nn.ModuleList(aux_decoders)
        

    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
          [N, C, D, H, W] = x_shape
          new_shape = [N*D, C, H, W]
          x = torch.transpose(x, 1, 2)
          x = torch.reshape(x, new_shape)

        f = self.encoder(x)
        output = self.decoder(f)
        if(len(x_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)

        if(self.training):
            aux_outputs = [aux_d(f) for aux_d in self.aux_decoders]
            if(len(x_shape) == 5):
                for i in range(len(aux_outputs)):
                    aux_outi = torch.reshape(aux_outputs[i], new_shape)
                    aux_outputs[i] = torch.transpose(aux_outi, 1, 2)
            return output, aux_outputs
        else:
            return output