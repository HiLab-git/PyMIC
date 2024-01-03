# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np 
import torch
import torch.nn as nn
from torch.nn import functional as F
from pymic.net.net2d.canet_module import *


def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, groups=group, bias=bias)


class SE_Conv_Block(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(SE_Conv_Block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes / 2))
        self.fc2 = nn.Linear(in_features=round(planes / 2), out_features=planes * 2)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        avg_att = out
        out = out * original_out
        # For global maximum pool
        out1 = F.adaptive_max_pool2d(out1, (1,1))
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)
        max_att = out1
        out1 = out1 * original_out

        att_weight = avg_att + max_att
        out += out1
        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out, att_weight

# # CBAM Convolutional block attention module
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # scalecoe = F.sigmoid(channel_att_sum)
        # print("channel att_sum", channel_att_sum.shape)
        # channel_att_sum = channel_att_sum.reshape(channel_att_sum.shape[0], 4, 4)
        # avg_weight = torch.mean(channel_att_sum, dim=2).unsqueeze(2)
        # avg_weight = avg_weight.expand(channel_att_sum.shape[0], 4, 4).reshape(channel_att_sum.shape[0], 16)
        # scale = F.sigmoid(avg_weight).unsqueeze(2).unsqueeze(3).expand_as(x)
        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale, scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)    # broadcasting
        return x * scale, scale

class SpatialAtten(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=1):
        super(SpatialAtten, self).__init__()
        self.conv1 = BasicConv(in_size, out_size, kernel_size, stride=stride,
                               padding=(kernel_size-1) // 2, relu=True)
        self.conv2 = BasicConv(out_size, in_size, kernel_size=1, stride=stride,
                               padding=0, relu=True, bn=False)

    def forward(self, x):
        residual = x
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        spatial_att = F.sigmoid(x_out)
        # .unsqueeze(4).permute(0, 1, 4, 2, 3)
        # spatial_att = spatial_att.expand(spatial_att.shape[0], 4, 4, spatial_att.shape[3], spatial_att.shape[4]).reshape(
                                        # spatial_att.shape[0], 16, spatial_att.shape[3], spatial_att.shape[4])
        x_out = residual * spatial_att

        x_out += residual

        return x_out, spatial_att

class Scale_atten_block(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(Scale_atten_block, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialAtten(gate_channels, gate_channels //reduction_ratio)

    def forward(self, x):
        x_out, ca_atten = self.ChannelGate(x)
        if not self.no_spatial:
            x_out, sa_atten = self.SpatialGate(x_out)

        return x_out, ca_atten, sa_atten


class scale_atten_convblock(nn.Module):
    def __init__(self, in_size, out_size, stride=1, downsample=None, use_cbam=True, no_spatial=False, drop_out=False):
        super(scale_atten_convblock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.no_spatial = no_spatial
        self.dropout = drop_out

        self.relu = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(in_size, out_size)
        self.bn3 = nn.BatchNorm2d(out_size)

        if use_cbam:
            self.cbam = Scale_atten_block(in_size, reduction_ratio=4, no_spatial=self.no_spatial)  # out_size
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out, scale_c_atten, scale_s_atten = self.cbam(x)

        out += residual
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out
    
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample_factor=4, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation', 'concat_proper', 'concat_proper_down']

        # print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample_factor = sub_sample_factor if isinstance(sub_sample_factor, list) else [sub_sample_factor]

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None

        if mode in ['embedded_gaussian', 'dot_product', 'concatenation', 'concat_proper', 'concat_proper_down']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode in ['concatenation']:
                self.wf_phi = nn.Linear(self.inter_channels, 1, bias=False)
                self.wf_theta = nn.Linear(self.inter_channels, 1, bias=False)
            elif mode in ['concat_proper', 'concat_proper_down']:
                self.psi = nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1,
                                     padding=0, bias=True)

        if mode == 'embedded_gaussian':
            self.operation_function = self._embedded_gaussian
        elif mode == 'dot_product':
            self.operation_function = self._dot_product
        elif mode == 'gaussian':
            self.operation_function = self._gaussian
        elif mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concat_proper':
            self.operation_function = self._concatenation_proper
        elif mode == 'concat_proper_down':
            self.operation_function = self._concatenation_proper_down
        else:
            raise NotImplementedError('Unknown operation function.')

        if any(ss > 1 for ss in self.sub_sample_factor):
            self.g = nn.Sequential(self.g, max_pool(kernel_size=sub_sample_factor))
            if self.phi is None:
                self.phi = max_pool(kernel_size=sub_sample_factor)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=sub_sample_factor))
            if mode == 'concat_proper_down':
                self.theta = nn.Sequential(self.theta, max_pool(kernel_size=sub_sample_factor))

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size = x.size(0)

        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample_factor > 1:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _dot_product(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation(self, x):
        batch_size = x.size(0)

        # g=>(b, c, t, h, w)->(b, 0.5c, thw/s**2)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw/s**2, 0.5c)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)

        # theta => (b, thw, 0.5c) -> (b, thw, 1) -> (b, 1, thw) -> (expand) (b, thw/s**2, thw)
        # phi => (b, thw/s**2, 0.5c) -> (b, thw/s**2, 1) -> (expand) (b, thw/s**2, thw)
        # f=> RELU[(b, thw/s**2, thw) + (b, thw/s**2, thw)] = (b, thw/s**2, thw)
        f = self.wf_theta(theta_x).permute(0, 2, 1).repeat(1, phi_x.size(1), 1) + \
            self.wf_phi(phi_x).repeat(1, 1, theta_x.size(1))
        f = F.relu(f, inplace=True)

        # Normalise the relations
        N = f.size(-1)
        f_div_c = f / N

        # g(x_j) * f(x_j, x_i)
        # (b, 0.5c, thw/s**2) * (b, thw/s**2, thw) -> (b, 0.5c, thw)
        y = torch.matmul(g_x, f_div_c)
        y = y.contiguous().view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation_proper(self, x):
        batch_size = x.size(0)

        # g=>(b, c, t, h, w)->(b, 0.5c, thw/s**2)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw/s**2)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # theta => (b, 0.5c, thw) -> (expand) (b, 0.5c, thw/s**2, thw)
        # phi => (b, 0.5c, thw/s**2) ->  (expand) (b, 0.5c, thw/s**2, thw)
        # f=> RELU[(b, 0.5c, thw/s**2, thw) + (b, 0.5c, thw/s**2, thw)] = (b, 0.5c, thw/s**2, thw)
        f = theta_x.unsqueeze(dim=2).repeat(1,1,phi_x.size(2),1) + \
            phi_x.unsqueeze(dim=3).repeat(1,1,1,theta_x.size(2))
        f = F.relu(f, inplace=True)

        # psi -> W_psi^t * f -> (b, 1, thw/s**2, thw) -> (b, thw/s**2, thw)
        f = torch.squeeze(self.psi(f), dim=1)

        # Normalise the relations
        f_div_c = F.softmax(f, dim=1)

        # g(x_j) * f(x_j, x_i)
        # (b, 0.5c, thw/s**2) * (b, thw/s**2, thw) -> (b, 0.5c, thw)
        y = torch.matmul(g_x, f_div_c)
        y = y.contiguous().view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation_proper_down(self, x):
        batch_size = x.size(0)

        # g=>(b, c, t, h, w)->(b, 0.5c, thw/s**2)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw/s**2)
        theta_x = self.theta(x)
        downsampled_size = theta_x.size()
        theta_x = theta_x.view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # theta => (b, 0.5c, thw) -> (expand) (b, 0.5c, thw/s**2, thw)
        # phi => (b, 0.5, thw/s**2) ->  (expand) (b, 0.5c, thw/s**2, thw)
        # f=> RELU[(b, 0.5c, thw/s**2, thw) + (b, 0.5c, thw/s**2, thw)] = (b, 0.5c, thw/s**2, thw)
        f = theta_x.unsqueeze(dim=2).repeat(1,1,phi_x.size(2),1) + \
            phi_x.unsqueeze(dim=3).repeat(1,1,1,theta_x.size(2))
        f = F.relu(f, inplace=True)

        # psi -> W_psi^t * f -> (b, 0.5c, thw/s**2, thw) -> (b, 1, thw/s**2, thw) -> (b, thw/s**2, thw)
        f = torch.squeeze(self.psi(f), dim=1)

        # Normalise the relations
        f_div_c = F.softmax(f, dim=1)

        # g(x_j) * f(x_j, x_i)
        # (b, 0.5c, thw/s**2) * (b, thw/s**2, thw) -> (b, 0.5c, thw)
        y = torch.matmul(g_x, f_div_c)
        y = y.contiguous().view(batch_size, self.inter_channels, *downsampled_size[2:])

        # upsample the final featuremaps # (b,0.5c,t/s1,h/s2,w/s3)
        y = F.upsample(y, size=x.size()[2:], mode='trilinear')

        # attention block output
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample_factor=2, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample_factor=sub_sample_factor,
                                              bn_layer=bn_layer)

    
class CANet(nn.Module):
    """
    Implementation of CANet (Comprehensive Attention Network) for image segmentation.
    
    * Reference: R. Gu et al. `CA-Net: Comprehensive Attention Convolutional Neural Networks 
        for Explainable Medical Image Segmentation <https://ieeexplore.ieee.org/document/9246575>`_. 
      IEEE Transactions on Medical Imaging, 40(2),2021:699-711. 
    
    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param in_chns: (int) Input channel number.
    :param feature_chns: (list) Feature channel for each resolution level. 
      The length should be 5, such as [16, 32, 64, 128, 256].
    :param dropout: (list) The dropout ratio for each resolution level. 
      The length should be the same as that of `feature_chns`.
    :param class_num: (int) The class number for segmentation task. 
    :param bilinear: (bool) Using bilinear for up-sampling or not. 
        If False, deconvolution will be used for up-sampling.
    """
    def __init__(self,  params): #args, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                #  nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super(CANet, self).__init__()
        self.in_channels = params['in_chns']
        self.num_classes = params['class_num']
        self.is_deconv   = params.get('is_deconv', True)
        self.is_batchnorm  = params.get('is_batchnorm', True)
        self.feature_scale = params.get('feature_scale', 4)
        nonlocal_mode      = 'concatenation'
        attention_dsample  = (1, 1)

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # attention blocks
        # self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[1],
        #                                             inter_channels=filters[0])
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[4], inter_channels=filters[4] // 4)

        # upsampling
        self.up_concat4 = UpCat(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UpCat(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UpCat(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UpCat(filters[1], filters[0], self.is_deconv)
        self.up4 = SE_Conv_Block(filters[4], filters[3], drop_out=True)
        self.up3 = SE_Conv_Block(filters[3], filters[2])
        self.up2 = SE_Conv_Block(filters[2], filters[1])
        self.up1 = SE_Conv_Block(filters[1], filters[0])

        # For deep supervision, project the multi-scale feature maps to the same number of channels
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=filters[0]//2, kernel_size=1)
        self.dsv2 = nn.Conv2d(in_channels=filters[1], out_channels=filters[0]//2, kernel_size=1)
        self.dsv3 = nn.Conv2d(in_channels=filters[2], out_channels=filters[0]//2, kernel_size=1)
        self.dsv4 = nn.Conv2d(in_channels=filters[3], out_channels=filters[0]//2, kernel_size=1)

        self.scale_att = scale_atten_convblock(in_size=filters[0]//2 * 4, out_size=filters[0])
        self.final = nn.Conv2d(filters[0], self.num_classes, kernel_size=1)

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        up4 = self.up_concat4(conv4, center)
        g_conv4 = self.nonlocal4_2(up4)

        up4, att_weight4 = self.up4(g_conv4)
        g_conv3, att3 = self.attentionblock3(conv3, up4)

        # atten3_map = att3.cpu().detach().numpy().astype(np.float)
        # atten3_map = ndimage.interpolation.zoom(atten3_map, [1.0, 1.0, 224 / atten3_map.shape[2],
        #                                                      300 / atten3_map.shape[3]], order=0)

        up3 = self.up_concat3(g_conv3, up4)
        up3, att_weight3 = self.up3(up3)
        g_conv2, att2 = self.attentionblock2(conv2, up3)

        up2 = self.up_concat2(g_conv2, up3)
        up2, att_weight2 = self.up2(up2)
   
        up1 = self.up_concat1(conv1, up2)
        up1, att_weight1 = self.up1(up1)

        # Deep Supervision
        dsv1 = self.dsv1(up1)
        dsv2 = F.interpolate(self.dsv2(up2), dsv1.shape[2:], mode = 'bilinear')
        dsv3 = F.interpolate(self.dsv3(up3), dsv1.shape[2:], mode = 'bilinear')
        dsv4 = F.interpolate(self.dsv4(up4), dsv1.shape[2:], mode = 'bilinear')
        
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        out = self.scale_att(dsv_cat)

        out = self.final(out)

        return out
      
if __name__ == "__main__":
    params = {'in_chns':3,
              'class_num':2}
    Net = CANet(params)
    Net = Net.double()

    x  = np.random.rand(4, 3, 224, 224)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    print(len(y.size()))
    y = y.detach().numpy()
    print(y.shape)
