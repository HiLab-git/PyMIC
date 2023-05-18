# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np 
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from pymic.net.net3d.unet3d import ConvBlock, Encoder
from pymic.net.net3d.trans3d.MedFormer_v1 import Block
from pymic.net.net3d.trans3d.MedFormer_v2 import SwinTransformerBlock, window_partition

class GLAttLayer(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        # build blocks
        
        self.lcl_att = SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path)
        self.adpool = nn.AdaptiveAvgPool3d([12, 12, 12])

        params = {'attention_hidden_size': dim,
              'attention_num_heads': 4,
              'attention_mlp_dim': dim,
              'attention_dropout_rate': 0.2}
        self.glb_att = Block(params)
        self.conv1x1 = nn.Sequential(
            nn.Conv3d(2*dim, dim, kernel_size=1),
            nn.BatchNorm3d(dim),
            nn.LeakyReLU())

    def forward(self, x):
        [B, C, S, H, W] = list(x.shape)
        # calculate attention mask for SW-MSA
        Sp = int(np.ceil(S / self.window_size)) * self.window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        s_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size * self.window_size)  
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        # for local attention
        xl = x.flatten(2).transpose(1, 2).contiguous()
        xl = self.lcl_att(xl, attn_mask)
        xl = xl.view(-1, S, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        # for global attention
        xg = self.adpool(x)
        xg = self.glb_att(xg)
        xg = interpolate(xg, [S, H, W], mode = 'trilinear')
        out = torch.cat([xl, xg], dim=1)
        out = self.conv1x1(out)
        return out 

class AttUpBlock(nn.Module):
    """
    3D upsampling followed by ConvBlock
    
    :param in_channels1: (int) Channel number of high-level features.
    :param in_channels2: (int) Channel number of low-level features.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    :param trilinear: (bool) Use trilinear for up-sampling (by default).
        If False, deconvolution is used for up-sampling. 
    """
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 trilinear=True, with_att = False, att_params = None):
        super(AttUpBlock, self).__init__()
        self.trilinear = trilinear
        self.with_att  = with_att
        if trilinear:
            self.conv1x1 = nn.Conv3d(in_channels1, in_channels2, kernel_size = 1)
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)
        if(self.with_att):
            input_resolution = att_params['input_resolution']
            num_heads        = att_params['num_heads']
            window_size      = att_params['window_size']
            self.attn  = GLAttLayer(out_channels, input_resolution, num_heads, window_size, 2.0)

    def forward(self, x1, x2):
        if self.trilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x =  self.conv(x) 
        if(self.with_att):
            x = self.attn(x)
        return x


class AttDecoder(nn.Module):
    """
    Decoder of 3D UNet.

    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param in_chns: (int) Input channel number.
    :param feature_chns: (list) Feature channel for each resolution level. 
      The length should be 4 or 5, such as [16, 32, 64, 128, 256].
    :param dropout: (list) The dropout ratio for each resolution level. 
      The length should be the same as that of `feature_chns`.
    :param class_num: (int) The class number for segmentation task. 
    :param trilinear: (bool) Using bilinear for up-sampling or not. 
        If False, deconvolution will be used for up-sampling.
    """
    def __init__(self, params):
        super(AttDecoder, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.dropout   = self.params['dropout']
        self.n_class   = self.params['class_num']
        self.trilinear = self.params.get('trilinear', True)
        self.mul_pred  = self.params['multiscale_pred']
       
        assert(len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

        if(len(self.ft_chns) == 5):
            self.up1 = AttUpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], self.dropout[3], self.trilinear) 
        att_params = {"input_resolution": [24, 24, 24], "num_heads": 4, "window_size": 7}
        self.up2 = AttUpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], self.dropout[2], self.trilinear, True, att_params) 
        att_params = {"input_resolution": [48, 48, 48], "num_heads": 4, "window_size": 7}
        self.up3 = AttUpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], self.dropout[1], self.trilinear, True, att_params) 
        self.up4 = AttUpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], self.dropout[0], self.trilinear) 
        self.out_conv = nn.Conv3d(self.ft_chns[0], self.n_class, kernel_size = 1)
        if(self.mul_pred):
            self.out_conv1 = nn.Conv3d(self.ft_chns[1], self.n_class, kernel_size = 1)
            self.out_conv2 = nn.Conv3d(self.ft_chns[2], self.n_class, kernel_size = 1)
            self.out_conv3 = nn.Conv3d(self.ft_chns[3], self.n_class, kernel_size = 1)

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
        if(self.mul_pred):
            output1 = self.out_conv1(x_d1)
            output2 = self.out_conv2(x_d2)
            output3 = self.out_conv3(x_d3)
            output = [output, output1, output2, output3]
        return output

class MedFormerV3(nn.Module):
    """
    An implementation of the U-Net.
        
    * Reference: Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
      3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. 
      `MICCAI (2) 2016: 424-432. <https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49>`_
    
    Note that there are some modifications from the original paper, such as
    the use of batch normalization, dropout, leaky relu and deep supervision.

    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param in_chns: (int) Input channel number.
    :param feature_chns: (list) Feature channel for each resolution level. 
      The length should be 4 or 5, such as [16, 32, 64, 128, 256].
    :param dropout: (list) The dropout ratio for each resolution level. 
      The length should be the same as that of `feature_chns`.
    :param class_num: (int) The class number for segmentation task. 
    :param trilinear: (bool) Using trilinear for up-sampling or not. 
        If False, deconvolution will be used for up-sampling.
    """
    def __init__(self, params):
        super(MedFormerV3, self).__init__()
        self.params   = params
        self.encoder  = Encoder(params)
        self.decoder  = AttDecoder(params)  
        params["attention_hidden_size"] = params['feature_chns'][-1]
        params["attention_mlp_dim"]     = params['feature_chns'][-1]
        self.attn = Block(params)

    def forward(self, x):
        f = self.encoder(x)
        f[-1] = self.attn(f[-1])
        output = self.decoder(f)
        return output

if __name__ == "__main__":
    params = {'in_chns':4,
              'class_num': 2,
              'feature_chns':[16, 32, 64, 128],
              'dropout' : [0, 0, 0, 0.5],
              'trilinear': True,
              'multiscale_pred': True,
              'attention_num_heads': 4,
              'attention_dropout_rate': 0.2}

    Net = MedFormerV3(params)
    Net = Net.double()

    x  = np.random.rand(2, 4, 96, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    print("output length", len(y))
    for yi in y:
        yi = yi.detach().numpy()
        print(yi.shape)
