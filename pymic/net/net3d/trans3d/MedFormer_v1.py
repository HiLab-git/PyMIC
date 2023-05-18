# -*- coding: utf-8 -*-
from __future__ import print_function, division

import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from pymic.net.net3d.unet3d import Encoder, Decoder

class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()
        hidden_size = params["attention_hidden_size"]
        self.num_attention_heads = params["attention_num_heads"]
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key   = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(params["attention_dropout_rate"])
        self.proj_dropout = Dropout(params["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output

class MLP(nn.Module):
    def __init__(self, params):
        super(MLP, self).__init__()
        hidden_size = params["attention_hidden_size"]
        mlp_dim     = params["attention_mlp_dim"]
        self.fc1 = Linear(hidden_size, mlp_dim)
        self.fc2 = Linear(mlp_dim, hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(params["attention_dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, params):
        super(Block, self).__init__()
        hidden_size = params["attention_hidden_size"]
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = MLP(params)
        self.attn = Attention(params)

    def forward(self, x):
        # convert the tensor shape from [B, C, D, H, W] to [B, DHW, C]
        [B, C, D, H, W] = list(x.shape)
        new_shape = [B, C, D*H*W]
        x = torch.reshape(x, new_shape)
        x = torch.transpose(x, 1, 2)

        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        # convert the result back to [B, C, D, H, W]
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, [B, C, D, H, W])
        return x

class MedFormerV1(nn.Module):
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
    :param deep_supervise: (bool) Using deep supervision for training or not.
    """
    def __init__(self, params):
        super(MedFormerV1, self).__init__()
        self.params   = params
        self.encoder  = Encoder(params)
        self.decoder  = Decoder(params)  
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
              'deep_supervise': True,
              'attention_hidden_size': 128,
              'attention_num_heads': 4,
              'attention_mlp_dim': 256,
              'attention_dropout_rate': 0.2}
    Net = MedFormerV1(params)
    Net = Net.double()

    x  = np.random.rand(1, 4, 96, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    print("output length", len(y))
    for yi in y:
        yi = yi.detach().numpy()
        print(yi.shape)
