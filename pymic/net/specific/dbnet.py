# -*- coding: utf-8 -*-
'''`
Implementation of DBNet (dual branch network) used in DMSPS for scribble-supervised segmentation. 
  https://www.sciencedirect.com/science/article/abs/pii/S1361841524001993?dgcid
    @article{han2024dmsps,
    title={DMSPS: Dynamically mixed soft pseudo-label supervision for scribble-supervised medical image segmentation},
    author={Han, Meng and Luo, Xiangde and Xie, Xiangjiang and Liao, Wenjun and Zhang, Shichuan and Song, Tao and Wang, Guotai and Zhang, Shaoting},
    journal={Medical Image Analysis},
    pages={103274},
    year={2024},
    publisher={Elsevier}
}

'''

from __future__ import print_function, division
import torch
import torch.nn as nn
from pymic.util.parse_config import *
from pymic.net.cnn.unet import Encoder, Decoder


def Dropout(x, dim,  p=0.5):
    if dim == 2:
        x = torch.nn.functional.dropout2d(x, p)
    else:
        x = torch.nn.functional.dropout3d(x, p)
    return x

class DBNet(nn.Module): 
    def __init__(self, params):
        super(DBNet, self).__init__()
        self.dim     = params['dimension']
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder  = Decoder(params) 

    # def forward(self, x): 
    #     feature = self.encoder(x)
    #     main_out = self.main_decoder(feature)
    #     aux_feature = [Dropout(i) for i in feature]
    #     aux_out1 = self.aux_decoder1(aux_feature)        
    #     return main_out, aux_out1
    
    def forward(self, x):
        x_shape = list(x.shape)
        if (self.dim == 2 and len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)

        main_feat= self.encoder(x)
        main_out = self.main_decoder(main_feat)
        aux_feat = [Dropout(i, self.dim) for i in main_feat]
        aux_out  = self.aux_decoder(aux_feat)        

        if(self.dim == 2 and len(x_shape) == 5):
            if(isinstance(main_out, (list,tuple))):
                for i in range(len(main_out)):
                    new_shape = [N, D] + list(main_out[i].shape)[1:]
                    main_out[i] = torch.transpose(torch.reshape(main_out[i], new_shape), 1, 2)
                    aux_out[i]  = torch.transpose(torch.reshape(aux_out[i],  new_shape), 1, 2)
            else:
                new_shape = [N, D] + list(main_out.shape)[1:]
                main_out  = torch.transpose(torch.reshape(main_out, new_shape), 1, 2) 
                aux_out   = torch.transpose(torch.reshape(aux_out,  new_shape), 1, 2) 
        return main_out, aux_out