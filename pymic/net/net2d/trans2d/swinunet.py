# -*- coding: utf-8 -*-
"""
code adapted from: https://github.com/HuCaoFighting/Swin-Unet

"""
from __future__ import print_function, division

import copy
import numpy as np 
import torch
import torch.nn as nn

from pymic.net.net2d.trans2d.swinunet_sys import SwinTransformerSys

class SwinUNet(nn.Module):
    """
    Implementatin of Swin-UNet.
    
    * Reference: Hu Cao, Yueyue Wang et al:
     Swin-Unet: Unet-Like Pure Transformer for Medical Image Segmentation. 
      `ECCV 2022 Workshops. <https://link.springer.com/chapter/10.1007/978-3-031-25066-8_9>`_

    Note that the input channel can only be 1 or 3, and the input image size should be 224x224.
    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param img_size: (tuple) The input image size, should be [224, 224].
    :param class_num: (int) The class number for segmentation task. 
    """    
    def __init__(self, params):
        super(SwinUNet, self).__init__()
        img_size    = params['img_size']
        if(isinstance(img_size, tuple) or isinstance(img_size, list)):
            img_size = img_size[0]
        self.num_classes = params['class_num']
        self.swin_unet = SwinTransformerSys(img_size = img_size, num_classes=self.num_classes)
        # self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
        #                         patch_size=config.MODEL.SWIN.PATCH_SIZE,
        #                         in_chans=config.MODEL.SWIN.IN_CHANS,
        #                         num_classes=self.num_classes,
        #                         embed_dim=config.MODEL.SWIN.EMBED_DIM,
        #                         depths=config.MODEL.SWIN.DEPTHS,
        #                         num_heads=config.MODEL.SWIN.NUM_HEADS,
        #                         window_size=config.MODEL.SWIN.WINDOW_SIZE,
        #                         mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        #                         qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        #                         qk_scale=config.MODEL.SWIN.QK_SCALE,
        #                         drop_rate=config.MODEL.DROP_RATE,
        #                         drop_path_rate=config.MODEL.DROP_PATH_RATE,
        #                         ape=config.MODEL.SWIN.APE,
        #                         patch_norm=config.MODEL.SWIN.PATCH_NORM,
        #                         use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
          [N, C, D, H, W] = x_shape
          new_shape = [N*D, C, H, W]
          x = torch.transpose(x, 1, 2)
          x = torch.reshape(x, new_shape)

        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)

        if(len(x_shape) == 5):
            new_shape = [N, D] + list(logits.shape)[1:]
            logits = torch.reshape(logits, new_shape)
            logits = torch.transpose(logits, 1, 2)

        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")

        
if __name__ == "__main__":
    params = {'img_size': [224, 224],
              'class_num': 2}
    net = SwinUNet(params)
    net.double()

    x  = np.random.rand(4, 3, 224, 224)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = net(xt)
    print(len(y.size()))
    y = y.detach().numpy()
    print(y.shape)