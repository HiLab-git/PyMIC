
import torch
import numpy as np
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from copy import deepcopy
from torch import nn
from pymic.net.net3d.trans3d.HiFormer_v1 import BasicLayer

class ConvBlock(nn.Module):
    """
    2D or 3D convolutional block
    
    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dim: (int) Should be 2 or 3, for 2D and 3D convolution, respectively.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dim = 2, dropout_p = 0.0):
        super(ConvBlock, self).__init__()
        assert(dim == 2 or dim == 3)
        if(dim == 2):
            kernel_size = [1, 3, 3]
            padding     = [0, 1, 1]
        else:
            kernel_size = 3 
            padding     = 1

        self.conv_conv = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.PReLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
            nn.Dropout(dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        return self.conv_conv(x) 


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, dim = 2, first_layer = False):
        super(DownSample, self).__init__()
        assert(dim == 2 or dim == 3)
        if(dim == 2):
            kernel_size = [1, 3, 3]
            stride      = [1, 2, 2]
            padding     = [0, 1, 1]
        else:
            kernel_size = 3 
            stride      = 2
            padding     = 1

        if(first_layer):
            self.down = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                padding=padding, stride = stride)
        else:
            self.down = nn.Sequential(
                nn.BatchNorm3d(in_channels),
                nn.PReLU(),
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                    padding=padding, stride = stride),
            )

    def forward(self, x):
        return self.down(x)



class ConvTransBlock_backup(nn.Module):
    def __init__(self,
                 input_resolution= [32, 32, 32],
                 chns=96,
                 depth=2,
                 num_head=4,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 ):
        super().__init__()
        self.conv  = ConvBlock(chns, chns, dim = 3, dropout_p = drop_rate)
        self.trans = BasicLayer(
                dim= chns,
                input_resolution= input_resolution,
                depth=depth,
                num_heads=num_head,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                downsample= None
                )
        self.norm_layer = nn.LayerNorm(chns)
        self.pos_drop = nn.Dropout(p=drop_rate)
 
    def forward(self, x):
        """Forward function."""
        x1 = self.conv(x) 
        C, Ws, Wh, Ww = x.size(1), x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.pos_drop(x)
        x2, S, H, W, x, Ws, Wh, Ww = self.trans(x, Ws, Wh, Ww)
        # x2 = self.norm_layer(x2)
        x2 = x2.view(-1, S, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        return x1 + x2

# only using the conv block
class ConvTransBlock(nn.Module):
    def __init__(self,
                 input_resolution= [32, 32, 32],
                 chns=96,
                 depth=2,
                 num_head=4,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 ):
        super().__init__()
        self.conv  = ConvBlock(chns, chns, dim = 3, dropout_p = drop_rate)
        # self.trans = BasicLayer(
        #         dim= chns,
        #         input_resolution= input_resolution,
        #         depth=depth,
        #         num_heads=num_head,
        #         window_size=window_size,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         qk_scale=qk_scale,
        #         drop=drop_rate,
        #         attn_drop=attn_drop_rate,
        #         drop_path=drop_path_rate,
        #         norm_layer=norm_layer,
        #         downsample= None
        #         )
        # self.norm_layer = nn.LayerNorm(chns)
        # self.pos_drop = nn.Dropout(p=drop_rate)
 
    def forward(self, x):
        """Forward function."""
        x1 = self.conv(x) 
        return x1
        # C, Ws, Wh, Ww = x.size(1), x.size(2), x.size(3), x.size(4)
        # x = x.flatten(2).transpose(1, 2).contiguous()
        # x = self.pos_drop(x)
        # x2, S, H, W, x, Ws, Wh, Ww = self.trans(x, Ws, Wh, Ww)
        # # x2 = self.norm_layer(x2)
        # x2 = x2.view(-1, S, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        # return x1 + x2
    
class UpCatBlock(nn.Module):
    """
    3D upsampling followed by ConvBlock
    
    :param in_channels1: (int) Channel number of high-level features.
    :param in_channels2: (int) Channel number of low-level features.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    :param trilinear: (bool) Use trilinear for up-sampling (by default).
        If False, deconvolution is used for up-sampling. 
    """
    def __init__(self, chns_l, chns_h, up_dim = 3, conv_dim = 3):
        super(UpCatBlock, self).__init__()
        assert(up_dim == 2 or up_dim == 3)
        if(up_dim == 2):
            kernel_size, stride = [1, 2, 2], [1, 2, 2]
        else:
            kernel_size, stride = 2, 2
        self.up = nn.ConvTranspose3d(chns_h, chns_l, 
            kernel_size = kernel_size, stride=stride)
        
        if(conv_dim == 2):
            kernel_size, padding = [1, 3, 3], [0, 1, 1]
        else:
            kernel_size, padding = 3, 1
        self.conv = nn.Sequential(
            nn.BatchNorm3d(chns_l*2),
            nn.PReLU(),
            nn.Conv3d(chns_l*2, chns_l, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x_l, x_h):
        # print("input shapes", x1.shape, x2.shape)
        # print("after upsample", x1.shape)
        y = torch.cat([x_l, self.up(x_h)], dim=1)
        return self.conv(y)

class Encoder(nn.Module):
    def __init__(self,
                 in_chns   = 1 ,
                 ft_chns   = [48, 192, 384, 768],
                 input_size= [32, 128, 128],
                 down_dims = [2, 2, 3, 3],
                 conv_dims = [2, 3, 3, 3], 
                 dropout   = [0, 0.2, 0.2, 0.2],
                 depths    = [2, 2, 2],
                 num_heads = [4, 8, 16],
                 window_sizes = [6, 6, 6],
                 high_res  = False,
                 ):
        super().__init__()
        self.high_res = high_res

        self.down1 = DownSample(in_chns, ft_chns[0], down_dims[0], first_layer=True)
        self.down2 = DownSample(ft_chns[0], ft_chns[1], down_dims[1])
        self.down3 = DownSample(ft_chns[1], ft_chns[2], down_dims[2])
        self.down4 = DownSample(ft_chns[2], ft_chns[3], down_dims[3])

        if(high_res):
            self.conv0 = ConvBlock(in_chns, ft_chns[0] // 2, 3, 0)
        self.conv1 = ConvBlock(ft_chns[0], ft_chns[0], conv_dims[0], dropout[0])
        self.conv2 = ConvBlock(ft_chns[1], ft_chns[1], conv_dims[1], dropout[1])

        down_scales = []
        for i in range(4):
            down_scale = [1, 2, 2] if down_dims[i] == 2 else [2, 2, 2]
            down_scales.append(down_scale)

        r_t2 = [input_size[i] // down_scales[0][i] // down_scales[1][i] for i in range(3)]
        r_t3 = [r_t2[i] // down_scales[2][i] for i in range(3)]
        r_t4 = [r_t3[i] // down_scales[3][i] for i in range(3)]

        self.conv_t2 = ConvTransBlock(chns = ft_chns[1], 
                            input_resolution = r_t2,
                            window_size  = window_sizes[0],
                            depth        = depths[0],
                            num_head     = num_heads[0],
                            drop_rate    = dropout[1],
                            attn_drop_rate=dropout[1]
        )
        self.conv_t3 = ConvTransBlock(chns = ft_chns[2], 
                            input_resolution = r_t3,
                            window_size  = window_sizes[1],
                            depth        = depths[1],
                            num_head     = num_heads[1],
                            drop_rate    = dropout[2],
                            attn_drop_rate=dropout[2]
        )
        self.conv_t4 = ConvTransBlock(chns = ft_chns[3], 
                            input_resolution = r_t4,
                            window_size  = window_sizes[2],
                            depth        = depths[2],
                            num_head     = num_heads[2],
                            drop_rate    = dropout[3],
                            attn_drop_rate=dropout[3]
        )

        
        
    def forward(self, x):
        """Forward function."""
        if(self.high_res):
            x0 = self.conv0(x)
        x1 = self.conv1(self.down1(x))
        x2 = self.conv2(self.down2(x1))
        x2 = self.conv_t2(x2)
        x3 = self.conv_t3(self.down3(x2))
        x4 = self.conv_t4(self.down4(x3))
        if(self.high_res):
            return x0, x1, x2, x3, x4
        else:
            return x1, x2, x3, x4

class Decoder(nn.Module):
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
    :param multiscale_pred: (bool) Get multi-scale prediction.
    """
    def __init__(self, 
                ft_chns    = [48, 192, 384, 768],
                input_size = [32, 128, 128],
                down_dims  = [2, 2, 3, 3],
                conv_dims  = [2, 3, 3, 3], 
                dropout    = [0, 0, 0.2, 0.2],
                depths    = [2, 2, 2],
                num_heads = [4, 8, 16],
                window_sizes = [6, 6, 6],
                high_res  = False, 
                class_num  = 2, 
                multiscale_pred = False
                ):
        super(Decoder, self).__init__()
        self.high_res = high_res
        if(self.high_res):
            self.up0 = UpCatBlock(ft_chns[0] // 2, ft_chns[0], down_dims[0], 3) 
            self.conv0 =  ConvBlock(ft_chns[0] // 2, ft_chns[0] // 2, 3, 0)
        self.up1 = UpCatBlock(ft_chns[0], ft_chns[1], down_dims[1], conv_dims[0]) 
        self.up2 = UpCatBlock(ft_chns[1], ft_chns[2], down_dims[2], conv_dims[1])
        self.up3 = UpCatBlock(ft_chns[2], ft_chns[3], down_dims[3], conv_dims[2]) 

        down_scales = []
        for i in range(4):
            down_scale = [1, 2, 2] if down_dims[i] == 2 else [2, 2, 2]
            down_scales.append(down_scale)

        r_t2 = [input_size[i] // down_scales[0][i] // down_scales[1][i] for i in range(3)]
        r_t3 = [r_t2[i] // down_scales[2][i] for i in range(3)]

        self.conv1 = ConvBlock(ft_chns[0], ft_chns[0], conv_dims[0], dropout[0])
        self.conv2 = ConvTransBlock(chns = ft_chns[1], 
                            input_resolution = r_t2,
                            window_size  = window_sizes[0],
                            depth        = depths[0],
                            num_head     = num_heads[0],
                            drop_rate    = dropout[1],
                            attn_drop_rate=dropout[1]
        )
        self.conv3 = ConvTransBlock(chns = ft_chns[2], 
                    input_resolution = r_t3,
                    window_size  = window_sizes[1],
                    depth        = depths[1],
                    num_head     = num_heads[1],
                    drop_rate    = dropout[2],
                    attn_drop_rate=dropout[2]
        )

        kernel_size, stride = 2, 2 
        if down_dims[0] == 2:
            kernel_size, stride = [1, 2, 2], [1, 2, 2]
        if(self.high_res):
            self.out_conv0 = nn.Conv3d(ft_chns[0] // 2, class_num, 
                kernel_size = [1, 3, 3], padding = [0, 1, 1])
        else:
            self.out_conv0 = nn.ConvTranspose3d(ft_chns[0], class_num, 
                kernel_size = kernel_size, stride= stride)
                
        self.mul_pred = multiscale_pred
        if(self.mul_pred):
            self.out_conv1 = nn.Conv3d(ft_chns[0], class_num, kernel_size = 1)
            self.out_conv2 = nn.Conv3d(ft_chns[1], class_num, kernel_size = 1)
            self.out_conv3 = nn.Conv3d(ft_chns[2], class_num, kernel_size = 1)

    def forward(self, x):
        if(self.high_res):
            x0, x1, x2, x3, x4 = x 
        else:
            x1, x2, x3, x4 = x 
        x_d3 = self.conv3(self.up3(x3, x4))
        x_d2 = self.conv2(self.up2(x2, x_d3))
        x_d1 = self.conv1(self.up1(x1, x_d2))
        if(self.high_res):
            x_d0 = self.conv0(self.up0(x0, x_d1))
            output = self.out_conv0(x_d0)
        else:
            output = self.out_conv0(x_d1)
        if(self.mul_pred):
            output1 = self.out_conv1(x_d1)
            output2 = self.out_conv2(x_d2)
            output3 = self.out_conv3(x_d3)
            output = [output, output1, output2, output3]
        return output

class HiFormer_v3(nn.Module):
    def __init__(self, params):
        """
        replace the embedding layer with convolutional blocks
        """
        super(HiFormer_v3, self).__init__()       
        in_chns     = params["in_chns"]
        class_num   = params["class_num"]
        input_size  = params["input_size"]
        ft_chns     = params.get("feature_chns", [48, 192, 384, 764])
        down_dims   = params.get("down_dims", [2, 2, 3, 3])
        conv_dims   = params.get("conv_dims", [2, 3, 3, 3])
        dropout     = params.get('dropout', [0, 0.2, 0.2, 0.2])
        high_res    = params.get("high_res", False)
        depths      = params.get("depths", [2, 2, 2]) 
        num_heads   = params.get("num_heads", [4, 8, 16])
        window_sizes= params.get("window_sizes", [6, 6, 6])
        multiscale_pred = params.get("multiscale_pred", False)
        
        self.encoder = Encoder(in_chns,
                        ft_chns    = ft_chns,
                        input_size = input_size,
                        down_dims  = down_dims,
                        conv_dims  = conv_dims, 
                        dropout    = dropout,
                        depths     = depths,
                        num_heads  = num_heads,
                        window_sizes= window_sizes,
                        high_res = high_res)
        
        self.decoder = Decoder(ft_chns = ft_chns,
                input_size = input_size,
                down_dims  = down_dims,
                conv_dims  = conv_dims, 
                dropout    = dropout,
                depths     = depths,
                num_heads  = num_heads,
                window_sizes= window_sizes,
                high_res = high_res,
                class_num  = class_num, 
                multiscale_pred = multiscale_pred
                )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 
        
        
if __name__ == "__main__":
    params = {"input_size": [64, 96, 96],
                "in_chns": 1,
                "down_dims": [3, 3, 3, 3],
                "conv_dims": [3, 3, 3, 3],
                "feature_chns": [96, 192, 384, 768],
                "high_res": True,
                "class_num": 5,
                "multiscale_pred": True}
    Net = HiFormer_v3(params)
    Net = Net.double()

    x  = np.random.rand(1, 1, 64, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    if(params['multiscale_pred']):
        for yi in y:
            print(yi.shape)
    else:
        print(y.shape)
   

   
