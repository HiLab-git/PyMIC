# -*- coding: utf-8 -*-
"""
An implementation of the Nested U-Net paper:
    Zongwei Zhou, et al.:
    UNet++: A Nested U-Net Architecture for Medical Image Segmentation. 
    MICCAI DLMIA workshop, 2018: 3-11.
Note that there are some modifications from the original paper, such as
the use of dropout and leaky relu here.
"""
import torch
import torch.nn as nn
from pymic.net.net2d.unet2d import *

class NestedUNet2D(nn.Module):
    def __init__(self, params):
        super(NestedUNet2D, self).__init__()
        self.params  = params
        self.in_chns = self.params['in_chns']
        self.filters = self.params['feature_chns'] 
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ConvBlock(self.in_chns, self.filters[0], self.dropout[0])
        self.conv1_0 = ConvBlock(self.filters[0], self.filters[1], self.dropout[1])
        self.conv2_0 = ConvBlock(self.filters[1], self.filters[2], self.dropout[2])
        self.conv3_0 = ConvBlock(self.filters[2], self.filters[3], self.dropout[3])
        self.conv4_0 = ConvBlock(self.filters[3], self.filters[4], self.dropout[4])

        self.conv0_1 = ConvBlock(self.filters[0] + self.filters[1], self.filters[0], 0)
        self.conv1_1 = ConvBlock(self.filters[1] + self.filters[2], self.filters[1], 0)
        self.conv2_1 = ConvBlock(self.filters[2] + self.filters[3], self.filters[2], 0)
        self.conv3_1 = ConvBlock(self.filters[3] + self.filters[4], self.filters[3], 0)

        self.conv0_2 = ConvBlock(self.filters[0] * 2 + self.filters[1], self.filters[0], 0)
        self.conv1_2 = ConvBlock(self.filters[1] * 2 + self.filters[2], self.filters[1], 0)
        self.conv2_2 = ConvBlock(self.filters[2] * 2 + self.filters[3], self.filters[2], 0)

        self.conv0_3 = ConvBlock(self.filters[0] * 3 + self.filters[1], self.filters[0], 0)
        self.conv1_3 = ConvBlock(self.filters[1] * 3 + self.filters[2], self.filters[1], 0)

        self.conv0_4 = ConvBlock(self.filters[0] * 4 + self.filters[1], self.filters[0], 0)

        self.final = nn.Conv2d(self.filters[0], self.n_class, kernel_size=1)

    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
          [N, C, D, H, W] = x_shape
          new_shape = [N*D, C, H, W]
          x = torch.transpose(x, 1, 2)
          x = torch.reshape(x, new_shape)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        if(len(x_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)
        return output


if __name__ == "__main__":
    params = {'in_chns':4,
              'feature_chns':[2, 8, 32, 48, 64],
              'dropout':  [0, 0, 0.3, 0.4, 0.5],
              'class_num': 2}
    Net = NestedUNet2D(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 10, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    print(len(y.size()))
    y = y.detach().numpy()
    print(y.shape)
