# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime
from tensorboardX import SummaryWriter
from pymic.io.image_read_write import *
from pymic.io.nifty_dataset import NiftyDataset
from pymic.io.transform3d import *
from pymic.net.unet3d import UNet3D
from pymic.net.demonet import DemoNet


def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label 
        input_tensor: tensor with shae [B, 1, D, H, W]
        output_tensor: shape [B, num_class, D, H, W]
    """
    tensor_list = []
    for i in range(num_class):
        temp_prob = input_tensor == i*torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim = 1)
    output_tensor = output_tensor.double()

    return output_tensor

def soft_dice_loss(predict, soft_y, num_class, softmax = True):
    soft_y  = soft_y.permute(0, 2, 3, 4, 1)
    soft_y  = torch.reshape(soft_y, (-1, num_class))
    predict = predict.permute(0, 2, 3, 4, 1)
    predict = torch.reshape(predict, (-1, num_class))
    if(softmax):
        predict = nn.Softmax(dim = -1)(predict)
    y_vol = torch.sum(soft_y, dim = 0)
    p_vol = torch.sum(predict, dim = 0)
    intersect = torch.sum(soft_y * predict, dim = 0)
    dice_score = (2.0 * intersect + 1e-5)/ (y_vol + p_vol + 1e-5)
    dice_score = torch.mean(dice_score)
    return 1.0 - dice_score

if __name__ == "__main__":
    root_dir = '/home/guotai/data/brats/BraTS2018_Training'
    train_csv_file = '/home/guotai/projects/torch_brats/brats/config/brats18_train_train.csv'
    valid_csv_file = '/home/guotai/projects/torch_brats/brats/config/brats18_train_valid.csv'
    
    crop1 = CropWithBoundingBox(start = None, output_size = [4, 144, 176, 144])
    scale = Rescale(output_size = [96, 128, 96])
    norm  = ChannelWiseNormalize(mean = None, std = None, zero_to_random = True)
    labconv = LabelConvert([0, 1, 2, 4], [0, 1, 2, 3])
    crop2 = RandomCrop([80, 80, 80])
    transform_list = [crop1, scale,  norm, labconv, crop2]
    train_dataset = NiftyDataset(root_dir=root_dir,
                                csv_file  = train_csv_file,
                                modal_num = 4,
                                with_label= True,
                                transform = transforms.Compose(transform_list))
    valid_dataset = NiftyDataset(root_dir=root_dir,
                                csv_file  = valid_csv_file,
                                modal_num = 4,
                                with_label= True,
                                transform = transforms.Compose(transform_list))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2,
                            shuffle=True, num_workers=8)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=2,
                            shuffle=True, num_workers=8)
    trainIter = iter(trainloader)

    params = {'input_chn_num':4,
              'feature_chn_nums':[4, 16, 32, 64, 128],
              'class_num': 4,
              'acti_func': 'leakyrelu'}
    net = UNet3D(params)
    net.double()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print('device', device)
    net.to(device)  

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    summ_writer = SummaryWriter("brats/model/multi_cls2")
    min_iter  = 10000
    max_iter  = 15000
    valid_gap = 100
    save_iter = 1000
    running_loss = 0
    if(min_iter > 0):
        net.load_state_dict(torch.load("brats/model/multi_cls2_{0:}.pt".format(min_iter)))    
        net.eval()
    for it in range(min_iter, max_iter):
        try:
            data = next(trainIter)
        except StopIteration:
            trainIter = iter(trainloader)
            data = next(trainIter)

        # get the inputs
        inputs, labels = data['image'].double(), data['label']
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
            
        # forward + backward + optimize
        outputs = net(inputs)
        soft_y  = get_soft_label(labels,params['class_num'])
        loss    = soft_dice_loss(outputs, soft_y, params['class_num'])
        loss_value = loss.item()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = running_loss + loss.item()
        if (it % valid_gap == valid_gap - 1):
            train_avg_loss = running_loss / valid_gap
            valid_loss = 0.0
            with torch.no_grad():
                for data in validloader:
                    inputs, labels = data['image'].double(), data['label']
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    soft_y  = get_soft_label(labels,params['class_num'])
                    loss    = soft_dice_loss(outputs, soft_y, params['class_num'])
                    valid_loss = valid_loss + loss.item()
            valid_avg_loss = valid_loss / len(validloader)
            scalers = {'train':train_avg_loss, 'valid': valid_avg_loss}
            summ_writer.add_scalars('loss', scalers, it + 1)
            running_loss = 0.0
            print("{0:} it {1:}, loss {2:}, {3:}".format(
                datetime.now(), it + 1, train_avg_loss, valid_avg_loss))
        if (it % save_iter ==  save_iter - 1):
            torch.save(net.state_dict(), "brats/model/multi_cls2_{0:}.pt".format(it + 1))    
    summ_writer.close()

