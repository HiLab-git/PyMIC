# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pymic.io.image_read_write import *
from pymic.io.nifty_dataset import NiftyDataset
from pymic.io.transform3d import *

if __name__ == "__main__":
    root_dir = '/home/guotai/data/brats/BraTS2018_Training'
    csv_file = '/home/guotai/projects/torch_brats/brats/config/brats18_train_train.csv'
    
    crop1 = CropWithBoundingBox(start = None, output_size = [4, 144, 180, 144])
    norm  = ChannelWiseNormalize(mean = None, std = None, zero_to_random = True)
    labconv = LabelConvert([0, 1, 2, 4], [0, 1, 2, 3])
    crop2 = RandomCrop([128, 128, 128])
    rescale =Rescale([64, 64, 64])
    transform_list = [crop1, norm, labconv, crop2,rescale, ToTensor()]
    transformed_dataset = NiftyDataset(root_dir=root_dir,
                                    csv_file  = csv_file,
                                    modal_num = 4,
                                    transform = transforms.Compose(transform_list))
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)
    # Helper function to show a batch


    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
            sample_batched['label'].size())

        # # observe 4th batch and stop.
        modals = ['flair', 't1ce', 't1', 't2']
        if i_batch == 0:
            image =  sample_batched['image'].numpy()
            label =  sample_batched['label'].numpy()
            for i in range(image.shape[0]):
                for mod in range(4):
                    image_i = image[i][mod]
                    label_i = label[i][0]
                    image_name = "temp/image_{0:}_{1:}.nii.gz".format(i, modals[mod])
                    label_name = "temp/label_{0:}.nii.gz".format(i)
                    save_array_as_nifty_volume(image_i, image_name, reference_name = None)
                    save_array_as_nifty_volume(label_i, label_name, reference_name = None)