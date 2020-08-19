# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pymic.io.image_read_write import load_image_as_nd_array

class NiftyDataset(Dataset):
    """Dataset for loading images. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors 
    with dimention order [C, H, W] for 2D images"""

    def __init__(self, root_dir, csv_file, modal_num = 1, 
            with_label = False, with_weight = None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with image names.
            modal_num (int): Number of modalities. 
            with_label (bool): Load the data with segmentation ground truth.
            with_weight(bool): Load pixel-wise weight map.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir   = root_dir
        self.csv_items  = pd.read_csv(csv_file)
        self.modal_num  = modal_num
        self.with_label = with_label
        self.with_weight= with_weight
        self.transform  = transform

        if(self.with_label):
            self.label_idx = list(self.csv_items.keys()).index('label')
        if(self.with_weight):
            self.weight_idx = list(self.csv_items.keys()).index('weight')

    def __len__(self):
        return len(self.csv_items)

    def __getlabel__(self, idx):
        label_name = "{0:}/{1:}".format(self.root_dir, 
            self.csv_items.iloc[idx, self.label_idx])
        label = load_image_as_nd_array(label_name)['data_array']
        label = np.asarray(label, np.int32)
        return label

    def __getweight__(self, idx):
        weight_name = "{0:}/{1:}".format(self.root_dir, 
            self.csv_items.iloc[idx, self.weight_idx])
        weight = load_image_as_nd_array(weight_name)['data_array']
        weight = np.asarray(weight, np.float32)
        return weight        

    def __getitem__(self, idx):
        names_list, image_list = [], []
        for i in range (self.modal_num):
            image_name = self.csv_items.iloc[idx, i]
            image_full_name = "{0:}/{1:}".format(self.root_dir, image_name)
            image_dict = load_image_as_nd_array(image_full_name)
            image_data = image_dict['data_array']
            names_list.append(image_name)
            image_list.append(image_data)
        image = np.concatenate(image_list, axis = 0)
        image = np.asarray(image, np.float32)    
        sample = {'image': image, 'names' : names_list[0], 
                 'origin':image_dict['origin'],
                 'spacing': image_dict['spacing'],
                 'direction':image_dict['direction']}
        if (self.with_label):   
            sample['label'] = self.__getlabel__(idx) 
            assert(image.shape[1:] == sample['label'].shape[1:])
        if (self.with_weight):
            sample['weight'] = self.__getweight__(idx) 
            assert(image.shape[1:] == sample['weight'].shape[1:])
        if self.transform:
            sample = self.transform(sample)

        return sample


class ClassificationDataset(NiftyDataset):
    def __init__(self, root_dir, csv_file, modal_num = 1, class_num = 2, 
            with_label = False, transform=None):
        super(ClassificationDataset, self).__init__(root_dir, 
            csv_file, modal_num, with_label, transform)
        self.class_num = class_num
        print("class number for ClassificationDataset", self.class_num)

    def __getlabel__(self, idx):
        label_idx = self.csv_items.iloc[idx, -1]
        label = np.zeros((self.class_num, ))
        label[label_idx] = 1
        return label
