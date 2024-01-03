# -*- coding: utf-8 -*-
from __future__ import print_function, division

import logging
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from pymic import TaskType
from pymic.io.image_read_write import load_image_as_nd_array

class NiftyDataset(Dataset):
    """
    Dataset for loading images for segmentation. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors 
    with dimention order [C, H, W] for 2D images.

    :param root_dir: (str) Directory with all the images. 
    :param csv_file: (str) Path to the csv file with image names.
    :param modal_num: (int) Number of modalities. 
    :param with_label: (bool) Load the data with segmentation ground truth or not.
    :param transform:  (list) List of transforms to be applied on a sample.
        The built-in transforms can listed in :mod:`pymic.transform.trans_dict`.
    """
    def __init__(self, root_dir, csv_file, modal_num = 1, 
            with_label = False, transform=None, task = TaskType.SEGMENTATION):
        self.root_dir   = root_dir
        self.csv_items  = pd.read_csv(csv_file)
        self.modal_num  = modal_num
        self.with_label = with_label
        self.transform  = transform
        self.task       = task
        assert self.task in  [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]
       
        csv_keys = list(self.csv_items.keys())
        if('label' not in csv_keys):
            logging.warning("`label` section is not found in the csv file {0:}".format(
                csv_file) + "\n -- This is only allowed for self-supervised learning" + 
                "\n -- when `SelfSuperviseLabel` is used in the transform, or when" + 
                "\n -- loading the unlabeled data for preprocessing.")
            self.with_label = False
        self.image_weight_idx = None
        self.pixel_weight_idx = None
        if('image_weight' in csv_keys):
            self.image_weight_idx = csv_keys.index('image_weight')
        if('pixel_weight' in csv_keys):
            self.pixel_weight_idx = csv_keys.index('pixel_weight')

    def __len__(self):
        return len(self.csv_items)

    def __getlabel__(self, idx):
        csv_keys = list(self.csv_items.keys())        
        label_idx  = csv_keys.index('label')
        label_name = self.csv_items.iloc[idx, label_idx]
        label_name_full = "{0:}/{1:}".format(self.root_dir, label_name)
        label = load_image_as_nd_array(label_name_full)['data_array']
        if(self.task ==  TaskType.SEGMENTATION):
            label = np.asarray(label, np.int32)
        elif(self.task == TaskType.RECONSTRUCTION):
            label = np.asarray(label, np.float32)
        return label, label_name

    def __get_pixel_weight__(self, idx):
        weight_name = "{0:}/{1:}".format(self.root_dir, 
            self.csv_items.iloc[idx, self.pixel_weight_idx])
        weight = load_image_as_nd_array(weight_name)['data_array']
        weight = np.asarray(weight, np.float32)
        return weight        

    # def __getitem__(self, idx):
    #     sample_name = self.csv_items.iloc[idx, 0]
    #     h5f = h5py.File(self.root_dir + '/' +  sample_name, 'r')
    #     image = np.asarray(h5f['image'][:], np.float32)
        
    #     # this a temporaory process, will be delieted later
    #     if(len(image.shape) == 3 and image.shape[0] > 1):
    #         image = np.expand_dims(image, 0)
    #     sample = {'image': image, 'names':sample_name}
        
    #     if('label' in h5f):
    #         label = np.asarray(h5f['label'][:], np.uint8)
    #         if(len(label.shape) == 3 and label.shape[0] > 1):
    #             label = np.expand_dims(label, 0)
    #         sample['label'] = label
    #     if self.transform:
    #         sample = self.transform(sample)
    #     return sample

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
        
        sample = {'image': image, 'names' : names_list, 
                 'origin':image_dict['origin'],
                 'spacing': image_dict['spacing'],
                 'direction':image_dict['direction']}
        if (self.with_label):   
            sample['label'], label_name = self.__getlabel__(idx) 
            sample['names'].append(label_name)
            assert(image.shape[1:] == sample['label'].shape[1:])
        if (self.image_weight_idx is not None):
            sample['image_weight'] = self.csv_items.iloc[idx, self.image_weight_idx]
        if (self.pixel_weight_idx is not None):
            sample['pixel_weight'] = self.__get_pixel_weight__(idx) 
            assert(image.shape[1:] == sample['pixel_weight'].shape[1:])
        if self.transform:
            sample = self.transform(sample)

        return sample


class ClassificationDataset(NiftyDataset):
    """
    Dataset for loading images for classification. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors 
    with dimention order [C, H, W] for 2D images.

    :param root_dir: (str) Directory with all the images. 
    :param csv_file: (str) Path to the csv file with image names.
    :param modal_num: (int) Number of modalities. 
    :param class_num: (int) Class number of the classificaiton task.
    :param with_label: (bool) Load the data with segmentation ground truth or not.
    :param transform:  (list) List of transforms to be applied on a sample.
        The built-in transforms can listed in :mod:`pymic.transform.trans_dict`.
    """
    def __init__(self, root_dir, csv_file, modal_num = 1, class_num = 2, 
            with_label = False, transform=None, task = TaskType.CLASSIFICATION_ONE_HOT):
        super(ClassificationDataset, self).__init__(root_dir, 
            csv_file, modal_num, with_label, transform)
        self.class_num = class_num
        self.task      = task
        assert self.task in  [TaskType.CLASSIFICATION_ONE_HOT, TaskType.CLASSIFICATION_COEXIST]

    def __getlabel__(self, idx):
        csv_keys = list(self.csv_items.keys())
        label_idx = csv_keys.index('label')
        label = self.csv_items.iloc[idx, label_idx]
        return label
    
    def __getweight__(self, idx):
        weight = self.csv_items.iloc[idx, self.image_weight_idx]
        weight = weight + 0.0
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
        if (self.image_weight_idx is not None):
            sample['image_weight'] = self.__getweight__(idx) 
        if self.transform:
            sample = self.transform(sample)
        return sample
