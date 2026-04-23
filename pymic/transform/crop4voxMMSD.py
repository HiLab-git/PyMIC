# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch

import json
import math
import random
import numpy as np
from imops import crop_to_box
from typing import *
from scipy import ndimage
from pymic import TaskType
from pymic.transform.crop4vox2vec import sample_views
from pymic.transform.abstract_transform import AbstractTransform
from pymic.transform.crop import CenterCrop
from pymic.util.image_process import *
from pymic.transform.intensity import *

def chanel_block_wise_mask(image, block_size, mask_prob):
    C, D, H, W = image.shape
    img_out = copy.deepcopy(image)
    block   = np.zeros(block_size)
    for c in range(C):
        for d in range(0, D, block_size[0]):
            d1 = min(d + block_size[0], D) 
            for h in range(0, H, block_size[1]):
                h1 = min(h + block_size[1], H)
                for w in range(0, W, block_size[2]):
                    w1 = min(w + block_size[2], W)
                    if (random.random() < mask_prob):
                        img_out[c, d:d1, h:h1, w:w1] = block
    return img_out

class Crop4VoxMMSD(CenterCrop):
    """Take two random crops of one image as the query and key."""

    def __init__(self, params):
        self.output_size = params['Crop4VoxMMSD_output_size'.lower()]
        self.min_overlap = params.get('Crop4VoxMMSD_min_overlap'.lower(), [8, 12, 12])
        self.max_voxel   = params.get('Crop4VoxMMSD_max_voxel'.lower(), 1024)
        self.base_transform = params.get('Crop4VoxMMSD_base_transform'.lower(), None)
        self.inverse     = params.get('Crop4VoxMMSD_inverse'.lower(), False)
        self.task        = params['Task'.lower()]
        assert isinstance(self.output_size, (list, tuple))

    def __call__(self, sample):
        image = sample['image']
        channel, input_size = image.shape[0], image.shape[1:]
        input_dim = len(input_size)
        assert(input_dim == len(self.output_size))
        
        min_overlap = [max(self.output_size[i]*2 - input_size[i], self.min_overlap[i]) for i in range(3)]
        patches_1, patches_2, voxels_1, voxels_2 = sample_views(image, 
            min_overlap, self.output_size, self.max_voxel)
        
        label_1, label_2 = copy.deepcopy(patches_1), copy.deepcopy(patches_2)
        sample_1 = {"image": patches_1}
        sample_2 = {"image": patches_2}
        if(self.base_transform is not None):
            image_1 = self.base_transform(sample_1)["image"]
            image_2 = self.base_transform(sample_2)["image"]
        # sample['image'] = patches_1, patches_2, voxels_1, voxels_2
        # while(1):
        #     sample_1 = copy.deepcopy(x)
        #     sample_2 = copy.deepcopy(x)
        #     image_1 = sample_1['image']
        #     image_2 = sample_2['image']
        #     voxels_1 = np.argwhere(image_1!=0)
        #     voxels_2 = np.argwhere(image_2!=0)
        #     # print(voxels_1.max(),voxels_2.max())

        #     box_1 = sample_box(image_1.shape, patch_size)
        #     box_2 = sample_box(image_2.shape, patch_size)
        #     image_1 = image_1[tuple(slice(st,end) for st,end in zip(box_1[0],box_1[1]))]
        #     image_2 = image_2[tuple(slice(st,end) for st,end in zip(box_2[0],box_2[1]))]
        #     sample_1['image'] = image_1
        #     sample_2['image'] = image_2
        #     label_1 = copy.deepcopy(self.norm(sample_1)['image'])
        #     label_2 = copy.deepcopy(self.norm(sample_2)['image'])
        #     image_1 = self.mask_transform(sample_1)['image']
        #     image_2 = self.mask_transform(sample_2)['image']

        #     shift_1 = box_1[0]
        #     voxels_1 = voxels_1 - shift_1 
        #     shift_2 = box_2[0]
        #     voxels_2 = voxels_2 - shift_2
            
        #     valid_1 = np.all((voxels_1 >= 0) & (voxels_1 < patch_size), axis=1)
        #     valid_2 = np.all((voxels_2 >= 0) & (voxels_2 < patch_size), axis=1)
        #     valid = valid_1 & valid_2
        #     indices = np.where(valid)[0]

        #     overlapping_voxels_1 = voxels_1[indices]
        #     overlapping_voxels_2 = voxels_2[indices]
        #     # print(overlapping_voxels_1.max(),overlapping_voxels_2.max())
        #     random_mod_index_1 = np.random.randint(0,4,size=overlapping_voxels_1.shape[0])
        #     random_mod_index_2 = np.random.randint(0,4,size=overlapping_voxels_2.shape[0])
        #     unmasked_valid_1 = image_1[random_mod_index_1,overlapping_voxels_1[:,1],overlapping_voxels_1[:,2],overlapping_voxels_1[:,3]]>0
        #     unmasked_valid_2 = image_2[random_mod_index_2,overlapping_voxels_2[:,1],overlapping_voxels_2[:,2],overlapping_voxels_2[:,3]]>0

        #     unmasked_valid = unmasked_valid_1 & unmasked_valid_2
            
        #     if(np.sum(unmasked_valid)>=max_num_voxels):
        #         choice_index = np.random.choice(np.arange(unmasked_valid.shape[0])[unmasked_valid],max_num_voxels, replace=False)
        #         final_indices_1 = np.array([random_mod_index_1[choice_index],overlapping_voxels_1[choice_index,1],overlapping_voxels_1[choice_index,2],overlapping_voxels_1[choice_index,3]]).T
        #         final_indices_2 = np.array([random_mod_index_2[choice_index],overlapping_voxels_2[choice_index,1],overlapping_voxels_2[choice_index,2],overlapping_voxels_2[choice_index,3]]).T
        #         break

        # sample_1['image']=image_1
        # sample_2['image']=image_2 
        # image_1 = self.base_transform(sample_1)['image']
        # image_2 = self.base_transform(sample_2)['image']
        sample_out = {}
        sample_out['label'] = [label_1, label_2]
        sample_out['image'] = [image_1, image_2]
        sample_out['voxel'] = [voxels_1, voxels_2]
        return sample_out
   
