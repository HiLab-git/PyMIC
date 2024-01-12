# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import sys
import torch
import torchvision.transforms as transforms
from pymic.util.parse_config import *
from pymic.io.image_read_write import save_nd_array_as_image
from pymic.io.nifty_dataset import NiftyDataset
from pymic.transform.trans_dict import TransformDict
from pymic.net_run.agent_abstract import seed_torch
from pymic.net_run.self_sup.util import volume_fusion

class PreprocessAgent(object):
    def __init__(self, config):
        super(PreprocessAgent, self).__init__()
        self.config = config
        self.transform_dict  = TransformDict
        self.task_type       = config['dataset']['task_type'] 
        self.dataloader      = None 
        self.dataloader_unlab= None 

        deterministic = config['dataset'].get('deterministic', True)
        if(deterministic):
            random_seed = config['dataset'].get('random_seed', 1)
            seed_torch(random_seed)
        
    def get_dataset_from_config(self):
        root_dir  = self.config['dataset']['data_dir']
        modal_num = self.config['dataset'].get('modal_num', 1)
        transform_names = self.config['dataset']["transform"]
        
        self.transform_list  = []
        if(transform_names is None or len(transform_names) == 0):
            data_transform = None 
        else:
            transform_param = self.config['dataset']
            transform_param['task'] = self.task_type
            for name in transform_names:
                if(name not in self.transform_dict):
                    raise(ValueError("Undefined transform {0:}".format(name))) 
                one_transform = self.transform_dict[name](transform_param)
                self.transform_list.append(one_transform)
            data_transform = transforms.Compose(self.transform_list)

        data_csv         = self.config['dataset'].get('data_csv', None)
        data_csv_unlab   = self.config['dataset'].get('data_csv_unlab', None)
        batch_size       = self.config['dataset'].get('batch_size', 1)
        data_shuffle     = self.config['dataset'].get('data_shuffle', False)
        if(data_csv is not None):
            dataset  = NiftyDataset(root_dir  = root_dir,
                                    csv_file  = data_csv,
                                    modal_num = modal_num,
                                    with_label= True,
                                    transform = data_transform, 
                                    task = self.task_type)
            self.dataloader = torch.utils.data.DataLoader(dataset, 
                batch_size = batch_size, shuffle=data_shuffle, num_workers= 8,
                worker_init_fn=None, generator = torch.Generator())
        if(data_csv_unlab is not None):
            dataset_unlab  = NiftyDataset(root_dir  = root_dir,
                                    csv_file  = data_csv_unlab,
                                    modal_num = modal_num,
                                    with_label= False,
                                    transform = data_transform, 
                                    task = self.task_type)
            self.dataloader_unlab = torch.utils.data.DataLoader(dataset_unlab, 
                batch_size = batch_size, shuffle=data_shuffle, num_workers= 8,
                worker_init_fn=None, generator = torch.Generator())

    def run(self):
        """
        Do preprocessing for labeled and unlabeled data.  
        """
        self.get_dataset_from_config()
        out_dir   = self.config['dataset']['output_dir']
        modal_num = self.config['dataset']['modal_num']
        if(not os.path.isdir(out_dir)):
            os.mkdir(out_dir)
        batch_operation = self.config['dataset'].get('batch_operation', None)
        for dataloader in [self.dataloader, self.dataloader_unlab]:
            if(dataloader is None):
                continue
            for data in dataloader:
                inputs    = data['image']
                labels    = data.get('label', None)                    
                img_names = data['names']
                if(len(img_names) == modal_num): # for unlabeled dataset
                    lab_names = [item.replace(".nii.gz", "_lab.nii.gz") for item in img_names[0]] 
                else:
                    lab_names = img_names[-1]
                B, C    = inputs.shape[0], inputs.shape[1]
                spacing = [x.numpy()[0] for x in data['spacing']]
                
                if(batch_operation is not None and 'VolumeFusion' in batch_operation):
                    class_num   = self.config['dataset']['VolumeFusion_cls_num'.lower()]
                    block_range = self.config['dataset']['VolumeFusion_block_range'.lower()]
                    size_min    = self.config['dataset']['VolumeFusion_size_min'.lower()]
                    size_max    = self.config['dataset']['VolumeFusion_size_max'.lower()]
                    inputs, labels = volume_fusion(inputs, class_num - 1, block_range, size_min, size_max)

                for b in range(B):
                    for c in range(C):
                        image_name = out_dir + "/" + img_names[c][b]
                        print(image_name)
                        save_nd_array_as_image(inputs[b][c], image_name, reference_name = None, spacing=spacing)        
                    if(labels is not None):
                        label_name = out_dir + "/" + lab_names[b]
                        print(label_name)
                        save_nd_array_as_image(labels[b][0], label_name, reference_name = None, spacing=spacing)
