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



class PreprocessAgent(object):
    def __init__(self, config):
        super(PreprocessAgent, self).__init__()
        self.config = config
        self.transform_dict  = TransformDict
        self.task_type       = config['dataset']['task_type'] 
        self.dataloader      = None 
        self.dataloader_unlab= None 
        
    def get_dataset_from_config(self):
        root_dir  = self.config['dataset']['root_dir']
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
        if(data_csv is not None):
            dataset  = NiftyDataset(root_dir  = root_dir,
                                    csv_file  = data_csv,
                                    modal_num = modal_num,
                                    with_label= True,
                                    transform = data_transform, 
                                    task = self.task_type)
            self.dataloader = torch.utils.data.DataLoader(dataset, 
                batch_size = 1, shuffle=False, num_workers= 8,
                worker_init_fn=None, generator = torch.Generator())
        if(data_csv_unlab is not None):
            dataset_unlab  = NiftyDataset(root_dir  = root_dir,
                                    csv_file  = data_csv_unlab,
                                    modal_num = modal_num,
                                    with_label= False,
                                    transform = data_transform, 
                                    task = self.task_type)
            self.dataloader_unlab = torch.utils.data.DataLoader(dataset_unlab, 
                batch_size = 1, shuffle=False, num_workers= 8,
                worker_init_fn=None, generator = torch.Generator())

    def run(self):
        """
        Do preprocessing for labeled and unlabeled data.  
        """
        self.get_dataset_from_config()
        out_dir = self.config['dataset']['output_dir']
        for dataloader in [self.dataloader, self.dataloader_unlab]:
            for item in dataloader:
                img = item['image'][0] # the batch size is 1
                # save differnt modaliteis 
                img_names = item['names']
                spacing   = [x.numpy()[0] for x in item['spacing']]
                for i in range(img.shape[0]):
                    image_name = out_dir + "/" + img_names[i][0]
                    print(image_name)
                    save_nd_array_as_image(img[i], image_name, reference_name = None, spacing=spacing)        
                if('label' in item):
                    lab = item['label'][0]
                    label_name = out_dir + "/" + img_names[-1][0]
                    print(label_name)
                    save_nd_array_as_image(lab[0], label_name, reference_name = None, spacing=spacing)

def main():
    """
    The main function for data preprocessing.
    """
    if(len(sys.argv) < 2):
        print('Number of arguments should be 2. e.g.')
        print('   pymic_preprocess config.cfg')
        exit()
    cfg_file = str(sys.argv[1])
    if(not os.path.isfile(cfg_file)):
        raise ValueError("The config file does not exist: " + cfg_file)
    config = parse_config(cfg_file)
    config = synchronize_config(config)
    agent  = PreprocessAgent(config)
    agent.run()

if __name__ == "__main__":
    main()
    
