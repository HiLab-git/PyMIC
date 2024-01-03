# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import random
import logging
import torch
import numpy as np
import torch.optim as optim
from abc import ABCMeta, abstractmethod
from pymic.net_run.get_optimizer import get_lr_scheduler, get_optimizer

def seed_torch(seed=1):
    """
    Set random seed.

    :param seed: (int) the seed for random. 
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class NetRunAgent(object):
    """
    The abstract class for medical image segmentation.

    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        The config dictionary should have at least four sections: `dataset`,
        `network`, `training` and `inference`. See :doc:`usage.quickstart` and
        :doc:`usage.fsl` for example.

    """
    __metaclass__ = ABCMeta
    def __init__(self, config, stage = 'train'):
        assert(stage in ['train', 'inference', 'test'])
        self.config = config
        self.stage  = stage
        if(stage == 'inference'):
            self.stage = 'test'
        self.train_set = None 
        self.valid_set = None 
        self.test_set  = None
        self.net       = None
        self.optimizer = None
        self.scheduler = None 
        self.net_dict  = None
        self.loss_dict = None 
        self.transform_dict  = None
        self.inferer   = None
        self.tensor_type   = config['dataset']['tensor_type']
        self.task_type     = config['dataset']['task_type'] 
        self.deterministic = config['training'].get('deterministic', True)
        self.random_seed   = config['training'].get('random_seed', 1)
        if(self.deterministic):
            seed_torch(self.random_seed)
            logging.info("deterministric is true")
        
    def set_datasets(self, train_set, valid_set, test_set):
        """
        Set customized datasets for training and inference.
        
        :param train_set: (torch.utils.data.Dataset) The training set.
        :param valid_set: (torch.utils.data.Dataset) The validation set.
        :param test_set: (torch.utils.data.Dataset) The testing set.
        """
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set  = test_set

    def set_transform_dict(self, custom_transform_dict):
        """
        Set the available Transforms, including customized Transforms.

        :param custom_transform_dict: (dictionary) A dictionary of 
          available Transforms.
        """
        self.transform_dict = custom_transform_dict

    def set_network(self, net):
        """
        Set the network.

        :param net: (nn.Module) A deep learning network.
        """
        self.net = net 

    def set_net_dict(self, net_dict):
        """
        Set the available networks, including customized networks.

        :param net_dict: (dictionary) A dictionary of available networks.
        """
        self.net_dict = net_dict

    def set_loss_dict(self, loss_dict):
        """
        Set the available loss functions, including customized loss functions.

        :param loss_dict: (dictionary) A dictionary of available loss functions.
        """
        self.loss_dict = loss_dict

    def set_optimizer(self, optimizer):
        """
        Set the optimizer.

        :param optimizer: An optimizer.
        """        
        self.optimizer = optimizer
    
    def set_scheduler(self, scheduler):
        """
        Set the learning rate scheduler.

        :param scheduler: A learning rate scheduler.
        """
        self.scheduler = scheduler
    
    def set_inferer(self, inferer):
        """
        Set the inferer.

        :param inferer: An inferer object.
        """
        self.inferer = inferer

    def get_checkpoint_name(self):
        """
        Get the checkpoint name for inference based on config['testing']['ckpt_mode']. 
        """
        ckpt_mode = self.config['testing']['ckpt_mode']
        if(ckpt_mode == 0 or ckpt_mode == 1):
            ckpt_dir    = self.config['training']['ckpt_save_dir']
            ckpt_prefix = self.config['training'].get('ckpt_prefix', None)
            if(ckpt_prefix is None):
                ckpt_prefix = ckpt_dir.split('/')[-1]
            txt_name = ckpt_dir + '/' + ckpt_prefix
            txt_name += "_latest.txt" if ckpt_mode == 0 else "_best.txt"
            with open(txt_name, 'r') as txt_file:
                it_num = txt_file.read().replace('\n', '') 
                ckpt_name = "{0:}/{1:}_{2:}.pt".format(ckpt_dir, ckpt_prefix, it_num)
                if(ckpt_mode == 1 and not os.path.isfile(ckpt_name)):
                    ckpt_name = "{0:}/{1:}_best.pt".format(ckpt_dir, ckpt_prefix)
        else:
            ckpt_name =  self.config['testing']['ckpt_name']
        return ckpt_name

    @abstractmethod    
    def get_stage_transform_from_config(self, stage):
        """
        Get the transform list required by dataset for training, validation or inference stage. 

        :param stage: (str) `train`, `valid` or `test`.
        """
        raise(ValueError("not implemented"))

    @abstractmethod    
    def get_stage_dataset_from_config(self, stage):
        """
        Create dataset based on training, validation or inference stage. 

        :param stage: (str) `train`, `valid` or `test`.
        """
        raise(ValueError("not implemented"))

    @abstractmethod
    def get_parameters_to_update(self):
        """
        Get parameters for update. 
        """
        raise(ValueError("not implemented"))

    @abstractmethod
    def get_loss_value(self, data, pred, gt, param = None):
        """
        Get the loss value.  Assume `pred` and `gt` has been sent to self.device.
        `data` is obtained by dataloader, and is a dictionary containing extra 
        information, such as pixel-level weight. By default, such information 
        is not used by standard loss functions such as Dice loss and cross entropy loss.  


        :param data: (dictionary) A data dictionary obtained by dataloader.
        :param pred: (tensor) Prediction result by the network. 
        :param gt: (tensor) Ground truth.
        :param param: (dictionary) Other parameters if needed.
        """
        raise(ValueError("not implemented"))

    @abstractmethod
    def create_network(self):
        """
        Create network based on configuration.
        """
        raise(ValueError("not implemented"))

    @abstractmethod
    def create_loss_calculator(self):
        """
        Create loss function object.
        """
        raise(ValueError("not implemented"))

    @abstractmethod
    def training(self):
        """
        Train the network
        """
        raise(ValueError("not implemented"))
        
    @abstractmethod
    def validation(self):
        """
        Evaluate the performance on the validation set.
        """
        raise(ValueError("not implemented"))

    @abstractmethod
    def train_valid(self):
        """
        Train and valid. 
        """
        raise(ValueError("not implemented"))
    
    @abstractmethod
    def infer(self):
        """
        Inference on testing set. 
        """
        raise(ValueError("not implemented"))

    @abstractmethod
    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        """
        Write scalars using SummaryWriter.

        :param train_scalars: (dictionary) Scalars for training set. 
        :param valid_scalars: (dictionary) Scalars for validation set. 
        :param lr_value: (float) Current learning rate.
        :param glob_it: (int) Current iteration number.
        """
        raise(ValueError("not implemented"))

    def create_dataset(self):
        """
        Create datasets for training, validation or testing based on configuraiton.  
        """
        if(self.stage == 'train'):
            if(self.train_set is None):
                self.train_set = self.get_stage_dataset_from_config('train')
            if(self.valid_set is None):
                self.valid_set = self.get_stage_dataset_from_config('valid')
            if(self.deterministic):
                def worker_init_fn(worker_id):
                    # workder_seed = self.random_seed+worker_id 
                    workder_seed = torch.initial_seed() % 2 ** 32
                    np.random.seed(workder_seed)
                    random.seed(workder_seed)                    
                worker_init = worker_init_fn
            else:
                worker_init = None

            bn_train = self.config['dataset']['train_batch_size']
            bn_valid = self.config['dataset'].get('valid_batch_size', 1)
            num_worker = self.config['dataset'].get('num_worker', 8)
            g_train, g_valid = torch.Generator(), torch.Generator()
            g_train.manual_seed(self.random_seed)
            g_valid.manual_seed(self.random_seed)
            self.train_loader = torch.utils.data.DataLoader(self.train_set, 
                batch_size = bn_train, shuffle=True, num_workers= num_worker,
                worker_init_fn=worker_init, generator = g_train, drop_last = True)
            self.valid_loader = torch.utils.data.DataLoader(self.valid_set, 
                batch_size = bn_valid, shuffle=False, num_workers= num_worker,
                worker_init_fn=worker_init, generator = g_valid)
        else:
            bn_test = self.config['dataset'].get('test_batch_size', 1)
            if(self.test_set  is None):
                self.test_set  = self.get_stage_dataset_from_config('test')
            self.test_loader = torch.utils.data.DataLoader(self.test_set, 
                batch_size = bn_test, shuffle=False, num_workers= bn_test)
       
    def create_optimizer(self, params, checkpoint = None):
        """
        Create optimizer based on configuration. 

        :param params: network parameters for optimization. Usually it is obtained by 
            `self.get_parameters_to_update()`.
        """
        opt_params = self.config['training']
        if(self.optimizer is None):
            self.optimizer = get_optimizer(opt_params['optimizer'],
                    params, opt_params)
        last_iter = -1
        if(checkpoint is not None):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_iter = checkpoint['iteration'] - 1
        if(self.scheduler is None):
            opt_params["last_iter"] = last_iter
            self.scheduler = get_lr_scheduler(self.optimizer, opt_params)

    def convert_tensor_type(self, input_tensor):
        """
        Convert the type of an input tensor to float or double based on configuration. 
        """
        if(self.tensor_type == 'float'):
            return input_tensor.float()
        else:
            return input_tensor.double()

    def run(self):
        """
        Run the training or inference code according to configuration.
        """
        self.create_dataset()
        self.create_network()
        if(self.stage == 'train'):
            self.train_valid()
        else:
            self.infer()

