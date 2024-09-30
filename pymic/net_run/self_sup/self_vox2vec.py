# -*- coding: utf-8 -*-
from __future__ import print_function, division
import copy
import logging
import time
import logging
import torch
import torch.nn as nn
from datetime import datetime
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from pymic.io.image_read_write import save_nd_array_as_image
from pymic.net.net3d.fmunetv3 import FMUNetV3
from pymic.net_run.agent_seg import SegmentationAgent
from pymic.loss.cls.infoNCE import InfoNCELoss

def select_from_pyramid(feature_pyramid, indices):
    """Select features from feature pyramid by their indices w.r.t. base feature map.

    Args:
        feature_pyramid (Sequence[torch.Tensor]): Sequence of tensors of shapes ``(B, C_i, D_i, H_i, W_i)``.
        indices (torch.Tensor): tensor of shape ``(B, N, 3)``

    Returns:
        torch.Tensor: tensor of shape ``(B, N, \sum_i c_i)``
    """
    out = []
    for i, x in enumerate(feature_pyramid):  
        batch_size = list(x.shape)[0]
        x_move = x.moveaxis(1, -1)
        index_i = indices // 2 ** i
        x_i = [x_move[b][index_i[b][:, 0], index_i[b][:, 1], index_i[b][:, 2], :] for \
            b in range(batch_size)]
        x_i = torch.stack(x_i)
        out.append(x_i)
    out = torch.cat(out, dim = -1)
    return out

class Vox2VecHead(nn.Module):
    def __init__(self, params):
        super(Vox2VecHead, self).__init__()
        ft_chns    = params['feature_chns']
        hidden_dim = params['hidden_dim']
        proj_dim   = params['project_dim']
        embed_dim = sum(ft_chns)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        output = self.proj_head(x)
        output = nn.functional.normalize(output)
        return output

class Vox2VecWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(Vox2VecWrapper, self).__init__()
        self.backbone = backbone
        self.head = head
        
    def forward(self, x, vex_idx):
        if(isinstance(self.backbone, FMUNetV3)):
            x = self.backbone.project(x)
        f = self.backbone.encoder(x)
        B = list(f[0].shape)[0]
        f_fpn  = select_from_pyramid(f, vex_idx)
        feature_dim = list(f_fpn.shape)[-1]
        f_fpn    = f_fpn.view(-1, feature_dim)
        output   = self.head(f_fpn)
        proj_dim = list(output.shape)[-1]
        output   = output.view(B, -1, proj_dim)
        return output

class SelfSupVox2Vec(SegmentationAgent):
    """
    An agent for image self-supervised learning with DeSD.
    """
    def __init__(self, config, stage = 'train'):
        super(SelfSupVox2Vec, self).__init__(config, stage)

    def create_network(self):
        super(SelfSupVox2Vec, self).create_network()
        proj_dim   = self.config['self_supervised_learning'].get('project_dim', 1024)
        hidden_dim = self.config['self_supervised_learning'].get('hidden_dim', 1024)
        head_params= {'feature_chns': self.config['network']['feature_chns'],
            'hidden_dim':hidden_dim,
            'project_dim':proj_dim}
        self.head = Vox2VecHead(head_params)
        self.net_wrapper = Vox2VecWrapper(self.net, self.head)

    def create_loss_calculator(self):
        # constrastive loss
        self_sup_params = self.config['self_supervised_learning']
        self.loss_calculator = InfoNCELoss(self_sup_params)

    def get_parameters_to_update(self):
        params = self.net_wrapper.parameters()
        return params

    def training(self):
        iter_valid  = self.config['training']['iter_valid']
        train_loss  = 0
        err_info    = None
        data_time, gpu_time, loss_time, back_time = 0, 0, 0, 0
        self.net_wrapper.train()
        for it in range(iter_valid):
            t0 = time.time()
            try:
                data = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data = next(self.trainIter)
            t1 = time.time()
            patch1, patch2, vox_ids1, vox_ids2 = data['image']
            inputs  = torch.cat([patch1, patch2], dim = 0)
            vox_ids = torch.cat([vox_ids1, vox_ids2], dim = 0)
            inputs  = self.convert_tensor_type(inputs)
            inputs  = inputs.to(self.device)
            vox_ids = vox_ids.to(self.device)
            
            # for debug
            # for i in range(patch1.shape[0]):
            #     v1_i = patch1[i][0]
            #     v2_i = patch2[i][0]
            #     print("patch shape", v1_i.shape, v2_i.shape)
            #     image_name0 = "temp/image_{0:}_{1:}_v0.nii.gz".format(it, i)
            #     image_name1 = "temp/image_{0:}_{1:}_v1.nii.gz".format(it, i)
            #     save_nd_array_as_image(v1_i, image_name0, reference_name = None)
            #     save_nd_array_as_image(v2_i, image_name1, reference_name = None)
            # if(it > 10):
            #     return 

            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            # forward + backward + optimize
            out = self.net_wrapper(inputs, vox_ids)
            out1, out2 = out.chunk(2)

            t2 = time.time()
            loss = self.loss_calculator(out1, out2)         
            t3 = time.time()

            loss.backward()
            self.optimizer.step()
            train_loss = train_loss + loss.item()            
            t4 = time.time()

            data_time = data_time + t1 - t0 
            gpu_time  = gpu_time  + t2 - t1
            loss_time = loss_time + t3 - t2
            back_time = back_time + t4 - t3

        train_avg_loss = train_loss / iter_valid
        train_scalers = {'loss': train_avg_loss, 'data_time': data_time, 
         'gpu_time':gpu_time, 'loss_time':loss_time, 'back_time':back_time,
         'err_info': err_info}
        return train_scalers
        
   
    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_scalar ={'train':train_scalars['loss']}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
        logging.info('train loss {0:.4f}'.format(train_scalars['loss']))        

    def train_valid(self):
        device_ids = self.config['training']['gpus']
        if(len(device_ids) > 1):
            self.device = torch.device("cuda:0")
            self.net_wrapper = nn.DataParallel(self.net_wrapper, device_ids = device_ids)
        else:
            self.device = torch.device("cuda:{0:}".format(device_ids[0]))
        self.net_wrapper.to(self.device)

        ckpt_dir    = self.config['training']['ckpt_dir']
        ckpt_prefix = self.config['training'].get('ckpt_prefix', None)
        if(ckpt_prefix is None):
            ckpt_prefix = ckpt_dir.split('/')[-1]
        # iter_start  = self.config['training']['iter_start']
        iter_start  = 0 
        iter_max    = self.config['training']['iter_max']
        iter_valid  = self.config['training']['iter_valid']
        iter_save   = self.config['training'].get('iter_save', None)
        early_stop_it = self.config['training'].get('early_stop_patience', None)
        if(iter_save is None):
            iter_save_list = [iter_max]
        elif(isinstance(iter_save, (tuple, list))):
            iter_save_list = iter_save
        else:
            iter_save_list = range(0, iter_max + 1, iter_save)

        self.min_loss = 10000.0
        self.min_loss_it   = 0
        self.best_model_wts = None 
        self.bett_head_wts  = None
        checkpoint = None
         # initialize the network with pre-trained weights
        ckpt_init_name = self.config['training'].get('ckpt_init_name', None)
        ckpt_init_mode = self.config['training'].get('ckpt_init_mode', 0)
        ckpt_for_optm  = None 
        if(ckpt_init_name is not None):
            checkpoint = torch.load(ckpt_dir + "/" + ckpt_init_name, map_location = self.device)
            pretrained_dict = checkpoint['model_state_dict']
            pretrain_head_dict = checkpoint['head_state_dict']
            self.load_pretrained_weights(self.net, pretrained_dict, device_ids)
            self.load_pretrained_weights(self.head, pretrain_head_dict, device_ids)

            if(ckpt_init_mode > 0): # Load  other information
                self.min_loss = checkpoint.get('train_loss', 10000)
                iter_start = checkpoint['iteration']
                self.min_loss_it = iter_start
                self.best_model_wts = checkpoint['model_state_dict']
                self.best_head_wts  = checkpoint['head_state_dict']
                ckpt_for_optm = checkpoint
            
        self.create_optimizer(self.get_parameters_to_update(), ckpt_for_optm)
        self.create_loss_calculator()
    
        self.trainIter  = iter(self.train_loader)
        
        logging.info("{0:} training start".format(str(datetime.now())[:-7]))
        self.summ_writer = SummaryWriter(self.config['training']['ckpt_dir'])
        self.glob_it = iter_start
        for it in range(iter_start, iter_max, iter_valid):
            lr_value = self.optimizer.param_groups[0]['lr']

            t0 = time.time()
            train_scalars = self.training()
            t1 = time.time()
            if(isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau)):
                self.scheduler.step(-train_scalars['loss'])
            else:
                self.scheduler.step()

            self.glob_it = it + iter_valid
            logging.info("\n{0:} it {1:}".format(str(datetime.now())[:-7], self.glob_it))
            logging.info('learning rate {0:}'.format(lr_value))
            logging.info("training time: {0:.2f}s".format(t1-t0))
            logging.info("data: {0:.2f}s, gpu: {1:.2f}s, loss: {2:.2f}s, back: {3:.2f}s".format(
                train_scalars['data_time'], train_scalars['gpu_time'], 
                train_scalars['loss_time'], train_scalars['back_time']))

            self.write_scalars(train_scalars, None, lr_value, self.glob_it)
            if(train_scalars['loss'] < self.min_loss):
                self.min_loss = train_scalars['loss']
                self.min_loss_it  = self.glob_it
                if(len(device_ids) > 1):
                    self.best_model_wts = copy.deepcopy(self.net.module.state_dict())
                    self.best_head_wts = copy.deepcopy(self.head.module.state_dict())
                else:
                    self.best_model_wts = copy.deepcopy(self.net.state_dict())
                    self.best_head_wts = copy.deepcopy(self.head.state_dict())
                
                save_dict = {'iteration': self.min_loss_it,
                    'train_loss': self.min_loss,
                    'model_state_dict': self.best_model_wts,
                    'head_state_dict': self.best_head_wts,
                    'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}/{1:}_best.pt".format(ckpt_dir, ckpt_prefix)
                torch.save(save_dict, save_name) 
                txt_file = open("{0:}/{1:}_best.txt".format(ckpt_dir, ckpt_prefix), 'wt')
                txt_file.write(str(self.min_loss_it))
                txt_file.close()

            stop_now = True if(early_stop_it is not None and \
                self.glob_it - self.min_loss_it > early_stop_it) else False
            if(train_scalars['err_info'] is not None):
                logging.info("Early stopped due to error: {0:}".format(train_scalars['err_info']))
                stop_now = True
            if ((self.glob_it in iter_save_list) or stop_now):
                save_dict = {'iteration': self.glob_it,
                             'train_loss': train_scalars['loss'],
                             'model_state_dict': self.net.module.state_dict() \
                                 if len(device_ids) > 1 else self.net.state_dict(),
                             'head_state_dict': self.head.module.state_dict() \
                                 if len(device_ids) > 1 else self.head.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}/{1:}_{2:}.pt".format(ckpt_dir, ckpt_prefix, self.glob_it)
                torch.save(save_dict, save_name) 
                txt_file = open("{0:}/{1:}_latest.txt".format(ckpt_dir, ckpt_prefix), 'wt')
                txt_file.write(str(self.glob_it))
                txt_file.close()
            if(stop_now):
                logging.info("The training is early stopped")
                break
        # save the best performing checkpoint
        logging.info('The best performing iter is {0:}, train loss {1:}'.format(\
            self.min_loss_it, self.min_loss))
        self.summ_writer.close()