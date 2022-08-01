# -*- coding: utf-8 -*-
"""
Caculating the confidence map of labels of training samples,
which is used in the method of SLSR.
    Minqing Zhang et al., Characterizing Label Errors: Confident Learning
    for Noisy-Labeled Image Segmentation, MICCAI 2020.    
"""

from __future__ import print_function, division
import cleanlab
import logging
import os
import scipy
import sys
import torch
import numpy as np 
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pymic.io.nifty_dataset import NiftyDataset
from pymic.transform.trans_dict import TransformDict
from pymic.util.parse_config import *
from pymic.net_run.agent_seg import SegmentationAgent
from pymic.net_run.infer_func import Inferer

def get_confident_map(gt, pred, CL_type = 'both'):
    """
    gt: ground truth label (one-hot) with shape of NXC
    pred: digit prediction of network with shape of NXC
    """
    prob = scipy.special.softmax(pred, axis = 1)
    if CL_type in ['both', 'Qij']:
        noise = cleanlab.pruning.get_noise_indices(gt, prob, prune_method='both', n_jobs=1)
    elif CL_type == 'Cij':
        noise = cleanlab.pruning.get_noise_indices(gt, pred, prune_method='both', n_jobs=1)
    elif CL_type == 'intersection':
        noise_qij = cleanlab.pruning.get_noise_indices(gt, prob, prune_method='both', n_jobs=1)
        noise_cij = cleanlab.pruning.get_noise_indices(gt, pred, prune_method='both', n_jobs=1)
        noise = noise_qij & noise_cij
    elif CL_type == 'union':
        noise_qij = cleanlab.pruning.get_noise_indices(gt, prob, prune_method='both', n_jobs=1)
        noise_cij = cleanlab.pruning.get_noise_indices(gt, pred, prune_method='both', n_jobs=1)
        noise = noise_qij | noise_cij
    elif CL_type in ['prune_by_class', 'prune_by_noise_rate']:
        noise = cleanlab.pruning.get_noise_indices(gt, prob, prune_method=CL_type, n_jobs=1)
    return noise

class SegmentationAgentwithCL(SegmentationAgent):
    def __init__(self, config, stage = 'test'):
        super(SegmentationAgentwithCL, self).__init__(config, stage)

    def infer_with_cl(self):
        device_ids = self.config['testing']['gpus']
        device = torch.device("cuda:{0:}".format(device_ids[0]))
        self.net.to(device)

        if(self.config['testing'].get('evaluation_mode', True)):
            self.net.eval()
            if(self.config['testing'].get('test_time_dropout', False)):
                def test_time_dropout(m):
                    if(type(m) == nn.Dropout):
                        logging.info('dropout layer')
                        m.train()
                self.net.apply(test_time_dropout)

        ckpt_mode = self.config['testing']['ckpt_mode']
        ckpt_name = self.get_checkpoint_name()
        if(ckpt_mode == 3):
            assert(isinstance(ckpt_name, (tuple, list)))
            self.infer_with_multiple_checkpoints()
            return 
        else:
            if(isinstance(ckpt_name, (tuple, list))):
                raise ValueError("ckpt_mode should be 3 if ckpt_name is a list")

        # load network parameters and set the network as evaluation mode
        checkpoint = torch.load(ckpt_name, map_location = device)
        self.net.load_state_dict(checkpoint['model_state_dict'])

        if(self.inferer is None):
            infer_cfg = self.config['testing']
            class_num = self.config['network']['class_num']
            infer_cfg['class_num'] = class_num 
            self.inferer = Inferer(infer_cfg)
        pred_list  = []
        gt_list    = []
        filename_list = []
        with torch.no_grad():
            for data in self.test_loader:
                images = self.convert_tensor_type(data['image'])
                labels = self.convert_tensor_type(data['label_prob'])
                names  = data['names']
                filename_list.append(names)
                images = images.to(device)
    
                # for debug
                # for i in range(images.shape[0]):
                #     image_i = images[i][0]
                #     label_i = images[i][0]
                #     image_name = "temp/{0:}_image.nii.gz".format(names[0])
                #     label_name = "temp/{0:}_label.nii.gz".format(names[0])
                #     save_nd_array_as_image(image_i, image_name, reference_name = None)
                #     save_nd_array_as_image(label_i, label_name, reference_name = None)
                # continue
                
                pred = self.inferer.run(self.net, images)
                # convert tensor to numpy
                if(isinstance(pred, (tuple, list))):
                    pred = [item.cpu().numpy() for item in pred]
                else:
                    pred = pred.cpu().numpy()
                data['predict'] = pred
                # inverse transform
                for transform in self.transform_list[::-1]:
                    if (transform.inverse):
                        data = transform.inverse_transform_for_prediction(data) 

                pred = data['predict']
                # conver prediction from N, C, H, W to (N*H*W)*C
                print(names, pred.shape, labels.shape)
                pred_2d = np.swapaxes(pred, 1, 2)
                pred_2d = np.swapaxes(pred_2d, 2, 3)
                pred_2d = pred_2d.reshape(-1, class_num) 
                lab = labels.cpu().numpy()
                lab_2d = np.swapaxes(lab, 1, 2)
                lab_2d = np.swapaxes(lab_2d, 2, 3)
                lab_2d = lab_2d.reshape(-1, class_num) 
                pred_list.append(pred_2d)
                gt_list.append(lab_2d)

        pred_cat = np.concatenate(pred_list)
        gt_cat   = np.concatenate(gt_list)
        gt       = np.argmax(gt_cat, axis = 1)
        gt = gt.reshape(-1).astype(np.uint8)
        print(gt.shape, pred_cat.shape)
        conf = get_confident_map(gt, pred_cat)
        conf = conf.reshape(-1, 256, 256).astype(np.uint8) * 255
        save_dir = self.config['testing']['output_dir'] + "_conf"
        for idx in range(len(filename_list)):
            filename = filename_list[idx][0].split('/')[-1]
            conf_map = Image.fromarray(conf[idx])
            dst_path = os.path.join(save_dir, filename)
            conf_map.save(dst_path)

    def run(self):
        self.create_dataset()
        self.create_network()
        self.infer_with_cl()

def main():
    if(len(sys.argv) < 2):
        print('Number of arguments should be 3. e.g.')
        print('   python cl.py config.cfg')
        exit()
    cfg_file = str(sys.argv[1])
    config   = parse_config(cfg_file)
    config   = synchronize_config(config)

    # set dataset
    transform_names = config['dataset']['valid_transform']
    transform_list  = []
    transform_dict  = TransformDict
    if(transform_names is None or len(transform_names) == 0):
        data_transform = None 
    else:
        transform_param = config['dataset']
        transform_param['task'] = 'segmentation' 
        for name in transform_names:
            if(name not in transform_dict):
                raise(ValueError("Undefined transform {0:}".format(name))) 
            one_transform = transform_dict[name](transform_param)
            transform_list.append(one_transform)
        data_transform = transforms.Compose(transform_list)
    print('transform list', transform_list)
    csv_file = config['dataset']['train_csv']
    dataset  = NiftyDataset(root_dir  = config['dataset']['root_dir'],
                            csv_file  = csv_file,
                            modal_num = config['dataset']['modal_num'],
                            with_label= True,
                            transform = data_transform )

    agent = SegmentationAgentwithCL(config, 'test')
    agent.set_datasets(None, None, dataset)
    agent.transform_list = transform_list
    agent.run()

if __name__ == "__main__":
    main()