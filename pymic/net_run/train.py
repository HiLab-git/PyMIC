# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import os
import sys
import shutil
from datetime import datetime
from pymic.util.parse_config import *
from pymic.net_run.agent_cls import ClassificationAgent
from pymic.net_run.agent_seg import SegmentationAgent
from pymic.net_run.semi_sup import SSLMethodDict
from pymic.net_run.weak_sup import WSLMethodDict
from pymic.net_run.noisy_label import NLLMethodDict
from pymic.net_run.self_sup import SelfSLSegAgent

def get_segmentation_agent(config, sup_type):
    assert(sup_type in ['fully_sup', 'semi_sup', 'self_sup', 'weak_sup', 'noisy_label'])
    if(sup_type == 'fully_sup'):
        logging.info("\n********** Fully Supervised Learning **********\n")
        agent = SegmentationAgent(config, 'train')
    elif(sup_type == 'semi_sup'):
        logging.info("\n********** Semi Supervised Learning **********\n")
        method = config['semi_supervised_learning']['method_name']
        agent = SSLMethodDict[method](config, 'train')
    elif(sup_type == 'weak_sup'):
        logging.info("\n********** Weakly Supervised Learning **********\n")
        method = config['weakly_supervised_learning']['method_name']
        agent = WSLMethodDict[method](config, 'train')
    elif(sup_type == 'noisy_label'):
        logging.info("\n********** Noisy Label Learning **********\n")
        method = config['noisy_label_learning']['method_name']
        agent = NLLMethodDict[method](config, 'train')
    elif(sup_type == 'self_sup'):
        logging.info("\n********** Self Supervised Learning **********\n")
        method = config['self_supervised_learning']['method_name']
        if(method == "custom"):
            pass
        elif(method == "model_genesis"):
            transforms = ['RandomFlip', 'LocalShuffling', 'NonLinearTransform', 'InOutPainting']
            genesis_cfg = {
                'randomflip_flip_depth': True,
                'randomflip_flip_height': True,
                'randomflip_flip_width': True,
                'localshuffling_probability': 0.5,
                'nonLineartransform_probability': 0.9,
                'inoutpainting_probability': 0.9,
                'inpainting_probability': 0.2
            }
            config['dataset']['train_transform'].extend(transforms)
            config['dataset']['valid_transform'].extend(transforms)
            config['dataset'].update(genesis_cfg)
            logging_config(config['dataset'])
        else:
            raise ValueError("The specified method {0:} is not implemented. ".format(method) + \
                         "Consider to set `self_sl_method = custom` and use customized" + \
                         " transforms for self-supervised learning.")
        agent = SelfSLSegAgent(config, 'train')
    else:
        raise ValueError("undefined supervision type: {0:}".format(sup_type))
    return agent

def main():
    """
    The main function for running a network for training.
    """
    if(len(sys.argv) < 2):
        print('Number of arguments should be 2. e.g.')
        print('   pymic_train config.cfg')
        exit()
    cfg_file = str(sys.argv[1])
    if(not os.path.isfile(cfg_file)):
        raise ValueError("The config file does not exist: " + cfg_file)
    config   = parse_config(cfg_file)
    config   = synchronize_config(config)
    log_dir  = config['training']['ckpt_save_dir']
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir, exist_ok=True)
    dst_cfg = cfg_file if "/" not in cfg_file else cfg_file.split("/")[-1]
    shutil.copy(cfg_file, log_dir + "/" + dst_cfg)
    if sys.version.startswith("3.9"):
        logging.basicConfig(filename=log_dir+"/log_train_{0:}.txt".format(str(datetime.now())[:-7]), 
                            level=logging.INFO, format='%(message)s', force=True) # for python 3.9
    else:
        logging.basicConfig(filename=log_dir+"/log_train_{0:}.txt".format(str(datetime.now())[:-7]), 
                            level=logging.INFO, format='%(message)s') # for python 3.6
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging_config(config)
    task     = config['dataset']['task_type']
    assert task in ['cls', 'cls_nexcl', 'seg']
    if(task == 'cls' or task == 'cls_nexcl'):
        agent = ClassificationAgent(config, 'train')
    else:
        sup_type = config['dataset'].get('supervise_type', 'fully_sup')
        agent = get_segmentation_agent(config, sup_type)
    agent.run()

if __name__ == "__main__":
    main()
    

