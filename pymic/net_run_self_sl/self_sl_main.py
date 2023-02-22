
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging 
import os
import sys
import shutil
from pymic.util.parse_config import *
from pymic.net_run_self_sl.self_sl_agent import SelfSLSegAgent

def model_genesis(stage, cfg_file):
    config  = parse_config(cfg_file)
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

    config   = synchronize_config(config)
    log_dir  = config['training']['ckpt_save_dir']
    if(not os.path.exists(log_dir)):
        os.mkdir(log_dir)
    if(stage == "train"):
        dst_cfg = cfg_file if "/" not in cfg_file else cfg_file.split("/")[-1]
        shutil.copy(cfg_file, log_dir + "/" + dst_cfg)
    if sys.version.startswith("3.9"):
        logging.basicConfig(filename=log_dir+"/log_{0:}.txt".format(stage), level=logging.INFO,
                            format='%(message)s', force=True) # for python 3.9
    else:
        logging.basicConfig(filename=log_dir+"/log_{0:}.txt".format(stage), level=logging.INFO,
                            format='%(message)s') # for python 3.6
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging_config(config)
    agent = SelfSLSegAgent(config, stage)
    agent.run()

def default_self_sl(stage, cfg_file):
    config   = parse_config(cfg_file)
    config   = synchronize_config(config)
    log_dir  = config['training']['ckpt_save_dir']
    if(not os.path.exists(log_dir)):
        os.mkdir(log_dir)
    if(stage == "train"):
        dst_cfg = cfg_file if "/" not in cfg_file else cfg_file.split("/")[-1]
        shutil.copy(cfg_file, log_dir + "/" + dst_cfg)
    if sys.version.startswith("3.9"):
        logging.basicConfig(filename=log_dir+"/log_{0:}.txt".format(stage), level=logging.INFO,
                            format='%(message)s', force=True) # for python 3.9
    else:
        logging.basicConfig(filename=log_dir+"/log_{0:}.txt".format(stage), level=logging.INFO,
                            format='%(message)s') # for python 3.6
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging_config(config)
    agent = SelfSLSegAgent(config, stage)
    agent.run()

   
if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('   pymic_self_sl train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)
    method   = "default"
    if 'self_supervised_learning' in config:
        method = config['self_supervised_learning'].get('self_sl_method', 'default')
    print("the self supervised method is ", method)
    if(method == "default"):
        default_self_sl(stage, cfg_file)
    elif(method == 'model_genesis'):
        model_genesis(stage, cfg_file)
    else:
        raise ValueError("The specified method {0:} is not implemented. ".format(method) + \
                         "Consider to set `self_sl_method = default` and use customized" + \
                         " transforms for self-supervised learning.")

    