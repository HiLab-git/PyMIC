# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import os
import sys
import shutil
from pymic.util.parse_config import *
from pymic.net_run.agent_cls import ClassificationAgent
from pymic.net_run.agent_seg import SegmentationAgent

def main():
    """
    The main function for running a network for training or inference.
    """
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('   pymic_run train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)
    config   = synchronize_config(config)
    log_dir  = config['training']['ckpt_save_dir']
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir, exist_ok=True)
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
    task     = config['dataset']['task_type']
    assert task in ['cls', 'cls_nexcl', 'seg']
    if(task == 'cls' or task == 'cls_nexcl'):
        agent = ClassificationAgent(config, stage)
    else:
        agent = SegmentationAgent(config, stage)
    agent.run()

if __name__ == "__main__":
    main()
    

