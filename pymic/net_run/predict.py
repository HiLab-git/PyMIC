# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import logging
import os
import sys
from datetime import datetime
from pymic import TaskType
from pymic.util.parse_config import *
from pymic.net_run.agent_cls import ClassificationAgent
from pymic.net_run.agent_seg import SegmentationAgent
from pymic.net_run.agent_rec import ReconstructionAgent

def main():
    """
    The main function for running a network for inference.
    """
    if(len(sys.argv) < 2):
        print('Number of arguments should be at least 2. e.g.')
        print('   pymic_test config.cfg -test_csv train.csv -output_dir result_dir -ckpt_mode 1')
        exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="configuration file for testing")
    parser.add_argument("--test_csv", help="the csv file for testing images", 
                    required=False, default=None)
    parser.add_argument("--test_dir", help="the dir for testing images", 
                    required=False, default=None)
    parser.add_argument("--output_dir", help="the output dir for inference results", 
                    required=False, default=None)
    parser.add_argument("--ckpt_dir", help="the dir for trained model", 
                    required=False, default=None)
    parser.add_argument("--ckpt_mode", help="the mode for chekpoint: 0-latest, 1-best, 2-customized", 
                    required=False, default=None)
    parser.add_argument("--ckpt_name", help="the name chekpoint if ckpt_mode = 2", 
                    required=False, default=None)
    parser.add_argument("--gpus", help="the gpus for runing, e.g., [0]", 
                    required=False, default=None)
    args = parser.parse_args()
    if(not os.path.isfile(args.cfg)):
        raise ValueError("The config file does not exist: " + args.cfg)
    config   = parse_config(args)
    config   = synchronize_config(config)
    print(config)
    log_dir  = config['testing']['output_dir']
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir, exist_ok=True)

    if sys.version.startswith("3.9"):
        logging.basicConfig(filename=log_dir+"/log_test.txt", 
                            level=logging.INFO, format='%(message)s', force=True) # for python 3.9
    else:
        logging.basicConfig(filename=log_dir+"/log_test.txt", 
                            level=logging.INFO, format='%(message)s') # for python 3.6
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    dst_cfg = args.cfg if "/" not in args.cfg else args.cfg.split("/")[-1]
    wrtie_config(config, log_dir + "/" + dst_cfg)
    task    = config['dataset']['task_type']
    if(task == TaskType.CLASSIFICATION_ONE_HOT or task == TaskType.CLASSIFICATION_COEXIST):
        agent = ClassificationAgent(config, 'test')
    elif(task == TaskType.SEGMENTATION):
        agent = SegmentationAgent(config, 'test')
    elif(task == TaskType.RECONSTRUCTION):
        agent = ReconstructionAgent(config, 'test')
    else:
        raise ValueError("Undefined task for inference: {0:}".format(task))
    agent.run()

if __name__ == "__main__":
    main()
    

