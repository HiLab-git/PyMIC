# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import os
import sys
from datetime import datetime
from pymic.util.parse_config import *
from pymic.net_run.agent_preprocess import PreprocessAgent


def main():
    """
    The main function for data preprocessing.
    """
    if(len(sys.argv) < 2):
        print('Number of arguments should be 2. e.g.')
        print('   pymic_preprocess config.cfg')
        exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="configuration file for preprocessing")
    args = parser.parse_args()
    if(not os.path.isfile(args.cfg)):
        raise ValueError("The config file does not exist: " + args.cfg)
    config   = parse_config(args)
    config   = synchronize_config(config)
    agent  = PreprocessAgent(config)
    agent.run()

if __name__ == "__main__":
    main()
    
    

