
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging 
import os
import sys
from pymic.util.parse_config import *
from pymic.net_run_wsl.wsl_em import WSLEntropyMinimization
from pymic.net_run_wsl.wsl_gatedcrf import WSLGatedCRF
from pymic.net_run_wsl.wsl_mumford_shah import WSLMumfordShah
from pymic.net_run_wsl.wsl_tv import WSLTotalVariation
from pymic.net_run_wsl.wsl_ustm import WSLUSTM
from pymic.net_run_wsl.wsl_dmpls import WSLDMPLS

WSLMethodDict = {'EntropyMinimization': WSLEntropyMinimization,
    'GatedCRF': WSLGatedCRF,
    'MumfordShah': WSLMumfordShah,
    'TotalVariation': WSLTotalVariation,
    'USTM': WSLUSTM,
    'DMPLS': WSLDMPLS}

def main():
    """
    The main function for training and inference of weakly supervised segmentation. 
    """
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('   pymic_wsl train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)
    config   = synchronize_config(config)
    log_dir  = config['training']['ckpt_save_dir']
    if(not os.path.exists(log_dir)):
        os.mkdir(log_dir)
    logging.basicConfig(filename=log_dir+"/log_{0:}.txt".format(stage), level=logging.INFO,
                        format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging_config(config)
    wsl_method = config['weakly_supervised_learning']['wsl_method']
    agent = WSLMethodDict[wsl_method](config, stage)
    agent.run()

if __name__ == "__main__":
    main()

    