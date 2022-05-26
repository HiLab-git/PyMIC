
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging 
import os
import sys
from pymic.util.parse_config import *
from pymic.net_run_wsl.wsl_em import WSL_EntropyMinimization
from pymic.net_run_wsl.wsl_tv import WSL_TotalVariation
from pymic.net_run_wsl.wsl_ustm import WSL_USTM
from pymic.net_run_wsl.wsl_dmpls import WSL_DMPLS

WSLMethodDict = {'EntropyMinimization': WSL_EntropyMinimization,
    'TotalVariation': WSL_TotalVariation,
    'USTM': WSL_USTM,
    'DMPLS': WSL_DMPLS}

def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('   pymic_ssl train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)
    config   = synchronize_config(config)
    log_dir  = config['training']['ckpt_save_dir']
    if(not os.path.exists(log_dir)):
        os.mkdir(log_dir)
    logging.basicConfig(filename=log_dir+"/log.txt", level=logging.INFO,
                        format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging_config(config)
    wsl_method = config['weakly_supervised_learning']['wsl_method']
    agent = WSLMethodDict[wsl_method](config, stage)
    agent.run()

if __name__ == "__main__":
    main()

    