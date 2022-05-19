
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import sys
from pymic.util.parse_config import parse_config
from pymic.net_run_ssl.ssl_em import SSLSegAgent
from pymic.net_run_ssl.ssl_mt import SSLMeanTeacher
from pymic.net_run_ssl.ssl_uamt import SSLUncertaintyAwareMeanTeacher
from pymic.net_run_ssl.ssl_urpc import SSLURPC
from pymic.net_run_ssl.ssl_cps import SSLCrossPseudoSupervision

SSLMethodDict = {'EntropyMinimization': SSLSegAgent,
    'MeanTeacher': SSLMeanTeacher,
    'UAMT': SSLUncertaintyAwareMeanTeacher,
    'URPC': SSLURPC,
    'CPS': SSLCrossPseudoSupervision}

def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('   pymic_ssl train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)
    ssl_method = config['semi_supervised_learning']['ssl_method']
    agent = SSLMethodDict[ssl_method](config, stage)
    agent.run()

if __name__ == "__main__":
    main()

    