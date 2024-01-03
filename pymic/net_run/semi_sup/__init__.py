from __future__ import absolute_import
from pymic.net_run.semi_sup.ssl_abstract import SSLSegAgent
from pymic.net_run.semi_sup.ssl_em import SSLEntropyMinimization
from pymic.net_run.semi_sup.ssl_mt import SSLMeanTeacher
from pymic.net_run.semi_sup.ssl_mcnet import SSLMCNet
from pymic.net_run.semi_sup.ssl_uamt import SSLUncertaintyAwareMeanTeacher
from pymic.net_run.semi_sup.ssl_cct import SSLCCT
from pymic.net_run.semi_sup.ssl_cps import SSLCPS
from pymic.net_run.semi_sup.ssl_urpc import SSLURPC


SSLMethodDict = {'EntropyMinimization': SSLEntropyMinimization,
    'MeanTeacher': SSLMeanTeacher,
    'MCNet': SSLMCNet,
    'UAMT': SSLUncertaintyAwareMeanTeacher,
    'CCT': SSLCCT,
    'CPS': SSLCPS,
    'URPC': SSLURPC}