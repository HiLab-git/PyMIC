from __future__ import absolute_import
from pymic.net_run.weak_sup.wsl_abstract import WSLSegAgent
from pymic.net_run.weak_sup.wsl_em import WSLEntropyMinimization
from pymic.net_run.weak_sup.wsl_gatedcrf import WSLGatedCRF
from pymic.net_run.weak_sup.wsl_mumford_shah import WSLMumfordShah
from pymic.net_run.weak_sup.wsl_tv import WSLTotalVariation
from pymic.net_run.weak_sup.wsl_ustm import WSLUSTM
from pymic.net_run.weak_sup.wsl_dmpls import WSLDMPLS

WSLMethodDict = {'EntropyMinimization': WSLEntropyMinimization,
    'GatedCRF': WSLGatedCRF,
    'MumfordShah': WSLMumfordShah,
    'TotalVariation': WSLTotalVariation,
    'USTM': WSLUSTM,
    'DMPLS': WSLDMPLS}