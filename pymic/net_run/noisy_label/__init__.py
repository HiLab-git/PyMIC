from __future__ import absolute_import
from pymic.net_run.noisy_label.nll_co_teaching import NLLCoTeaching
from pymic.net_run.noisy_label.nll_trinet import NLLTriNet
from pymic.net_run.noisy_label.nll_dast import NLLDAST

NLLMethodDict = {'CoTeaching': NLLCoTeaching,
    "TriNet": NLLTriNet,
    "DAST": NLLDAST}