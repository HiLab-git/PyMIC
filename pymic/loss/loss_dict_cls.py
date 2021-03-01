# -*- coding: utf-8 -*-
from __future__ import print_function, division
from pymic.loss.cls.ce import CrossEntropyLoss
from pymic.loss.cls.l1 import L1Loss
from pymic.loss.cls.nll import NLLLoss
from pymic.loss.cls.mse import MSELoss

PyMICClsLossDict = {'CrossEntropyLoss': CrossEntropyLoss,
    "L1Loss": L1Loss,
    "MSELoss": MSELoss,
    "NLLLoss": NLLLoss}
