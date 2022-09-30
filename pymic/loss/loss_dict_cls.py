# -*- coding: utf-8 -*-
"""
Built-in loss functions for classification. 

* CrossEntropyLoss :mod:`pymic.loss.cls.basic.CrossEntropyLoss`
* SigmoidCELoss :mod:`pymic.loss.cls.basic.SigmoidCELoss`
* L1Loss :mod:`pymic.loss.cls.basic.L1Loss`
* MSELoss :mod:`pymic.loss.cls.basic.MSELoss`
* NLLLoss :mod:`pymic.loss.cls.basic.NLLLoss`

"""
from __future__ import print_function, division
from pymic.loss.cls.basic import *

PyMICClsLossDict = {"CrossEntropyLoss": CrossEntropyLoss,
    "SigmoidCELoss": SigmoidCELoss,
    "L1Loss":  L1Loss,
    "MSELoss": MSELoss,
    "NLLLoss": NLLLoss}
