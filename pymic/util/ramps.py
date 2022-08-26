
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np

"""Functions for ramping hyperparameters up or down

Each function takes the current training step or epoch, and the
ramp length (maximal step or epoch), and returns a multiplier between
0 and 1.
"""


def get_rampup_ratio(i, start, end, mode = "linear"):
    if( i < start):
        rampup = 0.0
    elif(i > end):
        rampup = 1.0
    elif(mode == "linear"):
        rampup = (i - start) / (end - start)
    elif(mode == "sigmoid"):
        phase  = 1.0 - (i - start) / (end - start)
        rampup = float(np.exp(-5.0 * phase * phase))
    return rampup


def cosine_rampdown(i, start, end):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    i = np.clip(i, 0.0, length)
    return float(.5 * (np.cos(np.pi * i / length) + 1))