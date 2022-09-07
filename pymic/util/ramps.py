
# -*- coding: utf-8 -*-
"""
Functions for ramping hyperparameters up or down.

Each function takes the current training step or epoch, and the
ramp length (start and end step or epoch), and returns a multiplier between
0 and 1.
"""
from __future__ import print_function, division
import numpy as np

def get_rampup_ratio(i, start, end, mode = "linear"):
    """
    Obtain the rampup ratio.

    :param i: (int) The current iteration.
    :param start: (int) The start iteration.
    :param end: (int) The end itertation.
    :param mode: (str) Valid values are {`linear`, `sigmoid`, `cosine`}.
    """
    i = np.clip(i, start, end)
    if(mode == "linear"):
        rampup = (i - start) / (end - start)
    elif(mode == "sigmoid"):
        phase  = 1.0 - (i - start) / (end - start)
        rampup = float(np.exp(-5.0 * phase * phase))
    elif(mode == "cosine"):
        phase  = 1.0 - (i - start) / (end - start)
        rampup = float(.5 * (np.cos(np.pi * phase) + 1))
    else:
        raise ValueError("Undefined rampup mode {0:}".format(mode))
    return rampup


def get_rampdown_ratio(i, start, end, mode = "linear"):
    """
    Obtain the rampdown ratio.

    :param i: (int) The current iteration.
    :param start: (int) The start iteration.
    :param end: (int) The end itertation.
    :param mode: (str) Valid values are {`linear`, `sigmoid`, `cosine`}.
    """
    i = np.clip(i, start, end)
    if(mode == "linear"):
        rampdown = 1.0 - (i - start) / (end - start)
    elif(mode == "sigmoid"):
        phase = (i - start) / (end - start)
        rampdown = float(np.exp(-5.0 * phase * phase))
    elif(mode == "cosine"):
        phase = (i - start) / (end - start)
        rampdown = float(.5 * (np.cos(np.pi * phase) + 1))
    else:
        raise ValueError("Undefined rampup mode {0:}".format(mode))
    return rampdown

    