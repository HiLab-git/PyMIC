
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np

"""Functions for ramping hyperparameters up or down

Each function takes the current training step or epoch, and the
ramp length (maximal step or epoch), and returns a multiplier between
0 and 1.
"""

def sigmoid_rampup(i, length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if length == 0:
        return 1.0
    else:
        i = np.clip(i, 0.0, length)
        phase = 1.0 - (i + 0.0) / length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(i, length):
    """Linear rampup"""
    assert i >= 0 and length >= 0
    i = np.clip(i, 0.0, length)
    return (i + 0.0) / length


def cosine_rampdown(i, length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    i = np.clip(i, 0.0, length)
    return float(.5 * (np.cos(np.pi * i / length) + 1))