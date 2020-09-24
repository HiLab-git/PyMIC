# -*- coding: utf-8 -*-
from __future__ import print_function, division

class AbstractTransform(object):
    def __init__(self, params):
        self.task = params['Task'.lower()]

    def __call__(self, sample):
        return sample

    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))
