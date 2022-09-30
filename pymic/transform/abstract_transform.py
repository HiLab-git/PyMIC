# -*- coding: utf-8 -*-
from __future__ import print_function, division

class AbstractTransform(object):
    """
    The abstract class for Transform.
    """
    def __init__(self, params):
        self.task = params['Task'.lower()]

    def __call__(self, sample):
        """
        Forward pass of the transform. 

        :arg sample: (dict) A dictionary for the input sample obtained by dataloader.
        """
        return sample

    def inverse_transform_for_prediction(self, sample):
        """
        Inverse transform for the sample dictionary.
        Especially, it will update sample['predict'] obtained by a network's
        prediction based on the inverse transform. This function is only useful for spatial transforms.
        """
        raise(ValueError("not implemented"))
