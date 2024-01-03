# -*- coding: utf-8 -*-
from __future__ import print_function, division
import copy
import logging
import time
from pymic.net_run.agent_rec import ReconstructionAgent


        
class SelfSLSegAgent(ReconstructionAgent):
    """
    Abstract class for self-supervised segmentation.

    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `self_supervised_learning` is needed. See :doc:`usage.selfsl` for details.
    """
    def __init__(self, config, stage = 'train'):
        super(SelfSLSegAgent, self).__init__(config, stage)
 