# -*- coding: utf-8 -*-
from __future__ import print_function, division
import copy
import logging
import time
from pymic.net_run.agent_rec import ReconstructionAgent

class SelfSupModelGenesis(ReconstructionAgent):
    """
    Patch swapping-based self-supervised learning. 
    
    Reference: Liang Chen et al., Self-supervised learning for medical image analysis
        using image context restoration, Medical Image Analysis, 2019. 

    A PatchSwaping transform need to be used in the cnfiguration. 

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `self_supervised_learning` is needed. See :doc:`usage.selfsl` for details.

    In the configuration file, it should look like this:
    ```
        [dataset]
        task_type = rec
        supervise_type  = self_sup
        train_transform = [..., ..., PatchSwaping]
        valid_transform = [..., ..., PatchSwaping]
        
        [self_supervised_learning]
        method_name = ModelGenesis

    """
    def __init__(self, config, stage = 'train'):
        super(SelfSupModelGenesis, self).__init__(config, stage)
    
    def get_transform_names_and_parameters(self, stage):
        trans_names, trans_params = super(SelfSupModelGenesis, self).get_transform_names_and_parameters(stage)
        # if(stage == 'train'):
        #     print('training transforms:', trans_names)
        #     if("LocalShuffling" not in trans_names):
        #         raise ValueError("LocalShuffling is required for model genesis, \
        #             but it is not given in training transform")
        #     if("NonLinearTransform" not in trans_names):
        #         raise ValueError("NonLinearTransform is required for model genesis, \
        #             but it is not given in training transform")
        #     if("InOutPainting" not in trans_names):
        #         raise ValueError("InOutPainting is required for model genesis, \
        #             but it is not given in training transform")                   
        return trans_names, trans_params
