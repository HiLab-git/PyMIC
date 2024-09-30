# -*- coding: utf-8 -*-
from __future__ import print_function, division
from pymic.net_run.agent_seg import SegmentationAgent


class SelfSupVolumeFusion(SegmentationAgent):
    """
    Abstract class for self-supervised segmentation.

    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`. 

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an 
        extra section `semi_supervised_learning` is needed. See :doc:`usage.ssl` for details.
    """
    def __init__(self, config, stage = 'train'):
        super(SelfSupVolumeFusion, self).__init__(config, stage)

    def get_transform_names_and_parameters(self, stage):
        trans_names, trans_params = super(SelfSupVolumeFusion, self).get_transform_names_and_parameters(stage)
        if(stage == 'train'):
            print('training transforms:', trans_names)
            if("Crop4VolumeFusion" not in trans_names):
                raise ValueError("Crop4VolumeFusion is required for VolF, \
                    but it is not given in training transform")
            if("VolumeFusion" not in trans_names):
                raise ValueError("VolumeFusion is required for VolF, \
                    but it is not given in training transform")
            if("LabelToProbability" not in trans_names):
                raise ValueError("LabelToProbability is required for VolF, \
                    but it is not given in training transform")                   
        return trans_names, trans_params
 
    