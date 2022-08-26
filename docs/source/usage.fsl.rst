.. _fully_supervised_learning:

Fully Supervised Learning
=========================

SegmentationAgent
-----------------

:mod:`pymic.net_run.agent_seg.SegmentationAgent` is the general class used for training 
and inference of deep learning models. You just need to specify a configuration file to 
initialize an instance of that class. An example code to use it is:

.. code-block:: none

   from pymic.util.parse_config import *

   config_name = "a_config_file.cfg"
   config   = parse_config(config_name)
   config   = synchronize_config(config)
   stage    = "train"  # or "test"
   agent    = SegmentationAgent(config, stage)
   agent.run()

The above code will use the dataset, network and loss function, etc specifcied in the 
configuration file for running. 

.. tip::

   If you use the built-in modules such as ``UNet`` and ``Dice`` + ``CrossEntropy`` loss 
   for segmentation, you don't need to write the above code. Just just use the `pymic_run`
   command. 