.. _weakly_supervised_learning:

Weakly-Supervised Learning
==========================

pymic_wsl
---------

:mod:`pymic_wsl` is the command for using built-in weakly-supervised methods for training. 
Similarly to :mod:`pymic_run`, it should be followed by two parameters, specifying the 
stage and configuration file, respectively. The training and testing commands are:

.. code-block:: bash

    pymic_wsl train myconfig_wsl.cfg
    pymic_wsl test  myconfig_wsl.cfg

.. tip::

   If the WSL method only involves one network, either ``pymic_wsl`` or  ``pymic_run``
   can be used for inference. Their difference only exists in the training stage. 

.. note::

   Currently, the weakly supervised methods supported by PyMIC are only for learning 
   from partial annotations, such scribble-based annotation. Learning from image-level 
   or point annotations may involve several training stages and will be considered 
   in the future. 


WSL Configurations
------------------

In the configuration file for ``pymic_wsl``, in addition to those used in fully 
supervised learning, there are some items specificalized for weakly-supervised learning.

First, in the :mod:`train_transform` list, a special transform named :mod:`PartialLabelToProbability`
should be used to transform patial labels into a one-hot probability map and a weighting 
map of pixels (i.e., the weight of a pixel is 1 if labeled and 0 otherwise). The patial
cross entropy loss on labeled pixels is actually implemented by a weighted cross entropy loss.
The loss setting is `loss_type = CrossEntropyLoss`.

Second, there is a ``weakly_supervised_learning`` section that is specifically designed
for WSL methods. In that section, users need to specify the ``wsl_method`` and configurations
related to the WSL method. For example, the correspoinding configuration for GatedCRF is:



.. code-block:: none

    [dataset]
    ...
    root_dir  = ../../PyMIC_data/ACDC/preprocess
    train_csv = config/data/image_train.csv
    valid_csv = config/data/image_valid.csv
    test_csv  = config/data/image_test.csv

    train_batch_size = 4

    # data transforms
    train_transform = [Pad, RandomCrop, RandomFlip, NormalizeWithMeanStd, PartialLabelToProbability]
    valid_transform = [NormalizeWithMeanStd, Pad, LabelToProbability]
    test_transform  = [NormalizeWithMeanStd, Pad]
    ...

    [network]
    ...

    [training]
    ...
    loss_type     = CrossEntropyLoss
    ...

    [weakly_supervised_learning]
    wsl_method     = GatedCRF
    regularize_w   = 0.1
    rampup_start   = 2000
    rampup_end     = 15000
    GatedCRFLoss_W0     = 1.0
    GatedCRFLoss_XY0    = 5
    GatedCRFLoss_rgb    = 0.1
    GatedCRFLoss_W1     = 1.0
    GatedCRFLoss_XY1    = 3
    GatedCRFLoss_Radius = 5

    [testing]
    ...

.. note::

   The configuration items vary with different WSL methods. Please refer to the API 
   of each built-in WSL method for details of the correspoinding configuration.  

Built-in WSL Methods
--------------------

:mod:`pymic.net_run_wsl.wsl_abstract.WSLSegAgent` is the abstract class used for 
weakly-supervised learning. The built-in WSL methods are child classes of  :mod:`WSLSegAgent`.
The available WSL methods implemnted in PyMIC are listed in :mod:`pymic.net_run_wsl.wsl_main.WSLMethodDict`, 
and they are:

* ``EntropyMinimization``: (`NeurIPS 2005 <https://papers.nips.cc/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf>`_)
  Using entorpy minimization to regularize unannotated pixels.

* ``GatedCRF``: (`arXiv 2019 <http://arxiv.org/abs/1906.04651>`_) 
  Use gated CRF to regularize unannotated pixels. 

* ``TotalVariation``: (`arXiv 2022 <https://arxiv.org/abs/2111.02403>`_) 
  Use Total Variation to regularize unannotated pixels. 

* ``MumfordShah``: (`TIP 2020 <https://doi.org/10.1109/TIP.2019.2941265>`_) 
  Use Mumford Shah loss to regularize unannotated pixels. 

* ``USTM``: (`PR 2022 <https://doi.org/10.1016/j.patcog.2021.108341>`_) 
  Adapt USTM with transform-consistency regularization.

* ``DMPLS``: (`MICCAI 2022 <https://arxiv.org/abs/2203.02106>`_) 
  Dynamically mixed pseudo label supervision

Customized WSL Methods
----------------------

PyMIC alo supports customizing WSL methods by inheriting the :mod:`WSLSegAgent` class. 
You may only need to rewrite the :mod:`training()` method and reuse most part of the 
existing pipeline, such as data loading, validation and inference methods. For example:

.. code-block:: none

    from pymic.net_run_wsl.wsl_abstract import WSLSegAgent

    class MyWSLMethod(WSLSegAgent):
      def __init__(self, config, stage = 'train'):
          super(MyWSLMethod, self).__init__(config, stage)
          ...
        
      def training(self):
          ...
    
    agent = MyWSLMethod(config, stage)
    agent.run()

You may need to check the source code of built-in WSL methods to be more familar with 
how to implement your own WSL method. 