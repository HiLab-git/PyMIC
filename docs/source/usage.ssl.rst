.. _semi_supervised_learning:

Semi-Supervised Learning
=========================

SSL Configurations
------------------

In the configuration file for semi-supervised segmentation, in addition to those used in fully 
supervised learning, there are some items specificalized for semi-supervised learning.

Users should provide values for the following items in ``dataset`` section of 
the configuration file:

* ``supervise_type`` (string): The value should be "`semi_sup`".  

* ``train_csv_unlab`` (string): the csv file for unlabeled dataset. 
  Note that ``train_csv`` is only used for labeled dataset.  

* ``train_batch_size_unlab`` (int): the batch size for unlabeled dataset. 
  Note that  `train_batch_size` means the batch size for the labeled dataset. 

* ``train_transform_unlab`` (list): a list of transforms used for unlabeled data. 


The following is an example of the ``dataset`` section for semi-supervised learning:

.. code-block:: none

    ...

    tensor_type    = float
    task_type      = seg
    supervise_type = semi_sup

    root_dir  = ../../PyMIC_data/ACDC/preprocess/
    train_csv = config/data/image_train_r10_lab.csv
    train_csv_unlab = config/data/image_train_r10_unlab.csv
    valid_csv = config/data/image_valid.csv
    test_csv  = config/data/image_test.csv

    train_batch_size = 4
    train_batch_size_unlab = 4

    # data transforms
    train_transform = [Pad, RandomRotate, RandomCrop, RandomFlip, NormalizeWithMeanStd, GammaCorrection, GaussianNoise, LabelToProbability]
    train_transform_unlab = [Pad, RandomRotate, RandomCrop, RandomFlip, NormalizeWithMeanStd, GammaCorrection, GaussianNoise]
    valid_transform       = [NormalizeWithMeanStd, Pad, LabelToProbability]
    test_transform        = [NormalizeWithMeanStd, Pad]
    ...

In addition, there is a ``semi_supervised_learning`` section that is specifically designed
for SSL methods. In that section, users need to specify the ``method_name`` and configurations
related to the SSL method. For example, the correspoinding configuration for CPS is:

.. code-block:: none

    ...
    [semi_supervised_learning]
    method_name    = CPS
    regularize_w   = 0.1
    rampup_start   = 1000
    rampup_end     = 20000
    ...

.. note::

   The configuration items vary with different SSL methods. Please refer to the API 
   of each built-in SSL method for details of the correspoinding configuration. 
   See  examples in `PyMIC_examples/seg_ssl/ <https://github.com/HiLab-git/PyMIC_examples/tree/main/seg_ssl/>`_.

Built-in SSL Methods
--------------------

:mod:`pymic.net_run.semi_sup.ssl_abstract.SSLSegAgent` is the abstract class used for 
semi-supervised learning. The built-in SSL methods are child classes of  `SSLSegAgent`.
The available SSL methods implemnted in PyMIC are listed in `pymic.net_run.semi_sup.SSLMethodDict`, 
and they are:

* ``EntropyMinimization``: (`NeurIPS 2005 <https://papers.nips.cc/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf>`_)
  Using entorpy minimization to regularize unannotated samples.

* ``MeanTeacher``: (`NeurIPS 2017 <https://arxiv.org/abs/1703.01780>`_) Use self-ensembling mean teacher to supervise the student model on
  unannotated samples. 

* ``UAMT``: (`MICCAI 2019 <https://arxiv.org/abs/1907.07034>`_) Uncertainty aware mean teacher. 

* ``CCT``: (`CVPR 2020 <https://arxiv.org/abs/2003.09005>`_) Cross-consistency training.

* ``CPS``: (`CVPR 2021 <https://arxiv.org/abs/2106.01226>`_) Cross-pseudo supervision.

* ``URPC``: (`MIA 2022 <https://doi.org/10.1016/j.media.2022.102517>`_) Uncertainty rectified pyramid consistency.

Customized SSL Methods
----------------------

PyMIC alo supports customizing SSL methods by inheriting the  `SSLSegAgent` class. 
You may only need to rewrite the  `training()` method and reuse most part of the 
existing pipeline, such as data loading, validation and inference methods. For example:

.. code-block:: none

    from pymic.net_run.semi_sup import SSLSegAgent

    class MySSLMethod(SSLSegAgent):
      def __init__(self, config, stage = 'train'):
          super(MySSLMethod, self).__init__(config, stage)
          ...
        
      def training(self):
          ...
    
    agent = MySSLMethod(config, stage)
    agent.run()

You may need to check the source code of built-in SSL methods to be more familar with 
how to implement your own SSL method. 