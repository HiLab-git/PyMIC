.. _semi_supervised_learning:

Semi-Supervised Learning
=========================

pymic_ssl
---------

:mod:`pymic_ssl` is the command for using built-in semi-supervised methods for training. 
Similarly to :mod:`pymic_run`, it should be followed by two parameters, specifying the 
stage and configuration files. The training and testing commands are:

.. code-block:: bash

    pymic_ssl train myconfig_ssl.cfg
    pymic_ssl test  myconfig_ssl.cfg

.. tip::

   If the SSL method only involves one network, either ``pymic_ssl`` or  ``pymic_run``
   can be used for inference. Their difference only exists in the training stage. 

SSL Configurations
------------------

In the configuration file for ``pymic_ssl``, in addition to those used in fully 
supervised learning, there are some items specified for semi-supervised learning.

Users should provide values for the following items in ``dataset`` section of 
the configuration file:

``train_csv_unlab`` (string): the csv file for unlabeled dataset. Note that ``train_csv`` 
is only used for labeled dataset.  

``train_batch_size_unlab`` (int): the batch size for unlabeled dataset. Note that 
``train_batch_size`` means the batch size for the labeled dataset. 

``train_transform_unlab`` (list): a list of transforms used for unlabeled data. 


The following is an example of the ``dataset`` section for semi-supervised learning:

.. code-block:: none

    ...
    root_dir  =../../PyMIC_data/ACDC/preprocess/
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
for SSL methods. In that section, users need to specify the ``ssl_method`` and configurations
related to the SSL method. For example, the correspoinding configuration for CPS is:

.. code-block:: none

    ...
    [semi_supervised_learning]
    ssl_method     = CPS
    regularize_w   = 0.1
    rampup_start   = 1000
    rampup_end     = 20000
    ...

:mod:`pymic.net_run_ssl.ssl_abstract.SSLSegAgent` is the abstract class used for 
semi-supervised learning. The reccesponding 
