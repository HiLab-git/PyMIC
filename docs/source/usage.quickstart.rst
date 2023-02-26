.. _quickstart:

Quick Start
===========


Train and Test
--------------

PyMIC accepts a configuration file for runing. For example, to train a network
for segmentation with full supervision, run the fullowing command:

.. code-block:: bash

    pymic_train myconfig.cfg 

After training, run the following command for testing:

.. code-block:: bash

    pymic_test myconfig.cfg

.. tip::

   We provide several examples in `PyMIC_examples. <https://github.com/HiLab-git/PyMIC_examples/>`_
   Please run these examples to quickly start with using PyMIC. 
   

.. _configuration:

Configuration File
------------------

PyMIC uses configuration files to specify the setting and parameters of a deep 
learning pipeline, so that users can reuse the code and minimize their workload.
Users can use configuration files to config almost all the componets involved, 
such as dataset, network structure, loss function, optimizer, learning rate 
scheduler and post processing methods, etc. 

.. note::

   Genreally, the configuration file have four sections: ``dataset``, ``network``, 
   ``training`` and ``testing``. 

The following is an example configuration
file used for segmentation of lung from radiograph, which can be find in 
`PyMIC_examples/segmentation/JSRT. <https://github.com/HiLab-git/PyMIC_examples/tree/main/segmentation/JSRT>`_

.. code-block:: none

   [dataset]
   # tensor type (float or double)
   tensor_type = float

   task_type = seg
   root_dir  = ../../PyMIC_data/JSRT
   train_csv = config/jsrt_train.csv
   valid_csv = config/jsrt_valid.csv
   test_csv  = config/jsrt_test.csv

   train_batch_size = 4

   # data transforms
   train_transform = [NormalizeWithMeanStd, RandomCrop, LabelConvert, LabelToProbability]
   valid_transform = [NormalizeWithMeanStd, LabelConvert, LabelToProbability]
   test_transform  = [NormalizeWithMeanStd]

   NormalizeWithMeanStd_channels = [0]
   RandomCrop_output_size = [240, 240]

   LabelConvert_source_list = [0, 255]
   LabelConvert_target_list = [0, 1]


   [network]
   # this section gives parameters for network
   # the keys may be different for different networks

   # type of network
   net_type = UNet2D

   # number of class, required for segmentation task
   class_num     = 2
   in_chns       = 1
   feature_chns  = [16, 32, 64, 128, 256]
   dropout       = [0,  0,  0.3, 0.4, 0.5]
   bilinear      = False
   multiscale_pred = False

   [training]
   # list of gpus
   gpus = [0]

   loss_type     = DiceLoss

   # for optimizers
   optimizer     = Adam
   learning_rate = 1e-3
   momentum      = 0.9
   weight_decay  = 1e-5

   # for lr scheduler (MultiStepLR)
   lr_scheduler  = MultiStepLR
   lr_gamma      = 0.5
   lr_milestones = [2000, 4000, 6000]

   ckpt_save_dir    = model/unet
   ckpt_prefix = unet

   # start iter
   iter_start = 0
   iter_max   = 8000
   iter_valid = 200
   iter_save  = 8000

   [testing]
   # list of gpus
   gpus       = [0]

   # checkpoint mode can be [0-latest, 1-best, 2-specified]
   ckpt_mode         = 0
   output_dir        = result/unet

   # convert the label of prediction output
   label_source = [0, 1]
   label_target = [0, 255]


Evaluation
----------

To evaluate a model's prediction results compared with the ground truth, 
use the ``pymic_eval_seg`` and  ``pymic_eval_cls`` commands for segmentation
and classfication tasks, respectively. Both of them accept a configuration 
file to specify the evaluation metrics, predicted results, ground truth and
other information. 

For example, for segmentation tasks, run:

.. code-block:: none

   pymic_eval_seg evaluation.cfg 

The configuration file is like (an example from 
`PyMIC_examples/seg_ssl/ACDC <https://github.com/HiLab-git/PyMIC_examples/tree/main/seg_ssl/ACDC>`_):

.. code-block:: none

   [evaluation]
   metric_list = [dice, hd95]
   label_list = [1,2,3]
   organ_name = heart

   ground_truth_folder_root  = ../../PyMIC_data/ACDC/preprocess
   segmentation_folder_root  = result/unet2d_urpc
   evaluation_image_pair     = config/data/image_test_gt_seg.csv

See :mod:`pymic.util.evaluation_seg.evaluation` for details of the configuration required.

For classification tasks, run:

.. code-block:: none

   pymic_eval_cls evaluation.cfg 

The configuration file is like (an example from 
`PyMIC_examples/classification/CHNCXR <https://github.com/HiLab-git/PyMIC_examples/tree/main/classification/CHNCXR>`_):

.. code-block:: none

   [evaluation]
   metric_list = [accuracy, auc]
   ground_truth_csv = config/cxr_test.csv
   predict_csv   = result/resnet18.csv
   predict_prob_csv   = result/resnet18_prob.csv

See :mod:`pymic.util.evaluation_cls.main` for details of the configuration required.
