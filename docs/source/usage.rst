Usage
=====

.. _installation:

Installation
------------

Install PyMIC using pip (e.g., within a `Python virtual environment <https://www.geeksforgeeks.org/python-virtual-environment/>`_):

.. code-block:: bash

    pip install PYMIC

Alternatively, you can download or clone the code from `GitHub <https://github.com/HiLab-git/PyMIC>`_ and install PyMIC by

.. code-block:: bash

    git clone https://github.com/HiLab-git/PyMIC
    cd PyMIC
    python setup.py install

Train and Test
--------------

PyMIC accepts a configuration file for runing. For example, to train a network
for segmentation with full supervision, run the fullowing command:

.. code-block:: bash

    pymic_run train myconfig.cfg 

After training, run the following command for testing:

.. code-block:: bash

    pymic_run test myconfig.cfg
   
Configuration File
------------------

PyMIC uses configuration files to specify the setting and parameters of a deep 
learning pipeline, so that users can reuse the code and minimize their workload.
Users can use configuration files to config almost all the componets involved, 
such as dataset, network structure, loss function, optimizer, learning rate 
scheduler and post processing methods, etc. 

.. tip::
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
   net_type = UNet2D
   # Parameters for UNet2D
   class_num     = 2
   in_chns       = 1
   feature_chns  = [16, 32, 64, 128, 256]
   dropout       = [0,  0,  0.3, 0.4, 0.5]
   bilinear      = False
   deep_supervise= False

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

   ckpt_save_dir = model/unet_dice_loss
   ckpt_prefix   = unet

   # start iter
   iter_start = 0
   iter_max   = 8000
   iter_valid = 200
   iter_save  = 8000

   [testing]
   # list of gpus
   gpus       = [0]
   # checkpoint mode can be [0-latest, 1-best, 2-specified]
   ckpt_mode  = 0
   output_dir = result

   # convert the label of prediction output
   label_source = [0, 1]
   label_target = [0, 255]


SegmentationAgent
-----------------

SegmentationAgent 