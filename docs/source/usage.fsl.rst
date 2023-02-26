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
   from pymic.net_run.agent_seg import SegmentationAgent

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
   for segmentation, you don't need to write the above code. Just just use the ``pymic_train``
   command. See examples in `PyMIC_examples/segmentation/ <https://github.com/HiLab-git/PyMIC_examples/tree/main/segmentation/>`_.

Dataset
-------

PyMIC provides two types of datasets for loading images from 
disk to memory: ``NiftyDataset`` and ``H5DataSet``. 
``NiftyDataset`` is designed for 2D and 3D images in common formats
such as png, jpeg, bmp and nii.gz. ``H5DataSet`` is used for 
hdf5 data that are more efficient to load. 

To use ``NiftyDataset``, users need to specify the root path 
of the dataset and the csv file storing the image and label 
file names. The configurations include the following items:

* ``tensor_type``: data type for tensors. Should be :mod:`float` or :mod:`double`.

* ``task_type``: should be :mod:`seg` for segmentation tasks. 

* ``root_dir`` (string): the root dir of dataset. 

* ``modal_num`` (int, default is 1): modalities number. For images with N modalities,
  each modality should be saved in an indepdent file. 

* ``train_csv`` (string): the path of csv file for training set. 

* ``valid_csv`` (string): the path of csv file for validation set. 

* ``test_csv`` (string): the path of csv file for testing set. 

* ``train_batch_size`` (int): the batch size for training set. 

* ``valid_batch_size`` (int, optional): the batch size for validation set. 
  The defualt value is set as :mod:`train_batch_size`.

* ``test_batch_size`` (int, optional): the batch size for testing set. 
  The defualt value is 1.

The csv file should have at least two columns (fields),
one for ``image`` and the other for ``label``. If the input image 
have multiple modalities with each modality saved in a single 
file, then the csv file should have N + 1 columns, where the 
first N columns are for the N modalities, and the last column  
is for the label. The following is an example for configuration of dataset. 

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


To use your own dataset, you can define a dataset as a child class 
of ``NiftyDataset``, ``H5DataSet``, :mod:`or torch.utils.data.Dataset`
, and use :mod:`SegmentationAgent.set_datasets()`
to set the customized datasets. For example:

.. code-block:: none

   from torch.utils.data import Dataset 
   from pymic.net_run.agent_seg import SegmentationAgent

   class MyDataset(Dataset):
      ...
      # define your custom dataset here
   
   trainset, valset, testset = MyDataset(...), MyDataset(...), MyDataset(...)
   agent = SegmentationAgent(config, stage)
   agent.set_datasets(trainset, valset, testset)
   agent.run()

Transforms
----------

Several transforms are defined in PyMIC to preprocess or augment the data 
before sending it to the network. The ``TransformDict`` in 
:mod:`pymic.transform.trans_dict` lists all the built-in transforms supported 
in PyMIC. 

In the configuration file, users can specify the transforms required for training, 
validation and testing data, respectively. The parameters of each tranform class 
should also be provided, such as following:

.. code-block:: none

   # data transforms
   train_transform = [Pad, RandomRotate, RandomCrop, RandomFlip, NormalizeWithMeanStd, GammaCorrection, GaussianNoise, LabelToProbability]
   valid_transform = [NormalizeWithMeanStd, Pad, LabelToProbability]
   test_transform  = [NormalizeWithMeanStd, Pad]

   # the inverse transform will be enabled during testing
   Pad_output_size = [8, 256, 256]
   Pad_ceil_mode   = False
   Pad_inverse     = True    

   RandomRotate_angle_range_d = [-90, 90]
   RandomRotate_angle_range_h = None
   RandomRotate_angle_range_w = None

   RandomCrop_output_size = [6, 192, 192]
   RandomCrop_foreground_focus = False
   RandomCrop_foreground_ratio = None
   Randomcrop_mask_label       = None

   RandomFlip_flip_depth  = False
   RandomFlip_flip_height = True
   RandomFlip_flip_width  = True

   NormalizeWithMeanStd_channels = [0]

   GammaCorrection_channels  = [0]
   GammaCorrection_gamma_min = 0.7
   GammaCorrection_gamma_max = 1.5

   GaussianNoise_channels = [0]
   GaussianNoise_mean     = 0
   GaussianNoise_std      = 0.05
   GaussianNoise_probability = 0.5

For spatial transforms, you can specify whether an inverse transform is enabled
or not. Setting the inverse flag as True will transform the prediction output 
inversely during testing, such as ``Pad_inverse = True`` shown above. 
If you want to make images with different shapes to have the same shape before testing,
then the correspoinding transform's inverse flag can be set as True, so 
that the prediction output will be transformed back to the original image space. 
This is also useful for test time augmentation. 

You can also define your own transform operations. To integrate your customized 
transform to the PyMIC pipeline, just add it to the ``TransformDict``, and you can 
also specify the parameters via a configuration file for the customized transform. 
The following is some example code for this:

.. code-block:: none

   from pymic.transform.trans_dict import TransformDict 
   from pymic.transform.abstract_transform import AbstractTransform
   from pymic.net_run.agent_seg import SegmentationAgent

   # customized transform 
   class MyTransform(AbstractTransform):
      def __init__(self, params):
         super(MyTransform, self).__init__(params)
         ...

      def __call__(self, sample):
         ...

      def  inverse_transform_for_prediction(self, sample):
         ...

   my_trans_dict = TransformDict
   my_trans_dict["MyTransform"] = MyTransform
   agent = SegmentationAgent(config, stage)
   agent.set_transform_dict(my_trans_dict)
   agent.run()

Networks
--------

The configuration file has a ``network`` section to specify the network's type and  
hyper-parameters. For example, the following is a configuration for using ``2DUNet``:

.. code-block:: none

   [network]
   net_type = UNet2D
   # Parameters for UNet2D
   class_num     = 2
   in_chns       = 1
   feature_chns  = [16, 32, 64, 128, 256]
   dropout       = [0,  0,  0.3, 0.4, 0.5]
   bilinear      = False
   multiscale_pred = False

The ``SegNetDict`` in :mod:`pymic.net.net_dict_seg` lists all the built-in network 
structures currently implemented in PyMIC. 

You can also define your own networks. To integrate your customized 
network to the PyMIC pipeline, just call ``set_network()`` of ``SegmentationAgent``. 
The following is some example code for this:

.. code-block:: none

   import torch.nn as nn
   from pymic.net_run.agent_seg import SegmentationAgent
   
   # customized network 
   class MyNetwork(nn.Module):
      def __init__(self, params):
         super(MyNetwork, self).__init__()
         ...

      def forward(self, x):
         ...

   net = MyNetwork(params)
   agent = SegmentationAgent(config, stage)
   agent.set_network(net)
   agent.run()

.. _fsl_loss:

Loss Functions
--------------

The setting of loss function is in the ``training`` section of the configuration file,
where the loss function name and hyper-parameters should be provided.
The ``SegLossDict`` in :mod:`pymic.loss.loss_dict_seg` lists all the built-in loss 
functions currently implemented in PyMIC. 

The following is an example of the setting of loss:

.. code-block:: none

   loss_type = DiceLoss
   loss_softmax = True 

Note that PyMIC supports using a combination of loss functions. Just set ``loss_type`` 
as a list of loss functions, and use ``loss_weight`` to specify the weight of each
loss, such as the following:

.. code-block:: none

   loss_type     = [DiceLoss, CrossEntropyLoss]
   loss_weight   = [0.5, 0.5]

You can also define your own loss functions. To integrate your customized 
loss function to the PyMIC pipeline, just add it to the ``SegLossDict``, and you can 
also specify the parameters via a configuration file for the customized loss. 
The following is some example code for this:

.. code-block:: none

   from pymic.loss.loss_dict_seg import SegLossDict 
   from pymic.net_run.agent_seg import SegmentationAgent

   # customized loss 
   class MyLoss(nn.Module):
      def __init__(self, params = None):
         super(MyLoss, self).__init__()
         ...

      def forward(self, loss_input_dict):
         ...

   my_loss_dict = SegLossDict
   my_loss_dict["MyLoss"] = MyLoss
   agent = SegmentationAgent(config, stage)
   agent.set_loss_dict(my_loss_dict)
   agent.run()


Training Options
----------------

In addition to the loss fuction, users can specify several training 
options in the ``training`` section of the configuration file. 

Itreations
^^^^^^^^^^

For training iterations, the following parameters need to be specified in 
the configuration file:

* ``iter_max``: the maximal allowed iteration for training. 

* ``iter_valid``: if the value is K, it means evaluating the performance on the 
  validaiton set for every K steps. 

* ``iter_save``: The iteations for saving the model. If the value is k, it means 
  the model will be saved every k iterations. It can also be a list of integer numbers, 
  which specifies the iterations to save the model.

* ``early_stop_patience``: if the value is k, it means the training will stop when 
  the performance on the validation set does not improve for k iteations. 


Optimizer
^^^^^^^^^

For optimizer, users need to set ``optimizer``, ``learning_rate``,
``momentum`` and ``weight_decay``. The built-in optimizers include ``SGD``,
``Adam``, ``SparseAdam``, ``Adadelta``, ``Adagrad``, ``Adamax``, ``ASGD``,
``LBFGS``, ``RMSprop`` and ``Rprop`` that are implemented in `torch.optim`. 

You can also use customized optimizers via `SegmentationAgent.set_optimizer()`.

Learning Rate Scheduler
^^^^^^^^^^^^^^^^^^^^^^^

The current built-in learning rate schedulers are ``ReduceLROnPlateau`` 
and ``MultiStepLR``, which can be specified in ``lr_scheduler`` in 
the configuration file.  

Parameters related to  ``ReduceLROnPlateau`` include ``lr_gamma``.  
Parameters related to  ``MultiStepLR`` include ``lr_gamma`` and ``lr_milestones``. 

You can also use customized lr schedulers via `SegmentationAgent.set_scheduler()`.

Other Options
^^^^^^^^^^^^^

Other options for training include:

* ``gpus``: a list of GPU index for training the model. If the length is larger than 
  one (such as [0, 1]), it means the model will be trained on multiple GPUs parallelly. 

* ``deterministic`` (bool, default is True): set the training deterministic or not. 

* ``random_seed`` (int, optioinal): the random seed customized by user. Default value is 1.

* ``ckpt_save_dir``: the path to the folder for saving the trained models. 

* ``ckpt_prefix``: the prefix of the name to save the checkpoints. 


Inference Options
-----------------

There are several options for inference after training the model. You can also select 
the GPUs for testing, enable sliding window inference or inference with 
test-time augmentation, etc. The following is a list of options availble for inference:

* ``gpus``: a list of GPU index. Atually, only the first GPU in the list is used. 

* ``evaluation_mode`` (bool, default is True): set the model to evaluation mode or not. 

* ``test_time_dropout`` (bool, default is False): use test-time dropout or not. 

* ``ckpt_mode`` (int): which checkpoint is used. 0--the last checkpoint; 1--the checkpoint
  with the best performance on the validation set; 2--a specified checkpoint. 

* ``ckpt_name`` (string, optinal): the full path to the checkpoint if ckpt_mode = 2.

* ``post_process`` (string, default is None): the post process method after inference. 
  The current available post processing is :mod:`pymic.util.post_process.PostKeepLargestComponent`. 
  Uses can also specify customized post process methods via `SegmentationAgent.set_postprocessor()`.

* ``sliding_window_enable`` (bool, default is False): use sliding window for inference or not.

* ``sliding_window_size`` (optinal): a list for sliding window size when sliding_window_enable = True.

* ``sliding_window_stride`` (optinal): a list for sliding window stride when sliding_window_enable = True.

* ``tta_mode`` (int, default is 0): the mode for Test Time Augmentation (TTA). 0--not using TTA; 1--using 
  TTA based on horizontal and vertical flipping.  

* ``output_dir`` (string): the dir to save the prediction output. 

* ``ignore_dir`` (bool, default is True): if the input image name has a `/`, it will be replaced
  with `_` in the output file name. 

* ``save_probability`` (bool, default is False): save the output probability for each class. 

* ``label_source`` (list, default is None): a list of label to be converted after prediction. For example,
  `label_source` = [0, 1] and `label_target` = [0, 255] will convert label value from 1 to 255. 

* ``label_target`` (list, default is None): a list of label after conversion. Use this with `label_source`.

* ``filename_replace_source`` (string, default is None): the substring in the filename will be replaced with 
  a new substring specified by `filename_replace_target`.

* ``filename_replace_target`` (string, default is None): work with `filename_replace_source`.