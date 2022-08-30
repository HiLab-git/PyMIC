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
   for segmentation, you don't need to write the above code. Just just use the ``pymic_run``
   command. 

Dataset
-------

PyMIC provides two types of datasets for loading images from 
disk to memory: ``NiftyDataset`` and ``H5DataSet``. 
``NiftyDataset`` is designed for 2D and 3D images in common formats
such as png, jpeg, bmp and nii.gz. ``H5DataSet`` is used for 
hdf5 data that are more efficient to load. 

To use ``NiftyDataset``, users need to specify the root path 
of the dataset and the csv file storing the image and label 
file names. Note that three csv files are needed, and they are
for training, validation and testing, respectively. For example:

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

By default, the ``valid_batch_size`` is set to the same as the ``train_batch_size``,
and the ``test_batch_size`` is 1. The csv file should have at least two columns (fields),
one for ``image`` and the other for ``label``. If the input image 
have multiple modalities with each modality saved in a single 
file, then the csv file should have N + 1 columnes, where the 
first N columns are for the N modalities, and the last column  
is for the label.

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
   deep_supervise= False

The ``SegNetDict`` in :mod:`pymic.net.neg_dict_seg` lists all the built-in network 
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

* iter_start: the start iteration, by default is 0. None zero value means the
iteration where a pre-trained model stopped for continuing with the trainnig.

* iter_max: the maximal allowed iteration for training. 

* iter_valid: if the value is K, it means evaluating the performance on the 
validaiton set for every K steps. 

* iter_save: The iteations for saving the model. If the value is k, it means 
the model will be savled every k iterations. It can also be a list of integer numbers, 
which specifies the iterations to save the model.



Optimizer
^^^^^^^^^

For optimizer, users need to set ``optimizer``, ``learning_rate``,
``momentum`` and ``weight_decay``.


Learning Rate Scheduler
^^^^^^^^^^^^^^^^^^^^^^^

The current supported learning rate schedulers are ``ReduceLROnPlateau`` 
and ``MultiStepLR``, which can be specified in ``lr_scheduler`` in 
the configuration file.  Parameters related to  ``ReduceLROnPlateau`` 
