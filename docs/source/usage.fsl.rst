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
:mod:`pymic.transform.trans_dict` lists all the built in transforms supported 
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
also specify the parameters via configuration file for the customized transform. 
The following is some example code for this:

.. code-block:: none
   from pymic.transform.trans_dict import TransformDict 
   from pymic.transform.abstract_transform import AbstractTransform

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