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

Dataset
-------

PyMIC provides two types of datasets for loading images from 
disk to memory: ``NiftyDataset`` and ``H5DataSet``. 

``NiftyDataset`` is designed for 2D and 3D images in common formats
such as .png, .jpeg, .bmp and nii.gz. ``H5DataSet`` is used for 
hdf5 data that are more efficient to load. 

To use ``NiftyDataset``, users need to specify the root path 
of the dataset and the csv file storing the image and label 
file names. Note that three .csv files are needed, and they are
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

The .csv file should have at least two columns (fields),
one for ``image`` and one for ``label``. If the input image 
have multiple modalities, and each modality is saved in a single 
file, then the .csv file should have N + 1 columnes, where the 
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
   
   trainset = MyDataset(...)
   valset   = MyDataset(...)
   testset  = MyDataset(...)
   agent    = SegmentationAgent(config, stage)
   agent.set_datasets(trainset, valset, testset)
   agent.run()
