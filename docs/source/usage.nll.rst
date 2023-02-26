.. _noisy_label_learning:

Noisy Label Learning
====================

.. note::

   Some NLL methods only use noise-robust loss functions without complex 
   training process, and just combining the standard :mod:`SegmentationAgent` with such  
   loss function works for training. 


NLL Configurations
------------------

In the configuration file for noisy label learning, in addition to those used in standard fully 
supervised learning, there is a ``noisy_label_learning`` section that is specifically designed
for NLL methods. In that section, users need to specify the ``method_name`` and configurations
related to the NLL method. ``supervise_type`` should be set as "`noisy_label`" in the ``dataset`` section.
 For example, the correspoinding configuration for CoTeaching is:

.. code-block:: none

    [dataset]
    ...
    supervise_type = noisy_label
    ...

    [network]
    ...

    [training]
    ...

    [noisy_label_learning]
    method_name  = CoTeaching
    co_teaching_select_ratio  = 0.8  
    rampup_start = 1000
    rampup_end   = 8000

    [testing]
    ...

.. note::

   The configuration items vary with different NLL methods. Please refer to the API 
   of each built-in NLL method for details of the correspoinding configuration.  
   See examples in `PyMIC_examples/seg_nll/ <https://github.com/HiLab-git/PyMIC_examples/tree/main/seg_nll/>`_.


Built-in NLL Methods
--------------------

Some NLL methods only use noise-robust loss functions. They are used with a standard fully supervised training
paradigm. Just set ``supervise_type`` = `fully_sup`, and use ``loss_type`` to one of them in the configuration file:

* ``GCELoss``: (`NeurIPS 2018 <https://arxiv.org/abs/1805.07836>`_)
  Generalized cross entropy loss. 

* ``MAELoss``: (`AAAI 2017 <https://arxiv.org/abs/1712.09482v1>`_)
  Mean Absolute Error loss. 

* ``NRDiceLoss``: (`TMI 2020 <https://ieeexplore.ieee.org/document/9109297>`_)
  Noise-robust Dice loss. 

The other NLL methods are implemented in child classes of 
:mod:`pymic.net_run.agent_seg.SegmentationAgent`, and they are:

* ``CLSLSR``: (`MICCAI 2020 <https://link.springer.com/chapter/10.1007/978-3-030-59710-8_70>`_)
  Confident learning with spatial label smoothing regularization. 

* ``CoTeaching``: (`NeurIPS 2018 <https://arxiv.org/abs/1804.06872>`_)
  Co-teaching between two networks for learning from noisy labels.

* ``TriNet``: (`MICCAI 2020 <https://link.springer.com/chapter/10.1007/978-3-030-59719-1_25>`_) 
  Tri-network combined with sample selection. 

* ``DAST``: (`JBHI 2022 <https://ieeexplore.ieee.org/document/9770406>`_) 
  Divergence-aware selective training. 

Customized NLL Methods
----------------------

PyMIC alo supports customized NLL methods by inheriting the `SegmentationAgent` class. 
You may only need to rewrite the `training()` method and reuse most part of the 
existing pipeline, such as data loading, validation and inference methods. For example:

.. code-block:: none

    from pymic.net_run.agent_seg import SegmentationAgent

    class MyNLLMethod(SegmentationAgent):
      def __init__(self, config, stage = 'train'):
          super(MyNLLMethod, self).__init__(config, stage)
          ...
        
      def training(self):
          ...
    
    agent = MyNLLMethod(config, stage)
    agent.run()

You may need to check the source code of built-in NLL methods to be more familar with 
how to implement your own NLL method. 

In addition, if you want to design a new noise-robust loss fucntion, 
just follow :doc:`usage.fsl` to impelement and use the customized loss. 