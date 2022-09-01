.. _noisy_label_learning:

Noisy Label Learning
====================

pymic_nll
---------

:mod:`pymic_nll` is the command for using built-in NLL methods for training. 
Similarly to :mod:`pymic_run`, it should be followed by two parameters, specifying the 
stage and configuration file, respectively. The training and testing commands are:

.. code-block:: bash

    pymic_nll train myconfig_nll.cfg
    pymic_nll test  myconfig_nll.cfg

.. tip::

   If the NLL method only involves one network, either ``pymic_nll`` or  ``pymic_run``
   can be used for inference. Their difference only exists in the training stage. 

.. note::

   Some NLL methods only use noise-robust loss functions without complex 
   training process, and just combining the standard :mod:`SegmentationAgent` with such  
   loss function works for training. ``pymic_run`` instead of ``pymic_nll`` should 
   be used for these methods.   


NLL Configurations
------------------

In the configuration file for ``pymic_nll``, in addition to those used in standard fully 
supervised learning, there is a ``noisy_label_learning`` section that is specifically designed
for NLL methods. In that section, users need to specify the ``nll_method`` and configurations
related to the NLL method. For example, the correspoinding configuration for CoTeaching is:

.. code-block:: none

    [dataset]
    ...

    [network]
    ...

    [training]
    ...

    [noisy_label_learning]
    nll_method   = CoTeaching
    co_teaching_select_ratio  = 0.8  
    rampup_start = 1000
    rampup_end   = 8000

    [testing]
    ...

.. note::

   The configuration items vary with different NLL methods. Please refer to the API 
   of each built-in NLL method for details of the correspoinding configuration.  

Built-in NLL Methods
--------------------

Some NLL methods only use noise-robust loss functions. They are used with ``pymic_run``
for training. Just set ``loss_type`` to one of them in the configuration file, similarly 
to the fully supervised learning. 

* ``GCELoss``: (`NeurIPS 2018 <https://arxiv.org/abs/1805.07836>`_)
  Generalized cross entropy loss. 

* ``MAELoss``: (`AAAI 2017 <https://arxiv.org/abs/1712.09482v1>`_)
  Mean Absolute Error loss. 

* ``NRDiceLoss``: (`TMI 2020 <https://ieeexplore.ieee.org/document/9109297>`_)
  Noise-robust Dice loss. 

The other NLL methods are implemented in child classes of 
:mod:`pymic.net_run_nll.nll_abstract.NLLSegAgent`, and they are:

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

PyMIC alo supports customizing NLL methods by inheriting the :mod:`NLLSegAgent` class. 
You may only need to rewrite the :mod:`training()` method and reuse most part of the 
existing pipeline, such as data loading, validation and inference methods. For example:

.. code-block:: none

    from pymic.net_run_nll.nll_abstract import NLLSegAgent

    class MyNLLMethod(NLLSegAgent):
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