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

:mod:`pymic.net_run_ssl.ssl_abstract.SSLSegAgent` is the abstract class used for 
semi-supervised learning. The reccesponding 
