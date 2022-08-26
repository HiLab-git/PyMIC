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
------------

PyMIC accepts a configuration file for runing. For example, to train a network
for segmentation with full supervision, run the fullowing command:

.. code-block:: bash

    pymic_run train myconfig.cfg 

After training, run the following command for testing:

.. code-block:: bash

    pymic_run test myconfig.cfg

