.. _installation:

.. role:: bash(code)
   :language: bash

Installation
============

Install PyMIC using pip (e.g., within a `Python virtual environment <https://www.geeksforgeeks.org/python-virtual-environment/>`_):

.. code-block:: bash

    pip install PYMIC

Alternatively, you can download or clone the code from `GitHub <https://github.com/HiLab-git/PyMIC>`_ and install PyMIC by

.. code-block:: bash

    git clone https://github.com/HiLab-git/PyMIC
    cd PyMIC
    python setup.py install

Dependencies
------------
PyMIC requires Python 3.6 (or higher) and depends on the following packages:

 - `h5py <https://www.h5py.org/>`_
 - `NumPy <https://numpy.org/>`_
 - `scikit-image <https://scikit-image.org/>`_
 - `SciPy <https://www.scipy.org/>`_
 - `SimpleITK <https://simpleitk.org/>`_

.. note::
   For the :mod:`pymia.data` package, not all dependencies are installed directly due to their heaviness.
   Meaning, you need to either manually install PyTorch by

       - :bash:`pip install torch`

   or TensorFlow by

       - :bash:`pip install tensorflow`

   depending on your preferred deep learning framework when using the :mod:`pymia.data` package.
   Upon loading a module from the :mod:`pymia.data` package, pymia will always check if the required dependencies are fulfilled.
