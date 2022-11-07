Installation
============

**Downloading and Installing q3dfit**

``q3dfit`` is only supported and tested for Python 3.7, 3.8 and
3.9. We recommend installation of clean environment via conda. Instructions and more information on how to set up conda, can be found `here <https://astroconda.readthedocs.io/en/latest/getting_started.html>`_, for example.

    #. When you have set up conda, download one of the supported environment files: `for Python 3.7 <https://raw.githubusercontent.com/Q3D/q3dfit/main/docs/q3dfit-py37.yml>`_, `for Python 3.8 <https://raw.githubusercontent.com/Q3D/q3dfit/main/docs/q3dfit-py38.yml>`_, or `for Python 3.9 <https://raw.githubusercontent.com/Q3D/q3dfit/main/docs/q3dfit-py39.yml>`_. We are currently conducting all primary testing and development in Python 3.8. 

    #. Create a new conda environment from the environment file: 

        .. code-block:: console

            conda env create -f q3dfit-py38.yml

    #. Activate it:

        .. code-block:: console

            conda activate q3dfit-py38

    #. Install ``q3dfit`` into the newly created environment:

        .. code-block:: console

            git clone https://github.com/Q3D/q3dfit
	    
    #. Alternatively, go to https://github.com/Q3D/q3dfit, go to the green button `Code' and click `Download zip'.

    #. Suppose you cloned ``q3dfit`` into a directory ``/Users/username/work/``. Your python must know the path to the directory where ``q3dfit`` is installed in order to be able to import it. This can be accomplished in a couple of different ways. One is to run ``jupyter notebook`` from the command line while being in the ``/Users/username/work/`` directory. In that case running ``import q3dfit`` in python will successfully find the package. 

       A less restrictive way is to append the location of ``q3dfit`` to your system path. To this end, start your python code or jupyter notebook with 

        .. code-block:: python

	    import sys
	    sys.path.append("/Users/username/work/")
	    import q3dfit


**Optional multi-processing capability**

``q3dfit`` has the optional capability to parallelize fitting across multiple processor cores using the Message Passing Interface (MPI) standard. This is accomplished using the ``mpich`` package and its python wrapper ``mpi4py``. Installation instructions for ``mpich`` vary dependending on the user's system configurations. Some common methods include: 

     * Install via Macports:

	.. code-block:: console

	    sudo port install mpich
	    sudo port select --set mpi mpich-mp-fortran

     * Install via Brew:

        .. code-block:: console

            brew install mpich

     * Install via Conda:

        .. code-block:: console

            conda install mpich

If ``mpich`` is successfully installed, you should be able to locate the path to its main executable by typing ``which mpiexec``. 

The final step is to install ``mpi4py``, the ``mpich`` wrapper for the multi-processor mode. This can be done by

        .. code-block:: console

            conda install mpi4py

or

        .. code-block:: console

            pip install mpi4py==3.1.3


.. 
 In multi-core processing, the system path is used. Thus the tool you
 use to run python (command line, Jupyter, Spyder) must inherit the
 system path to be able to find, e.g., ``mpiexec`` and ``q3dfit``. This
 can be accomplished in the case of Jupyter or Spyder by running these
 applications from the command line.


