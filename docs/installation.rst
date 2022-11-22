Installation
************

Downloading and Installing q3dfit
=================================

``q3dfit`` is only supported and tested for Python 3.7-3.9. We
recommend installation via pip:

    .. code-block:: console

	pip install q3dfit

Optional multi-processing capability
====================================

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

.. toctree::

.. 
 In multi-core processing, the system path is used. Thus the tool you
 use to run python (command line, Jupyter, Spyder) must inherit the
 system path to be able to find, e.g., ``mpiexec`` and ``q3dfit``. This
 can be accomplished in the case of Jupyter or Spyder by running these
 applications from the command line.
