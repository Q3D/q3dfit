Installation
============

**Downloading and Installing q3dfit**

``q3dfit`` is only supported and tested for Python 3.7, 3.8 and
3.9. We recommend installation of clean environment via conda.

    #. Download one of the supported environment files: `for Python 3.7 <https://github.com/Q3D/q3dfit/blob/main/docs/jwebbinar-test-py37.yml>`_, `for Python 3.8 <https://github.com/Q3D/q3dfit/blob/main/docs/jwebbinar-test-py37.yml>`_, or `for Python 3.9 <https://github.com/Q3D/q3dfit/blob/main/docs/jwebbinar-test-py37.yml>`_. We are currently conducting all primary testing and development in Python 3.8. 

    #. Create a new conda environment from the environment file: 

        .. code-block:: console

            conda env create -f jwebbinar-test-py38.yml

       You may be getting error messages (something to do with wheels). Even if that's the case, proceed to 

    #. Activate it:

        .. code-block:: console

            conda activate jwebbinar-test-py38

    #. If you had error messages previously, it is likely that you do not have either ``mpich`` or ``mpi4py`` or both. Install them via these commands:

        .. code-block:: console

            brew install mpich
            pip install mpi4py


    #. Update the environment again: 

        .. code-block:: console

            conda env update -f jwebbinar-test-py38.yml --prune

    #. Install ``q3dfit`` into the newly created environment:

        .. code-block:: console

            git clone https://github.com/Q3D/q3dfit

    #. Suppose you cloned ``q3dfit`` into a directory ``/Users/username/work/``. Your python must know the path to the directory where ``q3dfit`` is installed in order to be able to import it. This can be accomplished in a couple of different ways. One is to run ``jupyter notebook`` from the command line while being in the ``/Users/username/work/`` directory. In that case running ``import q3dfit`` in python will successfully find the package. 

       A less restrictive way is to append the location of ``q3dfit`` to your system path. To this end, start your python code or jupyter notebook with 

        .. code-block:: python

	    import sys
	    sys.path.append("/Users/username/work/")
	    import q3dfit


**Multi-processing capability**

The ``mpich`` package parallelizes ``q3dfit`` across multiple processor cores using the Message Passing Interface (MPI) standard.

``mpich`` install note for Macports: Run

.. code-block:: console

    sudo port select --set mpi mpich-mp-fortran

to get default commands (like ``mpiexec``) working.

The system path is used. Thus, similarly to the trick we used above to import ``q3dfit`` itself, we must make sure that the path to ``mpich`` is known in order for it to work. Again, one option is to install it in ``/Users/username/work/`` directory and run ``jupyter notebook`` or ``python`` from there. Alternatively, the path to ``mpich`` can be appended:

.. code-block:: python

    import sys
    sys.path.append("/usr/local/share/doc/")
    import mpich

.. 
 In multi-core processing, the system path is used. Thus the tool you
 use to run python (command line, Jupyter, Spyder) must inherit the
 system path to be able to find, e.g., ``mpiexec`` and ``q3dfit``. This
 can be accomplished in the case of Jupyter or Spyder by running these
 applications from the command line.


