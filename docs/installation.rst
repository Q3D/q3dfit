Installation
==================

**Downloading and Installing Q3DFIT**

Q3DFIT is only supported and tested for Python 3.7, 3.8 and 3.9. We recommend installation of clean environment via conda.

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

    #. Install Q3DFIT into the newly created environment:

        .. code-block:: console

            git clone https://github.com/Q3D/q3dfit

