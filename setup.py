from setuptools import setup, find_packages

setup(
    name='q3dfit',
    version='0.1.0',    
    url='https://q3dfit.readthedocs.io/en/latest/index.html',
    author='David Rupke',
    author_email='drupke@gmail.com',
    license='GPL-3',

    packages=find_packages(),

    include_package_data = True,
    
    install_requires=[
        #'mpi4py>=2.0',
        'astropy>=4.3.1',
        #'jupyter==1.0.0',
        'lmfit==1.0.3',
        'matplotlib==3.5.1',
        'numpy>=1.21.5',
        'pandas>=1.3.5',
        'ppxf==7.4.5',
        'scikit-image==0.19.1',
        'scipy>=1.7.3'
    ]
)
