from setuptools import setup, find_packages

setup(
    name='q3dfit',
    version='1.0.1',    
    url='https://q3dfit.readthedocs.io/en/latest/index.html',
    author='David Rupke and the Q3D Team',
    author_email='drupke@gmail.com',
    license='GPL-3',
    long_description='Model astronomical data from integral field spectrographs.',

    packages=find_packages(),

    include_package_data = True,

    install_requires=[
        'astropy>=4.3.1',
        'lmfit>=1.0.3',
        'matplotlib>=3.5.1',
        'numpy>=1.21.5',
        'pandas>=1.3.5',
        'photutils>=1.5.0',
        'plotbin>=3.1.3',
        'ppxf>=7.4.5',
        'scikit-image>=0.19.1',
        'scipy>=1.7.3'
    ]
)
