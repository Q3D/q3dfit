from setuptools import setup, find_packages

setup(
    name='q3dfit',
    version='2.0.0-rc.1',    
    url='https://q3dfit.readthedocs.io/',
    author='David Rupke and the Q3D Team',
    author_email='drupke@gmail.com',
    license='GPL-3',
    long_description='Model astronomical data from integral field spectrographs.',

    packages=find_packages(),

    include_package_data = True,

    install_requires=[
        'astropy',
        'lmfit',
        'matplotlib',
        'numpy',
        'pandas',
        'photutils',
        'plotbin',
        'ppxf',
        'scikit-image',
        'scipy'
    ]
)
