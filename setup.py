from setuptools import setup, find_packages

setup(
    name='q3dfit',
    version='1.2.0-beta',    
    url='https://q3dfit.readthedocs.io/en/latest/index.html',
    author='David Rupke and the Q3D Team',
    author_email='drupke@gmail.com',
    license='GPL-3',
    long_description='Model astronomical data from integral field spectrographs.',

    packages=find_packages(),

    include_package_data = True,

    install_requires=[
        'astropy',
        'lmfit>=1.0.3',
        'matplotlib',
        'numpy>=1.22',
        'pandas',
        'photutils',
        'plotbin',
        'ppxf',
        'scikit-image',
        'scipy'
    ]
)
