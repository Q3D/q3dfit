# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../q3dfit'))


# -- Project information -----------------------------------------------------

project = 'q3dfit'
copyright = '2025, David Rupke and the Q3D Team'
author = 'David Rupke and the Q3D Team'

# The full version, including alpha/beta/rc tags
version = '1.2.0'
release = '1.2.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon',
              'sphinx.ext.autodoc',
              #'numpydoc',
              'sphinx_autodoc_typehints',
              #'sphinx.ext.inheritance_diagram',
              'sphinx.ext.intersphinx',
              'sphinx.ext.viewcode']

intersphinx_mapping = {'astropy': ('https://docs.astropy.org/en/stable/', None),
                       'lmfit': ('https://lmfit.github.io/lmfit-py/', None),
                       'mpi4py': ('https://mpi4py.readthedocs.io/en/stable/', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'python': ('https://docs.python.org/3', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy', None)}


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# https://stackoverflow.com/questions/67473396/shorten-display-format-of-python-type-annotations-in-sphinx
autodoc_type_aliases = {
    'Iterable': 'Iterable',
    'ArrayLike': 'ArrayLike',
}

#autodoc_typehints = 'both'

# So __all__ will set order of functions in a module: https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html#confval-autosummary_ignore_module_all
autodoc_default_options = {
    'member-order': 'bysource',
    'ignore-module-all': False
}

# https://stackoverflow.com/questions/66182576/using-array-like-as-a-type-in-napoleon-without-warning
napoleon_type_aliases = {
    'array-like': ':term:`array-like <array_like>`',
    'array_like': ':term:`array_like`',
    'ArrayLike': ':term:`array_like`',
}
napoleon_use_rtype = False
#https://stackoverflow.com/questions/72220924/sphinx-how-to-show-attributes-as-in-scipy
napoleon_use_ivar = True
napoleon_attr_annotations = True #not sure this does anything

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
