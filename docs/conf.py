# Configuration file for the Sphinx documentation builder.
import os
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# Assuming your project structure is:
# my_project/
#   docs/
#     conf.py
#   my_package/
#     __init__.py

sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'ProDy'
copyright = '2010-2026, Bahar Group'
author = 'Bahar Group'
release = '6.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',      # Core library for html generation from docstrings
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',     # Add links to highlighted source code
    'sphinx.ext.mathjax',      # Render math equations
    'sphinx.ext.intersphinx',  # Link to other project's documentation (like ProDy's)
]

# Napoleon settings (optional but recommended for scientific code)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Intersphinx mapping to link to ProDy and Python docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'prody': ('http://prody.csb.pitt.edu/manual/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory.
html_static_path = ['_static']
html_logo = "_static/logo.png"   # path relative to docs/

# -- Mocking (Optional) ------------------------------------------------------
# If your build fails because it can't compile ProDy or Scipy on the server,
# uncomment the following lines to mock them. This fakes the import.
# autodoc_mock_imports = ["prody", "numpy", "scipy"]
