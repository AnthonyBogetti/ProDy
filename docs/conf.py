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

import prody

version = prody.__version__
release = prody.__version__


# -- Project information -----------------------------------------------------

project = 'ProDy'
copyright = '2010-2026, Bahar Group'
author = 'Bahar Group'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',      # Core library for html generation from docstrings
    'sphinx.ext.autosummary', # This is the key for individual pages
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',     # Add links to highlighted source code
    'sphinx.ext.mathjax',      # Render math equations
]

# Napoleon settings (optional but recommended for scientific code)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

add_module_names = True

autosummary_generate = True

autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'show-inheritance:': True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    # This ensures the logo is shown in the sidebar
    'logo_only': True,
    'display_version': True,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory.
html_static_path = ['_static']
html_logo = "_static/logo.png"   # path relative to docs/

# -- Mocking (Optional) ------------------------------------------------------
# If your build fails because it can't compile ProDy or Scipy on the server,
# uncomment the following lines to mock them. This fakes the import.
# autodoc_mock_imports = ["prody", "numpy", "scipy"]
