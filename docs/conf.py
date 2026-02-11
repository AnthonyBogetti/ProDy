import os
import sys
import sphinx_rtd_theme

# -- Path Setup --------------------------------------------------------------
# 1. Add the project root to the path so Sphinx can find 'prody'
sys.path.insert(0, os.path.abspath('..'))

# -- Project Information -----------------------------------------------------
project = 'ProDy'
copyright = '2010-2026, Bahar Lab'
author = 'Bahar Lab'
version = '2.6'
release = '2.6.1'

# -- General Configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # Reads your Python code
    'sphinx.ext.autosummary',   # Generates summary tables
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',       # Renders math equations
    'sphinx.ext.viewcode',      # Adds links to source code
    'sphinx.ext.napoleon',      # Parses NumPy/Scientific docstrings
    'sphinx_rtd_theme',         # The theme extension
    'sphinxcontrib.jquery',     # Fixes the search bar crash
]

# -- C-Extension Mocking (THE FIX) -------------------------------------------
# This tells Sphinx: "If you try to import these C-modules and fail, just 
# pretend they exist so you can keep documenting the rest of the code."
autodoc_mock_imports = [
    'prody.proteins.c_prody',
    'prody.dynamics.rtbtools',
    'prody.sequence.c_sequence',
    'prody.kdtree',
]

# -- Theme Settings ----------------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_logo = "_static/prody_logo.png" # Ensure this file exists, or comment out
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
}
html_static_path = ['_static']

# -- Napoleon Settings -------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
