import os
import sys
import sphinx_rtd_theme

# --- CRITICAL: PATH SETUP ---------------------------------------------------
# This tells Sphinx where to find the 'prody' source code.
# Without this, Autodoc fails, and your code docs (and search) will be empty.
sys.path.insert(0, os.path.abspath('..')) 
# ----------------------------------------------------------------------------

# -- Project Information -----------------------------------------------------
project = 'ProDy'
copyright = '2010-2026, Bahar Lab'
author = 'Bahar Lab'

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

# -- Theme Settings ----------------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# 1. PATH TO YOUR LOGO (Relative to the 'docs' folder)
html_logo = "_static/logo.png"

# 2. THEME OPTIONS
html_theme_options = {
    'logo_only': True,        # Set to True if you want to hide the text "ProDy" entirely
    # 'logo_only': False,       # Set to False if you want Logo + Text
    'display_version': True,  # Show the version number (e.g., v2.5) below the logo
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
}

# Ensure this is set so Sphinx looks in the _static folder
html_static_path = ['_static']


# -- Napoleon Settings (for ProDy docstrings) --------------------------------
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
