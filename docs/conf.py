import os
import sys
import sphinx_rtd_theme

# --- 1. PATH SETUP (Critical) -----------------------------------------------
# Adds the project root to Python path so Sphinx can find 'prody'.
sys.path.insert(0, os.path.abspath('..'))

# --- 2. PROJECT INFO --------------------------------------------------------
project = 'ProDy'
copyright = '2010-2026, Bahar Lab'
author = 'Bahar Lab'
version = '2.6'
release = '2.6.1'

# --- 3. EXTENSIONS ----------------------------------------------------------
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

# --- 4. AUTODOC SETTINGS (Fixes Empty Functions) ----------------------------
# This forces Sphinx to document 'parsePDB' even though it's imported
# from a different file (pdbfile.py).
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'imported-members': True,  # <--- CRITICAL: Shows imported functions like parsePDB
    'show-inheritance': True,
}

# --- 5. MOCK IMPORTS (Fixes "Missing C-Extension" Crash) --------------------
# This prevents Sphinx from skipping files when it can't compile C code.
autodoc_mock_imports = [
    'prody.proteins.c_prody',  # <--- CRITICAL for parsePDB
    'prody.dynamics.rtbtools',
    'prody.sequence.c_sequence',
    'prody.kdtree',
    'prody.lib',
    'Bio',
    'scipy',
    'matplotlib',
    'requests',
    'numpy',
]

# --- 6. THEME & LOGO SETTINGS -----------------------------------------------
html_theme = 'sphinx_rtd_theme'

# UPDATED: Matches your actual filename "logo.png" inside "_static" folder
html_logo = "_static/logo.png"

html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'logo_only': True, 
}

# Ensure Sphinx looks inside the '_static' folder for the logo
html_static_path = ['_static']

# --- 7. NAPOLEON SETTINGS (For Docstrings) ----------------------------------
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

# --- 8. EXCLUDE PATTERNS (Speed Fix) ----------------------------------------
exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store', 
    '**/tests', 
]
