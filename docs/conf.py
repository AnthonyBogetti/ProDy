import os
import sys
import sphinx_rtd_theme

# 1. Path Setup
sys.path.insert(0, os.path.abspath('..'))

# 2. Project Info
project = 'ProDy'
copyright = '2010-2026, Bahar Lab'
author = 'Bahar Lab'

# 3. Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinxcontrib.jquery',
]

# 4. CRITICAL FIX: Force Sphinx to document imported functions (like parsePDB)
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'imported-members': True,  # Shows functions imported from other files
    'show-inheritance': True,
}

# 5. CRITICAL FIX: Mock C-extensions so build doesn't crash
autodoc_mock_imports = [
    'prody.proteins.c_prody',
    'prody.dynamics.rtbtools',
    'prody.sequence.c_sequence',
    'prody.kdtree',
    'prody.lib',
    'Bio', 'scipy', 'matplotlib', 'requests', 'numpy'
]

# 6. Theme & Logo
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']   # Tells Sphinx to look in docs/_static/
html_logo = "_static/logo.png"   # path relative to docs/

# 7. Speed up build
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/tests']
