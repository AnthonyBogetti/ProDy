import os
import sys
sys.path.insert(0, os.path.abspath('..'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# This tells Sphinx to create a separate index entry for every function
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# This ensures that when you search "parsePDB", 
# the search bar finds the exact function, not just the file it's in.
add_module_names = False 

# Clean Sidebar Look
html_theme = 'sphinx_rtd_theme'
html_logo = "_static/logo.png" # Make sure this file exists!
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 4,
}

html_static_path = ['_static']
def setup(app):
    app.add_css_file('custom.css')
