import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project = "ProDy"
author = "Bahar Group"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

html_theme = "sphinx_rtd_theme"

