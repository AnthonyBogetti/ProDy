import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project = "ProDy"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "none"

autodoc_mock_imports = [
    "numpy",
    "scipy",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

