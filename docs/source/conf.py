# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sphinx_rtd_theme
import phlower

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "phlower"
copyright = "2024, RICOS"
author = "RICOS"
version = phlower.__version__
release = phlower.__version__


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    "sphinx.ext.autosummary",
    'sphinx.ext.napoleon',  # google, numpy styleのdocstring対応
    "sphinxcontrib.jquery"
]


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    'display_version': True
}


# -- Extension configuration -------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "exclude-members": "with_traceback",
}
