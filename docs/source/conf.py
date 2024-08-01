# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import glob
import os
import shutil

import sphinx_rtd_theme
from sphinx_gallery.scrapers import figure_rst
from sphinx_gallery.sorting import FileNameSortKey

import phlower

# # https://sphinx-gallery.github.io/stable/advanced.html#example-2-detecting-image-files-on-disk
# def png_scraper(block, block_vars, gallery_conf):
#     # Find all PNG files in the directory of this example.
#     path_current_example = os.path.dirname(block_vars['src_file'])
#     pngs = sorted(glob.glob(os.path.join(path_current_example, '*.png')))

#     # Iterate through PNGs, copy them to the Sphinx-Gallery output directory
#     image_names = list()
#     image_path_iterator = block_vars['image_path_iterator']
#     seen = set()
#     for png in pngs:
#         if png not in seen:
#             seen |= set(png)
#             this_image_path = image_path_iterator.next()
#             image_names.append(this_image_path)
#             shutil.copy(png, this_image_path)
#     # Use the `figure_rst` helper function to generate reST for image files
#     return figure_rst(image_names, gallery_conf['src_dir'])


# # https://sphinx-gallery.github.io/stable/advanced.html#example-2-detecting-image-files-on-disk
# def mmd_scraper(block, block_vars, gallery_conf):
#     # Find all PNG files in the directory of this example.
#     path_current_example = os.path.dirname(block_vars['src_file'])
#     pngs = sorted(glob.glob(os.path.join(path_current_example, '*.mmd')))

#     # Iterate through PNGs, copy them to the Sphinx-Gallery output directory
#     image_names = list()
#     image_path_iterator = block_vars['image_path_iterator']
#     seen = set()

#     print(f"mmd scraper: {path_current_example}, {pngs}")
#     for png in pngs:
#         if png not in seen:
#             seen |= set(png)
#             this_image_path = image_path_iterator.next()
#             image_names.append(this_image_path)
#             shutil.move(png, this_image_path)
#     # Use the `figure_rst` helper function to generate reST for image files
#     return ""


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
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # google, numpy styleのdocstring対応
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.mermaid",
]


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {"display_version": True}


# -- Extension configuration -------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "exclude-members": "with_traceback",
    "show-inheritance": False,
}


sphinx_gallery_conf = {
    "examples_dirs": [
        "../../tutorials/basic_usages",
    ],
    "gallery_dirs": [
        "tutorials/basic_usages",
    ],
    "within_subsection_order": FileNameSortKey,
    "filename_pattern": r"/*\.py",
}
