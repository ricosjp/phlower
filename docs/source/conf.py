# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import pathlib
import shutil

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


def mmd_scraper(block, block_vars, gallery_conf):
    # Find all PNG files in the directory of this example.
    path_current_example = os.path.dirname(block_vars["src_file"])
    mmds = pathlib.Path(path_current_example).glob("images/*.mmd")

    to_path = (
        pathlib.Path(path_current_example).parent.parent
        / "docs/source/tutorials/basic_usages/images"
    )

    for mmd in mmds:
        shutil.copy(mmd, to_path / mmd.name)
    return ""


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Phlower"
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
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "sidebar_hide_name": True,
    # "light_css_variables": {
    #     "color-brand-primary": "#f5f5f5"
    # },
}


# Options for logo
html_logo = "_static/logo.png"
html_title = f"phlower v{version}"

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
    "image_scrapers": ("matplotlib", mmd_scraper),
}
