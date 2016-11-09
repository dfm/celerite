#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

d = os.path.dirname
sys.path.insert(0, d(d(os.path.abspath(__file__))))
import genrp  # NOQA

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'astropy': ('http://docs.astropy.org/en/stable/', None),
    'emcee': ('http://dan.iel.fm/emcee/current/', None)
}

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

# General information about the project.
project = "GenRP"
author = "Dan Foreman-Mackey, Eric Agol, & contributors"
copyright = "2016, " + author

version = genrp.__version__
release = genrp.__version__

exclude_patterns = ["_build"]
pygments_style = "sphinx"

# Readthedocs.
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_context = dict(
    display_github=True,
    github_user="dfm",
    github_repo="GenRP",
    github_version="master",
    conf_py_path="/docs/",
    # script_files=[
    #     "_static/jquery.js",
    #     "_static/underscore.js",
    #     "_static/doctools.js",
    #     "//cdn.mathjax.org/mathjax/latest/MathJax.js"
    #     "?config=TeX-AMS-MML_HTMLorMML",
    #     "_static/js/analytics.js",
    # ],
)
html_static_path = ["_static"]
html_show_sourcelink = False
