#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import mock
MOCK_MODULES = ["numpy", "celerite._celerite"]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

d = os.path.dirname
sys.path.insert(0, os.path.join(d(d(os.path.abspath(__file__))), "python"))
import celerite  # NOQA

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
project = "celerite"
author = "Dan Foreman-Mackey, Eric Agol, & contributors"
copyright = "2016, 2017, " + author

version = celerite.__version__
release = celerite.__version__

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
    github_repo="celerite",
    github_version="master",
    conf_py_path="/docs/",
)
html_static_path = ["_static"]
html_show_sourcelink = False
