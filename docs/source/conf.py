# Configuration file for the Sphinx documentation builder.
#
# Syntonic: Tensor Library for Cosmological and Syntony Recursion Theory
# Author: Andrew Orth

import os
import sys

# Add the project root to sys.path for autodoc
# The syntonic package is at ../../python/syntonic relative to source/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'python'))

# -- Project information -----------------------------------------------------

project = 'Syntonic'
copyright = '2024-2026, Andrew Orth'
author = 'Andrew Orth'
version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'myst_parser',
    'sphinx_autodoc_typehints',
]

# Template paths
templates_path = ['_templates']

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document
master_doc = 'index'

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings (Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_typehints = 'description'
autodoc_class_signature = 'separated'
autosummary_generate = True

# MyST-Parser settings (for Markdown support)
myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'linkify',
    'replacements',
    'smartquotes',
    'strikethrough',
    'substitution',
    'tasklist',
]
myst_heading_anchors = 4

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# Todo extension settings
todo_include_todos = True

# MathJax configuration for SRT formulas
mathjax3_config = {
    'tex': {
        'macros': {
            'phi': r'\varphi',
            'PHI': r'\Phi',
            'syntony': r'\mathcal{S}',
            'dhsr': r'\mathcal{D}\mathcal{H}\mathcal{S}\mathcal{R}',
        }
    }
}

# -- Custom CSS --------------------------------------------------------------

def setup(app):
    """Add custom CSS for SRT documentation styling."""
    app.add_css_file('custom.css')
