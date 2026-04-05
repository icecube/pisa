"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

"""
Notes:
* In case navigation disappears for certain pages (see e.g. https://github.com/sphinx-doc/alabaster/issues/69),
remove build directory before re-running sphinx-build.
* The GitHub changelog extension requires a GITHUB_TOKEN environment variable -> generate a GitHub token
with a public repo scope and assign it to GITHUB_TOKEN before building the documentation
"""


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import pisa


# -- sphinx project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PISA'
copyright = '2014–2026, The IceCube Collaboration'
author = 'The IceCube Collaboration'
# The short X.Y version.
version = pisa.__version__
# The full version, including alpha/beta/rc tags.
release = version


# -- sphinx general configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ # TODO: need all these?
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel', # allow referencing sections by their title
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon', # for numpy-style docstrings
    'sphinx_github_changelog', # to include changelog generated from GitHub release notes
    'myst_nb', # for including jupyter notebooks
    #'sphinx_tippy', # for tooltips on links (TODO: get to work)
    #'myst_parser', # unnecessary in presence of myst_nb
]

templates_path = ['_templates']
# just document that we're aware of these not being cross-referenced
exclude_patterns = ['modules.rst',]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
}

# make sure the target is unique
autosectionlabel_prefix_document = True


# -- myst-nb configuration ---------------------------------------------------
# https://myst-nb.readthedocs.io
# https://docs.readthedocs.com/platform/latest/guides/jupyter.html

# don't execute notebooks: instead, we should make sure that they
# already have all the relevant outputs (and no more)
nb_execution_mode = "off"

# notebook cell execution timeout in seconds
#nb_execution_timeout = 120

# -- myst-parser configuration -----------------------------------------------
# https://myst-parser.readthedocs.io/

# detect then convert bare URLs into hyperlinks
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#linkify
myst_enable_extensions = ["linkify"]
# open external link in a new tab instead of leaving the documentation
myst_links_external_new_tab = True


# -- sphinx_github_changelog configuration -----------------------------------
# https://sphinx-github-changelog.readthedocs.io

# sphinx_github_changelog extension requires authenticating even for public repos
github_token = os.environ.get("GITHUB_TOKEN")
if github_token:
    sphinx_github_changelog_token = github_token


# -- sphinx_tippy configuration ----------------------------------------------
# https://sphinx-tippy.readthedocs.io/en/latest/#confval-tippy_anchor_parent_selector
# TODO: unable to get this to work
#tippy_anchor_parent_selector = "div.content"


# -- options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# Furo html theme docs: https://pradyunsg.me/furo

html_theme = 'furo'
html_static_path = ['_static']
# customisation
html_css_files = ['custom.css']
# same logo for light and dark mode, automagically copied
html_logo = "../../images/pisa4_logo_transparent.png"
html_theme_options = {
    # allow viewing and editing the .rst/.md source files in the GitHub repo:
    "top_of_page_buttons": ["view", "edit"],
    "source_repository": "https://github.com/icecube/pisa/",
    "source_branch": "master",
    "source_directory": "docs/source/",
    # display (shrunk) "PISA <tag> documentation" in addition to the logo:
    "sidebar_hide_name": False,
    # include a link to the GitHub repo via the GitHub icon (adapted from https://pradyunsg.me/furo/customisation/footer/)
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/icecube/pisa",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    # adapt and uncomment in case we need an announcement:
    #"announcement": "<em>Important</em> announcement!",
}
