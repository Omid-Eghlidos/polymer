# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import shutil

# Add root folder, i.e., "polymer", to the system path to import all the modules
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Polymer'
copyright = '2024, Omid Eghlidos'
author = 'Omid Eghlidos'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

language = 'en'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for PDF output -------------------------------------------------
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': r'''
        \usepackage{amsmath,amsfonts,amssymb,amsthm}
        \usepackage{graphicx}
    ''',
    'figure_align': 'htbp',
    'fontpkg': r'''
        \usepackage{times}
    ''',
    'babel': '\\usepackage[english]{babel}',
    'releasename': 'Version',
}


def copy_files_after_build(app, exception):
    """
    Copy the soft link to the generated index.html and copy the generated pdf file
    from the build/html and build/pdf folders to the root folder for easier
    access and convenience.
    """
    # Check if the build was successful
    if not exception:
        if app.builder.name == 'html':
            # Copy index.html to the root directory if it is already generated
            html_source_path = os.path.join(app.outdir, 'index.html')
            if os.path.exists(html_source_path):
                html_target_path = os.path.join(app.srcdir, '../polymer.html')
                if os.path.exists(html_target_path):
                    os.remove(html_target_path)
                os.symlink(html_source_path, html_target_path)
                print('Soft linked index.html to the root directory as polymer.html.')
        elif app.builder.name == 'latex':
            # Copy LaTex PDF to the root directory if it is already generated
            pdf_source_path  = os.path.join(app.outdir, 'polymer.pdf')
            if os.path.exists(pdf_source_path):
                pdf_target_path = os.path.join(app.srcdir, '../polymer.pdf')
                shutil.copyfile(pdf_source_path, pdf_target_path)
                print('Copied polymer.pdf to the root directory.')


def setup(app):
    app.connect('build-finished', copy_files_after_build)

