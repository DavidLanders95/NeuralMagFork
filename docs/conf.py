# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "NeuralMag"
copyright = "2024, NeuralMag Team"
author = "NeuralMag Team"
release = "0.9.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.mathjax", "nbsphinx"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
numfig = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

mathjax3_config = {
    "tex": {
        "macros": {
            "vec": ["{\\mathbf #1}", 1],
            "tensor": ["{\\tilde{\\mathbf #1}}", 1],
            "mat": ["{\\mathbf #1}", 1],
            "dx": "{\\;\\text{d}\\vec{x}}",
            "ds": "{\\;\\text{d}\\vec{s}}",
        }
    }
}
