# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

default: dirhtml
TUTORIALS = tutorials/first.rst tutorials/modeling.rst tutorials/normalization.rst \
			tutorials/julia-first.rst

xml/index.xml: ../cpp/include/celerite/*.h ../cpp/include/celerite/solver/*.h
	doxygen Doxyfile

tutorials/%.rst: _static/notebooks/%.ipynb tutorials/tutorial_rst.tpl
	jupyter nbconvert --template tutorials/tutorial_rst --to rst $< --output-dir tutorials

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/* $(TUTORIALS)

.PHONY: html
html: $(TUTORIALS) xml/index.xml
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

.PHONY: dirhtml
dirhtml: $(TUTORIALS) xml/index.xml
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

.PHONY: singlehtml
singlehtml: $(TUTORIALS) xml/index.xml
	$(SPHINXBUILD) -b singlehtml $(ALLSPHINXOPTS) $(BUILDDIR)/singlehtml
	@echo
	@echo "Build finished. The HTML page is in $(BUILDDIR)/singlehtml."
