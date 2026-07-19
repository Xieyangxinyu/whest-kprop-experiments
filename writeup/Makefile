.PHONY: pdf clean

LATEXMK ?= latexmk
MAIN ?= main

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	$(LATEXMK) -C $(MAIN).tex