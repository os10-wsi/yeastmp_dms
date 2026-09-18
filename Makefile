PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/pip
INPUT  ?= data/raw
OUTPUT ?= results

# Secondary-structure strips are drawn only if there is something to draw, so
# data/structures is picked up when it exists and passed to nothing when it
# does not -- that way `make all` works either way.
STRUCTURES ?= data/structures
STRUCTURE_FLAG = $(if $(wildcard $(STRUCTURES)),--structures $(STRUCTURES),)

.PHONY: help venv install all filtered shared facet test clean

help:
	@echo "make install   create .venv and install the package (editable, with dev deps)"
	@echo "make all       run the pipeline over \$$INPUT (default: $(INPUT)),"
	@echo "               with the secondary structure from \$$STRUCTURES if it exists"
	@echo "make filtered  same, but keep only variants with >=20 input reads in every replicate"
	@echo "make shared    same as 'all', with one colour scale shared across all datasets"
	@echo "make facet     same as 'all', with the distribution split into one panel per class"
	@echo "make test      run the test suite"
	@echo "make clean     remove generated results and caches"

venv:
	test -d .venv || python3 -m venv .venv

install: venv
	$(PIP) install -e ".[dev]"

all:
	$(PYTHON) -m dms_heatmap.cli --config config/default.toml --input $(INPUT) \
		--output $(OUTPUT) $(STRUCTURE_FLAG)

filtered:
	$(PYTHON) -m dms_heatmap.cli --config config/default.toml --input $(INPUT) \
		--output $(OUTPUT)/filtered $(STRUCTURE_FLAG) --min-input-count 20 --min-replicates 3

shared:
	$(PYTHON) -m dms_heatmap.cli --config config/default.toml --input $(INPUT) \
		--output $(OUTPUT)/shared $(STRUCTURE_FLAG) --shared-scale

facet:
	$(PYTHON) -m dms_heatmap.cli --config config/default.toml --input $(INPUT) \
		--output $(OUTPUT)/facet $(STRUCTURE_FLAG) --distribution-layout facet

test:
	$(PYTHON) -m pytest -q

clean:
	# Everything under $(OUTPUT) is generated; .gitkeep is the only thing kept.
	test ! -d $(OUTPUT) || find $(OUTPUT) -mindepth 1 ! -name .gitkeep -delete
	rm -rf .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
