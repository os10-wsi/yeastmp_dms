PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/pip
INPUT  ?= data/raw
OUTPUT ?= results

.PHONY: help venv install all filtered shared test clean

help:
	@echo "make install   create .venv and install the package (editable, with dev deps)"
	@echo "make all       run the pipeline over \$$INPUT (default: $(INPUT))"
	@echo "make filtered  same, but keep only variants with >=20 input reads in every replicate"
	@echo "make shared    same as 'all', with one colour scale shared across all datasets"
	@echo "make test      run the test suite"
	@echo "make clean     remove generated results and caches"

venv:
	test -d .venv || python3 -m venv .venv

install: venv
	$(PIP) install -e ".[dev]"

all:
	$(PYTHON) -m dms_heatmap.cli --config config/default.toml --input $(INPUT) --output $(OUTPUT)

filtered:
	$(PYTHON) -m dms_heatmap.cli --config config/default.toml --input $(INPUT) \
		--output $(OUTPUT)/filtered --min-input-count 20 --min-replicates 3

shared:
	$(PYTHON) -m dms_heatmap.cli --config config/default.toml --input $(INPUT) \
		--output $(OUTPUT)/shared --shared-scale

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf $(OUTPUT)/figures $(OUTPUT)/tables $(OUTPUT)/summary.tsv $(OUTPUT)/run_manifest.json
	rm -rf .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
