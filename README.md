# yeastmp_dms — DMS fitness heatmap pipeline

A reproducible pipeline that takes a folder of deep mutational scanning (DMS)
fitness tables, normalises every dataset onto a common biological scale
(**wild-type protein = 1, nonsense = 0**), and renders a residue × position
heatmap for each one.

```
dms-heatmap --input data/raw --output results
```

That command walks `data/raw` recursively, and for every dataset writes a
figure, the normalised per-variant table, the matrix behind the figure, and a
QC record.

---

## Contents

- [Install](#install)
- [Usage](#usage)
- [What the pipeline does](#what-the-pipeline-does)
  - [1. Load and validate](#1-load-and-validate)
  - [2. Normalise](#2-normalise-wild-type--1-nonsense--0)
  - [3. Colour](#3-colour)
  - [4. Render](#4-render)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Two things worth knowing about the input data](#two-things-worth-knowing-about-the-input-data)
- [Tests](#tests)
- [Project layout](#project-layout)

---

## Install

Requires Python ≥ 3.11.

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

or with plain pip:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

Raw data is not in version control (see `.gitignore`); drop the `*.tsv`
datasets into `data/raw/` — subfolders are fine.

## Usage

```bash
# every dataset under data/raw, each on its own colour scale
dms-heatmap --input data/raw --output results

# one shared colour scale, so figures are directly comparable between genes
dms-heatmap --input data/raw --output results --shared-scale

# a plain red→blue ramp instead of a ramp anchored at wild type
dms-heatmap --input data/raw --output results --center none

# a single file, PNG only, wider blocks
dms-heatmap -i data/raw/tna1_fitness_estimation.tsv -o results \
            --formats png --block-size 80

# settings from a file (see config/default.toml)
dms-heatmap --config config/default.toml
```

`make all` runs the default configuration; `make test` runs the test suite.
`dms-heatmap --help` lists every flag.

## What the pipeline does

### 1. Load and validate

Column names are canonicalised (`wt aa` → `wt_aa`, `mean fitness` →
`mean_fitness`), so spelling differences between files do not matter. A file is
usable if it has `wt_aa`, `pos`, `mut_aa`, `aa_ham` and at least one
`raw_fitness_rep<N>` column; replicates are detected by pattern, so datasets
with two, three or six replicates all work with no configuration. If no `raw`
columns exist the loader falls back to `rescaled_fitness_rep<N>` and records a
warning.

The per-row `aa_seq`/`nt_seq` columns are dropped after the wild-type protein
sequence is extracted from them — they repeat the full ORF on every line and
account for most of the file size. Rows that cannot be placed (a substitution
with no position or no mutant residue) are dropped and counted.

A dataset that fails validation is **skipped with an error, not fatal** — the
rest of the folder still runs, and the failure is recorded in
`run_manifest.json`.

### 2. Normalise (wild type = 1, nonsense = 0)

Each replicate is anchored **independently**, then the replicates are averaged.
For replicate *r*:

```
fitness_r = (raw_r − stop_r) / (wt_r − stop_r)
```

- `wt_r` — **median** raw fitness of the wild-type-protein variants, i.e. rows
  with `aa_ham == 0`. These are synonymous at the nucleotide level, so the
  protein is wild type; there are hundreds of them, which makes a far more
  stable anchor than the single true wild-type row.
- `stop_r` — **median** raw fitness of the nonsense variants (`mut_aa == "*"`).

Why anchor per replicate rather than averaging first: replicates have different
dynamic ranges (in the TNA1 data, `wt − stop` is 1.32, 1.36 and 1.42 across the
three), so averaging on the raw scale lets the widest replicate dominate.

Why medians: both anchor distributions have long tails — nonsense variants near
the C-terminus are often tolerated and sit close to wild type — and a median
ignores them.

The transformation is linear and **nothing is clamped**. Variants fitter than
wild type exceed 1, and variants worse than a typical stop fall below 0; only
the *colour* saturates, never the stored numbers.

The reported `fitness` is the mean of the available normalised replicates, with
`fitness_sd` and `n_replicates` alongside it. `--min-replicates N` blanks
variants supported by fewer than N replicates so they cannot reach the heatmap
looking as solid as a fully measured cell.

> **Cross-check.** On the TNA1 dataset this procedure reproduces the authors'
> own `rescaled_fitness_rep*` columns exactly (max absolute difference
> 1.8 × 10⁻¹⁴), shifted by the +1 that moves wild type from 0 to 1. The
> pipeline derives its anchors from the raw data rather than reading those
> columns, so it behaves identically on datasets that never had them.

### 3. Colour

Low fitness is **red**, high fitness is **blue**, with a neutral grey midpoint.
The two arms are lightness-matched: the red ramp was derived from the blue one
by holding OKLCh lightness and chroma and rotating the hue, so a cell 0.3 below
the midpoint is exactly as dark as one 0.3 above and neither pole visually
dominates. Red↔blue is also the colour-vision-safe diverging pair — unlike
red/green it stays separable under deuteranopia and protanopia, and the poles
differ in lightness as well as hue.

**Saturating the tails.** The colour limits are the 10th and 90th percentiles
of the measured cells. Everything at or below the 10th percentile takes the
same deepest red; everything at or above the 90th takes the same deepest blue.
This stops a handful of extreme values from flattening the middle of the
distribution, where nearly all the biology is. The colourbar is drawn with
arrowheads at both ends to show that the ends are saturated. Change the tails
with `--lower-quantile` / `--upper-quantile`.

**Where the neutral point sits.** By default the midpoint is pinned to
wild-type fitness (`--center wt`, i.e. 1.0), using a two-slope norm: grey means
"indistinguishable from wild type", blue means "fitter than wild type", red
means "impaired", and **those meanings are identical in every dataset in the
run**, which is the point of normalising in the first place. If you would
rather have a straight ramp from the bottom decile to the top with no anchored
midpoint, use `--center none`; `--center 0.5` puts grey halfway between null
and wild type. Both variants of the TNA1 figure are worth looking at once
before you settle on one for a paper.

By default each dataset gets its own limits. `--shared-scale` pools every
dataset in the run and gives them one common scale, which is what you want when
comparing genes side by side.

### 4. Render

Rows are the 20 amino acids plus a nonsense (`*`) row, ordered by side-chain
chemistry by default (`--aa-order alphabetical` for the other option).

The protein is wrapped into stacked blocks of 60 positions (`--block-size`).
Cell size is fixed rather than figure size, so a 200-residue and a 900-residue
protein render at the same scale and can be laid side by side.

- A small black dot marks the wild-type residue in each column, so the sequence
  is readable straight off the figure.
- Cells that were never measured are light grey and clearly distinct from any
  colour on the ramp — a missing variant never looks like a measured one.
- Positions absent from the data (TNA1 has no variants at 141, 392 or 422)
  stay as empty columns rather than being closed up, so the position axis
  always means the real residue number.

## Outputs

```
results/
├── figures/
│   ├── tna1.png              # 300 dpi
│   └── tna1.pdf              # vector, for figure assembly
├── tables/
│   ├── tna1.normalised.tsv   # one row per variant: fitness, sd, n_replicates,
│   │                         #   and the per-replicate normalised values
│   └── tna1.matrix.tsv       # the residue × position grid behind the figure
├── summary.tsv               # one row per dataset: counts, coverage, limits,
│                             #   worst replicate correlation, warning count
└── run_manifest.json         # full provenance (below)
```

`run_manifest.json` records the resolved configuration, package versions, and
per dataset: variant counts, coverage, **the anchor values and sample sizes for
every replicate**, pairwise replicate correlations, the fitness quantiles, the
colour limits actually used, every warning, and the files written. A figure can
always be traced back to the numbers and settings that produced it.

## Configuration

Settings resolve in this order, later winning: dataclass defaults → TOML config
file (`--config`) → command line flags. `config/default.toml` documents every
option with its default.

| Flag | Default | Effect |
|---|---|---|
| `--input` / `--output` | `data/raw` / `results` | input file or folder (searched recursively), output folder |
| `--patterns` | `*.tsv *.csv` | which files count as datasets |
| `--anchor-statistic` | `median` | `median` or `mean` for both anchors |
| `--min-replicates` | `1` | blank variants below this replicate count |
| `--lower-quantile` / `--upper-quantile` | `0.10` / `0.90` | where the colour saturates |
| `--center` | `wt` | `wt`, `none`, or a number — where neutral grey sits |
| `--shared-scale` | off | one colour scale across the whole run |
| `--block-size` | `60` | positions per stacked block |
| `--aa-order` | `chemistry` | `chemistry` or `alphabetical` |
| `--no-stops` | off | drop the nonsense row from the figure |
| `--no-wt-marks` | off | drop the wild-type dots |
| `--formats` | `png pdf` | figure formats |
| `--no-tables` | off | figures only |

## Two things worth knowing about the input data

**The `mean fitness` column in these files is wrong — the pipeline ignores it.**
In `tna1_fitness_estimation.tsv` it is not the mean of the replicates. It is
the mean of `raw_fitness_rep3` and `rescaled_fitness_rep2` — two columns on
different scales — which reproduces the shipped column for **100 % of rows**
(exactly, to floating point). The symptom that gives it away: for variants with
only two replicates, `mean fitness` equals `rescaled_fitness_rep2` on the nose,
because `raw_fitness_rep3` is empty. It looks like a spreadsheet formula
pointing at a mis-specified cell range. The accompanying `fitness sd` column
is presumably built the same way and is equally untrustworthy. **Check whether
the other datasets in the folder share this bug**, and re-derive the summary
columns from the replicates — which is what this pipeline does.

**Coverage is uneven.** TNA1 has 8,887 substitutions over 530 of the 533
mutagenised positions (534-residue protein), but only 174 positions carry all
20 substitutions; the median position has 17. This is honest in the figure —
unmeasured cells are grey — but it matters when averaging per position.

## Tests

```bash
make test        # or: .venv/bin/pytest
```

The suite covers the normalisation maths against a synthetic dataset with known
anchors (including the per-replicate dynamic-range behaviour), column
canonicalisation and schema validation, matrix reshaping and gap preservation,
colour-limit derivation including the degenerate cases, and an end-to-end run
that checks the written outputs. No test needs the real data.

## Project layout

```
src/dms_heatmap/
├── dataset.py     # discovery, loading, canonical column names, validation
├── normalise.py   # per-replicate anchoring to wild type = 1 / nonsense = 0
├── matrix.py      # long variant table → residue × position grid
├── scale.py       # percentile colour limits and the diverging norm
├── palette.py     # the lightness-matched red↔blue ramp
├── plot.py        # figure geometry and rendering
├── pipeline.py    # orchestration, QC, manifest
└── cli.py         # argument parsing and config merging
```
