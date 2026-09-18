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
  - [4. Render the heatmap](#4-render-the-heatmap)
  - [5. Draw the secondary structure](#5-draw-the-secondary-structure)
  - [6. Render the distribution](#6-render-the-distribution)
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
datasets into `data/raw/` — subfolders are fine. Structures, if you have them,
go in `data/structures/` named after the gene (`tna1.pdb`, `tna1.dssp`, …).

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

# with the secondary structure drawn under each block, coloured by fitness
dms-heatmap --input data/raw --output results --structures data/structures

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

### 4. Render the heatmap

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

### 5. Draw the secondary structure

Given `--structures`, each block of the heatmap gets two SSDraw-style strips
beneath it, sharing its x axis position for position: **helices as coiled
ribbons, strands as arrows pointing towards the C terminus, loops as thin
bars**, and a hairline where the model does not cover the position.

```bash
dms-heatmap --input data/raw --output results --structures data/structures
```

The shape says where a residue sits in the fold. The **colour** says what
mutating it does, and the two strips colour the same shapes two ways:

| Strip | Coloured by |
|---|---|
| `→P` | the proline substitution at that position |
| `mean` | the mean of all substitutions measured at that position |

Both take the heatmap's own colormap and its own limits, so a red helix turn
means exactly what a red cell means, and `--shared-scale` puts the strips of
every dataset in the run on one scale too. Proline is the default first strip
(`--structure-residue`) because it is the substitution whose effect is a
statement about the backbone — it cannot donate a backbone hydrogen bond and it
kinks the chain — so an intolerant helix shows up as a red element while the
loops either side stay pale. The mean excludes nonsense variants by default
(`--structure-mean-with-stops` to include them): a stop is not one substitution
among twenty, its effect grows with how much of the protein it removes, and
averaging it in tilts every position's mean by an amount that says more about
where the position sits in the sequence than about the residue.

Two things to know when reading the strips:

- **A position mean spans a narrower range than a single substitution.** The
  colour limits are percentiles of the *cells*, so the mean strip cannot reach
  the ends of the ramp as often as the cells do and legitimately looks milder.
  The figure caption says so.
- **Colour inside a shape is per residue, not per element.** A single helix
  carries a different colour on every turn. The colour is painted as a strip
  and clipped to the outline of the shapes, which is the only way to do that
  for a coil without visible seams.

**Where the structure comes from.** A file or a folder searched recursively,
matched to a dataset by filename — `tna1.pdb`, `tna1_alphafold.cif` and
`TNA1.dssp` all belong to dataset `tna1`, and a lone structure with a lone
dataset is paired whatever it is called. Four formats are read directly, with
no extra dependencies:

| | |
|---|---|
| `*.dssp` | DSSP output; the eight-state code is collapsed to helix/strand/loop |
| `*.pdb` `*.ent` | `HELIX`/`SHEET` records, first model only |
| `*.cif` `*.mmcif` | the `_struct_conf` loop |
| `*.ss` | a bare `HHHEEELL` string, FASTA-style header optional |

A coordinate file carrying no secondary-structure records — which is the normal
case for a predicted model — is passed to `mkdssp` (or `dssp`) if one is on
`PATH`. If neither is, the strips would read as flat loop from end to end, so
the run says so and records it in the manifest rather than drawing a structure
nobody asserted.

**Numbering is checked, not assumed.** A structure numbered by author
numbering, a model of one domain, or a construct with a tag each put residue
*i* of the model at a different position of the assayed protein, and a silently
shifted structure strip is worse than no strip at all. So the offset is
*derived*: the model's own sequence is matched against the wild-type protein
sequence from the dataset, the identity achieved goes into `run_manifest.json`,
and a structure that cannot be placed is refused with a warning instead of
drawn in the wrong frame. A domain model of residues 60–180 numbered 1–121
lands at offset +59 on its own. `--structure-offset` overrides the search;
`--structure-chain` picks a chain out of a multi-chain file.

### 6. Render the distribution

Alongside each heatmap the pipeline draws the distribution of normalised
fitness split by variant class — **missense blue, synonymous green, nonsense
red** — as smooth curves on a shared grid, so they are comparable across
classes and not just within one.

The curve is a Gaussian kernel density **scaled to counts**, not to unit area:
its height at *x* is how many variants fall in a bin-width window there, so
the y axis still answers "how many variants are around here" — it is a
histogram with the chunking taken out, not a change of units. Bandwidth is
Silverman's rule per class, computed against the smaller of the standard
deviation and IQR/1.34 so a tail of noisy outliers cannot smear out the real
modes. `--distribution-smoothing` scales it: above 1 is smoother, below 1
follows the data more closely, and **0 goes back to a step histogram**.

This is the plot that tells you whether the normalisation worked. Synonymous
variants must pile up on 1 and nonsense variants on 0, because that is what
the anchors were defined to do; if they do not, something upstream is wrong.
On TNA1 they land on 1.00 and −0.01. The shape of the missense distribution
between the two modes is then the actual result of the screen.

Details that matter for reading it:

- Vertical dashed lines mark both anchors, so you are not doing the
  arithmetic off the axis.
- The axis range comes from a quantile trim (`--distribution-trim`, default
  0.005). At 0.001 a dozen noisy outliers stretch TNA1's axis to 6.3 and
  squash all the structure; at 0.005 it runs −0.86 to 3.07. Trimmed variants
  are never dropped from the calculation — the kernel is evaluated on a
  padded grid, so data just outside the limits still shapes the curve at the
  edge rather than letting it decay to zero for want of neighbours. They are
  simply off the visible axis, and the caption says how many. (In step mode
  they go into the end bins instead.)
- Class colours were chosen by running the trio through a colour-vision
  check, not by eye. The obvious saturated green against a mid red lands at
  protanope ΔE 7.2 — inside the band where a palette is only legal with
  secondary encoding. These three clear it at ΔE 15.0. Blue and red are
  steps from the heatmap's own ramps, so the two figures share their ink.
  Each class also gets its own line style, which is what keeps them apart
  in greyscale print.
- In TNA1 missense outnumbers nonsense 18:1, so the nonsense peak is a low
  bump on the shared count axis. `--distribution-layout facet` stacks the
  three classes in panels sharing the x axis, each with its own count axis,
  which is the fix when one class dwarfs the others.

`--no-distribution` skips it; `--distribution-bins` sets the bin count.

## Outputs

```
results/
├── figures/
│   ├── tna1.png                  # heatmap, 300 dpi
│   ├── tna1.pdf                  # vector, for figure assembly
│   ├── tna1.distribution.png     # per-class fitness histogram
│   └── tna1.distribution.pdf
├── tables/
│   ├── tna1.normalised.tsv   # one row per variant: fitness, sd, n_replicates,
│   │                         #   and the per-replicate normalised values
│   ├── tna1.matrix.tsv       # the residue × position grid behind the figure
│   └── tna1.structure.tsv    # one row per position: secondary structure and
│                             #   the two values the strips are coloured by
├── summary.tsv               # one row per dataset: counts, coverage, limits,
│                             #   worst replicate correlation, warning count,
│                             #   and the per-class medians that must read 1/0
└── run_manifest.json         # full provenance (below)
```

`run_manifest.json` records the resolved configuration, package versions, and
per dataset: variant counts, coverage, **the anchor values and sample sizes for
every replicate**, pairwise replicate correlations, the fitness quantiles, the
per-class distribution summaries, the colour limits actually used, **the
secondary structure — its source, chain, the numbering offset applied and the
sequence identity that justified it, the helix/strand/loop composition, and a
summary of each strip** — every warning, and the files written. A figure can
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
| `--structures` | none | structure file or folder for the secondary-structure strips |
| `--no-structure-tracks` | off | ignore `--structures` and draw the heatmap alone |
| `--structure-residue` | `P` | residue whose substitution colours the first strip |
| `--structure-mean-with-stops` | off | average nonsense into the mean strip too |
| `--structure-offset` | auto | add N to the structure's residue numbers |
| `--structure-chain` | first | chain to read from a multi-chain structure |
| `--no-distribution` | off | skip the per-class distribution figure |
| `--distribution-smoothing` | `1.0` | kernel bandwidth multiplier; `0` for a step histogram |
| `--distribution-bins` | `60` | reference bin width for the y axis (and bins in step mode) |
| `--distribution-layout` | `overlay` | `overlay` or `facet` (one panel per class) |
| `--distribution-trim` | `0.005` | quantile trimmed from each end of the axis |

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
colour-limit derivation including the degenerate cases, class splitting, and
the kernel density (that it integrates to the variant count, peaks on the
mode, resolves two modes rather than merging them, and that its bandwidth
resists a heavy tail). For the secondary structure it covers all four input
formats against fixed-column fixtures, chain selection, the eight-to-three
state collapse, and the placement rules that matter most — that a domain model
is found by its sequence, that a structure of the wrong protein is refused
rather than drawn, and that an element cut by a block boundary keeps its coil
phase and does not grow a second arrowhead. An end-to-end run checks the
written outputs. No test needs the real data.

## Project layout

```
src/dms_heatmap/
├── dataset.py     # discovery, loading, canonical column names, validation
├── normalise.py   # per-replicate anchoring to wild type = 1 / nonsense = 0
├── matrix.py      # long variant table → residue × position grid
├── scale.py       # percentile colour limits and the diverging norm
├── palette.py     # the lightness-matched red↔blue ramp, class colours
├── structure.py   # DSSP/PDB/mmCIF/string → three-state codes, placed on the protein
├── track.py       # SSDraw-style ribbons, coloured by fitness
├── plot.py        # heatmap geometry and rendering
├── distribution.py # per-class fitness histogram
├── pipeline.py    # orchestration, QC, manifest
└── cli.py         # argument parsing and config merging
```
