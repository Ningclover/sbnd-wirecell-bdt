# sbnd-wirecell-bdt

Retraining the SBND Wire-Cell selection BDTs (the ν<sub>e</sub> CC tagger first) on the large Aurora Monte Carlo
campaigns, and evaluating every model the way the collaboration compares selections. Everything here runs with
Python 3.9 or newer, numpy, pandas, pyarrow, uproot and xgboost; no ROOT.

The same scripts are used on three machines: ALCF Aurora (training samples), the Fermilab gpvm / FermiGrid (inference
samples) and wcgpu1 (toolkit reference). Nothing in the scripts names a machine; sample names, file lists and output
folders are arguments.

## The workflow, in order

| step | what | code |
|---|---|---|
| 1 | **Specification**: label, fiducial volume, key, tables, split, scoring, acceptance test. Read this first. | `spec/spec_v1.md`, `spec/spec_v1.py` |
| 2 | **Tables**: one parquet per sample and table (candidates with all tagger scalars and the 258 model inputs, truth, files with exposure, objects). | `tables/build_tables.py`, `merge_tables.py`, `compare_tables.py` |
| 3 | **Baseline**: the shipped weights on the tables. | `baseline/baseline_full.py` |
| 4 | **Training**: round 0 reproduces the small-sample study (acceptance test); round 1 is the shipped structure at full statistics with a learning curve and a per-energy-bin comparison. | `training/round0/`, `training/round1/` |
| 5 | **Evaluation standard**: the collaboration's efficiency / purity table with absolute normalization. | `evaluation/eval_standard.py`, `docs/evaluation_standard.md` |
| 6 | **Re-score the shipped TMVA model** without ROOT (used to prove the merged CAFs carry the complete model inputs). | `tools/tmva_grad_forest.py` |

`WORKFLOW.md` explains how to run it end to end and how to adapt it to another BDT (NC π⁰, the "other" BDT of the
single-photon selection, the ν<sub>μ</sub> CC tagger). Results of the ν<sub>e</sub> retraining are kept in the project's
internal record, not in this repository.

## Inputs

Merged flat CAFs with the Wire-Cell tree `recTreeWireCell` beside `recTree` (the SBND Wire-Cell productions). The label comes from `T_match` (SPINE-style truth match written by the toolkit), the
SCE-shifted true vertex from `T_dlvtx_call`, the exposure from the CAF header. No Bee files are needed.

## Quick start

```bash
# tables for one sample (paths or root:// URLs; a list with --list file --slice a:b; --nproc N locally)
python tables/build_tables.py aurora-r2-nuecc OUT --list nuecc-merged-cafs.txt --nproc 8 --tag s0
python tables/merge_tables.py OUT aurora-r2-nuecc MERGED
# the shipped weights on them
python baseline/baseline_full.py MERGED baseline.json
# round 1: the shipped 258-input structure at full statistics, learning curve
python training/round1/train_round1.py MERGED OUT_R1 5
python training/round1/energy_bins.py MERGED OUT_R1/model_A_round1_seed5.json OUT_R1/energy_bins.json
# the evaluation standard (needs the inference-sample tables and their exposure)
python evaluation/eval_standard.py --help
```

## Status (2026-11-17)

The ν<sub>e</sub> CC tagger has been retrained at full statistics on the Aurora campaigns (round 1); round 2 (sample
weighting, early stopping, depth scan, new input families) is next. Numbers are reported inside the collaboration.

## Contributing

One topic per issue. Every number in a result comes from a script in this repository and names it. Changes to the
specification get a new version number. Commits are authored by the person running the work; a `Co-Authored-By`
trailer names an assistant when one wrote the code.
