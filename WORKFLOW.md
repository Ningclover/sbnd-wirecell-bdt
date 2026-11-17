# Workflow: how to run it, and how to adapt it to another BDT

## 0. What a BDT needs here

One row per neutrino candidate with (a) the model inputs, (b) a label, (c) a train/test split, plus one row per true
interaction for the efficiency denominator and one row per file for the exposure. `spec/spec_v1.md` fixes all of this
for the ν<sub>e</sub> CC tagger; `spec/spec_v1.py` is the same in code and every script imports it. A different BDT changes
the label, the signal definition and the input list, and keeps everything else.

## 1. Environment

Python 3.9+ with numpy, pandas, pyarrow, uproot (5.6+), xgboost (2.1+), scikit-learn. On Aurora:
`module load frameworks` plus pip --user installs of uproot and xgboost. On the gpvm the grid bundle
(Python 3.9.25, uproot 5.6.9, xgboost 2.1.4) runs every script here. Train with `base_score=0.5` and export JSON: a model
written by xgboost 3.x with a fitted base_score loads in 2.1 with a silent constant shift of the margin.

## 2. Tables (`tables/`)

`build_tables.py SAMPLE OUT --list LIST [--slice a:b] [--nproc N] [--objs] [--root ROOT] [--tag TAG]` reads merged CAFs
and writes `cands_`, `truth_`, `files_` (and `objs_`) parquet files. Per candidate it stores every scalar `T_tagger`
branch (1,246; the 258 shipped ν<sub>e</sub> model inputs are among them), `kine_reco_Enu` and `reco_e_mev` from `T_kine`,
the DL-vertex columns, and the truth match: the best partner of the candidate at `T_match` level 2 (point IoU), its
truth, the SCE-shifted true vertex distance. Per true interaction: kinematics, the true lepton from `T_truth_pf`,
and the truth → reco best match (`nu_ct_*`) for the efficiency loop. Per file: the exposure by the header rule
(`pot_hdr`, `n_bnb_spills`, `n_offbeam_gates`, `ngenevt`) and the flags `label_unreadable`, `truth_pf_unreadable`.

Two uproot limits are handled: in about 0.1 to 0.3 % of files the partner-list baskets of `T_match`, or the E/KE
baskets of `T_truth_pf`, cannot be read; the file is kept and flagged. Data files have an empty or missing `T_match`:
every candidate is then "no partner" (cosmic background), by design.

`merge_tables.py IN SAMPLE OUT` joins slices; `compare_tables.py A B` compares two tables column by column (the
shared-file check: run the builder on the same file on two machines and compare before any large run).

Cost on Aurora: about 5 s per file per process; 100 processes per node; 90,000 files in three one-hour debug jobs.

## 3. Label and FV (`spec/spec_v1.py`)

- Pool: `br_filled == 1` and reco vertex in the FV (the tagger ran; the pre-cut of the selection).
- Signal (ν<sub>e</sub> CC tagger): best partner is a neutrino, reco vertex within 5 cm of its SCE-shifted true vertex,
  |pdg| 12 and CC, true vertex in the FV, deposited energy > 75 MeV. Everything else in the pool is background,
  classed (ν<sub>e</sub> CC with a bad vertex, outside FV or low energy, ν<sub>μ</sub> CC, NC, cosmic).
- FV: the collaboration ν<sub>e</sub> FV (x, y ±190, z 10–450 cm, 5 cm cathode band, East-TPC cut-out), one function
  for truth, reco and evaluation, computed when a table is read.
- Split: even subrun = train, odd = test (beam-off data: by run).

**For another BDT** write a `signal_mask` for its signal (for NC π⁰: a neutrino partner, NC, a true π⁰, vertex rule as
wanted) and keep `pool_mask`, `in_fv`, `background_class` (extend its classes), `true_signal_mask` (the denominator)
and `split`. Put the new function next to the old one with a version note; do not change the ν<sub>e</sub> ones.

## 4. Inputs

`spec/nue_l2_inputs_258.txt` lists the 258 inputs of the shipped ν<sub>e</sub> final model in the order the toolkit's
TMVA reader expects (257 `T_tagger` scalars plus `kine_reco_Enu`). For another BDT list its inputs the same way (the
toolkit's `UbooneNueBDTScorer` / the π⁰ and "other" scorers name them in their `AddVariable` calls); every one is a
`T_tagger` scalar and is already in the candidates table.

## 5. Training (`training/`)

`round0/`: reproduce a known small-sample result with its own recipe before anything else (acceptance test on the
gains). `round1/train_round1.py MERGED OUT SEED`: the shipped structure (XGBoost, depth 30, 600 trees, lr 0.1,
`binary:logitraw`, subsample 0.8, colsample 0.3/0.3, gamma 0.1, no weights) on the train half of every sample, a
learning curve over 1/16 … 1 of the training events, metrics against the stored shipped score on the test half.
`round1/energy_bins.py` compares a model with the shipped weights per true-energy bin at matched overall efficiency,
paired per true neutrino: judge models there, not on AUC alone.

Lessons from the small-sample study that still hold: the label matters more than the model; keep the shipped
small-model scores as inputs (retraining them leaks); never quote the train half; compare at matched efficiency.

### Where the small-sample study differs from this workflow

The recipe here started from a small-sample study, and its numbers are the round-0 reference (`training/round0/`). Compare against it with these differences in mind:
- **Sample:** 21,976 MC events from an older reconstruction (an earlier version of the same Wire-Cell chain): 13,099 BNB CV + 8,877 ν<sub>e</sub> CC. That is 5,384 pool rows, half for training (2,662 rows, 704 background), against about 84,000 training rows here. No beam-off data and no NC Δ-radiative sample.
- **Input format:** per-event `tracking-pr.root` files, not merged CAFs.
- **Label:** a truth match built from true energy deposits. It used the IoU (overlap divided by union) of occupied 1 cm voxels, all cosmic deposits pooled as one truth object, and the candidate's energy segments. This workflow uses `T_match` level 2 instead (point IoU on reconstructed points, one truth object per cosmic particle). The vertex rule, signal definition, pool and background classes are the same. At the same sample size the two labels give the same retraining gain (the round-0 acceptance test).
- **FV:** the box x ±191.05, y ±189.31, z 10.85–450.15 cm with a 5 cm cathode band, and no East-TPC cut-out. It is kept as the `in_wcgpu1_box` column.
- **Split:** the same even/odd subrun rule.
- **Inputs:** the same 258 shipped inputs with the stored small-model scores. Model B added 288 summary statistics of the per-object lists and tied with model A.
- **Training:** the same XGBoost settings, except `base_score`, which xgboost chose there (0.7355) and which is fixed at 0.5 here.
- **Evaluation:** AUC and purity at matched efficiency on the unweighted training mix. There were no POT weights and no beam-off, so its purities are not beam purities.

## 6. Baseline and evaluation (`baseline/`, `evaluation/`)

`baseline_full.py MERGED OUT.json`: the shipped score on the tables, per sample and for the mix, test half and all.
`evaluation/eval_standard.py`: Matteo's efficiency / purity table with the collaboration normalization (POT weights
from the CAF header, the ν<sub>e</sub>CC-filtered MC mixed by spill, beam-off by gates with f = 0.0725, two-loop
efficiency and purity with the SPINE match and no vertex rule, progressive cuts, per-energy-bin curves). It takes a
`--score` column, so a retrained model is evaluated exactly like the shipped one. `docs/evaluation_standard.md`
explains every weight.

## 7. Adapting to the NC π⁰ or the "other" single-photon BDT: the short list

1. Input list: the scorer's variable names, in order, into `spec/<model>_inputs.txt`.
2. Signal and background: a new mask pair in `spec_v1.py`; the denominator for the efficiency.
3. Samples: which campaigns carry the signal (the Aurora NC Δ-radiative campaign and BNB CV for π⁰ / single photon);
   beam-off as the pure-cosmic background.
4. Build the tables once with `build_tables.py` (they already hold every tagger scalar, so no rebuild for a new input
   list), run the baseline with the stored shipped score of that model, then `train_round1.py` with the new names.
5. Evaluate per energy bin at matched efficiency, and through `eval_standard.py` with the signal definition of the
   selection you compare against.
