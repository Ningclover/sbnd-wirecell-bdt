# Digest of the SBND Wire-Cell production catalog (training data on Aurora)

Source: https://haiwangyu.github.io/sbnd-wirecell-production/ (data file `productions.js`, "Last updated 2026-11-09"; a copy is saved next to this file as `catalog_productions_2026-11-09.js`). Paths and counts below were re-checked on Flare on 2026-11-10 with `ls`, `wc -l` and one uproot comparison; nothing else was swept.

## What the catalog is

One page listing every Wire-Cell production, split into two sides:

- **Training (Aurora)**: run over Avinay's privately generated gen2 MC (wct0319). This is our training data.
- **Evaluation (FNAL)**: run at Fermilab over the official Gen2 SAM definitions, MC and data. We cannot reach these from Aurora; the sbndgpvm side can.

Each production has a config label (chain + toolkit build + knobs), an event count, a unit count with OK/PARTIAL/EMPTY/FAIL, an output path, and a status checked on 2026-11-09.

## Training productions on Aurora (all "available")

| Production | Config | Events with WC output | Units | Merged CAFs | Output root | File list |
|---|---|---|---|---|---|---|
| BNB CV 1M, r2 (2026-11-05) | aurora-r2 | 627,114 of 627,205 | 11,888 (11,872 OK, 16 PARTIAL) | 421.0 GB | `production-prep/dlspace/bnb/node<k>/<unit>/merged/` | `dlspace/lists/bnb-merged-cafs.txt` (11,888 lines) |
| BNB CV 1M, r1 (2026-10-22) | aurora-r1 | 627,127 of 627,205 | 11,888 (11,870 OK, 18 PARTIAL) | about 1 TB at semiprod | `production-prep/bnb1m/full/{A,B}/node<k>/<unit>/merged/` | pairs.tsv per campaign |
| nueCC, r2 (2026-11-08) | aurora-r2 | 268,789 of 268,865 | 61,252 (61,225 OK, 23 PARTIAL, 4 FAIL) | 374.9 GB | `dlspace/nuecc/node<k>/<unit>/merged/` | `dlspace/lists/nuecc-merged-cafs.txt` (61,248 lines) |
| nueCC, r1 (2026-11-03) | aurora-r1 | 268,619 of 268,865 | 62,050 (61,159 OK, 798 EMPTY, 83 PARTIAL, 10 FAIL) | 361 GB | `production-prep/rare/nuecc/full/{A,B}/node<k>/<unit>/merged/` | pairs.tsv |
| NC Δ-rad, r2 (2026-11-05) | aurora-r2 | 26,744 of 26,745 | 17,604 (17,603 OK, 1 FAIL) | 59.7 GB | `dlspace/ncdr/node<k>/<unit>/merged/` | `dlspace/lists/ncdr-merged-cafs.txt` (17,603 lines) |
| NC Δ-rad, r1 (2026-11-03) | aurora-r1 | 26,744 of 26,745 | 29,779 (17,603 OK, 12,175 EMPTY, 1 FAIL) | 59 GB | `production-prep/rare/ncdr/full/work/node<k>/<unit>/merged/` | pairs.tsv |

All roots are under `/lus/flare/projects/neutrinoGPU/yuhw/`.

## aurora-r2 versus aurora-r1 (what changed for us)

- Same events. r2 reran only step 2 (pattern recognition, DL vertex, truth match, flat tree) on r1's step-1 output. Checked on one BNB unit: 49 events in both files, identical run/subrun/event set.
- Schema grew from 2,401 to 2,416 branches in recTreeWireCell (md5 af9c87c8). The 15 new branches are all about DL-vertex space points: a new tree `T_dlvtx_space` (runNo, subRunNo, eventNo, cluster_id, blob_index, x, y, z, q, plus `..length`) and new `T_dlvtx_call` fields `n_space`, `space_cluster_id` with its `..idx/..length/..totarraysize`. Nothing was removed.
- In the test unit `T_dlvtx_space` holds 36,060 points over 49 events (0 to 1,238 per event). It is the Bee-style space-point dump for training the DL vertex finder, not something the BDT needs, but it is there if a later model wants cluster geometry.
- Fewer lost events: BNB 91 missing (was 78, 3 units lost 78 in r2), nueCC 76 missing (was 246), NC 1 missing (same).
- Per-event `tracking-pr.root` is still kept under `<unit>/pr/`; Bee files are not.
- Ready-made file lists exist for r2 only. They remove the need to walk pairs.tsv and status files.

## Things the catalog tells us that matter for training

1. **Use aurora-r2 as the training source.** Same events as r1, fewer failures, the superset schema, and a one-file list per species. Step 0 numbers were taken from r1 files; the 19 branches we read are unchanged, so they carry over.
2. **Training and evaluation MC are different productions.** The catalog warns no event-level cross-check has been done between Avinay's private gen2 and the official SAM Gen2. Any model trained here should be evaluated on the FNAL fg1 outputs by the sbndgpvm side. Those are at `/pnfs/sbnd/persistent/users/yuhw/prod-fermigrid/fg1/{mc-cv,mc-nuecc,beam-off}/` (merged CAFs 447 + 123 + 8 GB) and `run3bnb/bnb-data/` (Run 3 unblinded BNB data, 109,415 events, 26.6 GB).
3. **Leakage warning on the FNAL side.** Every r8 pair and r7's events are inside fg1, and older rounds reused the same 1,000-file lists. Not our problem on Aurora, but if any FNAL-side training reuses r7/r8 events it must exclude them from fg1 evaluation. fg1 event lists: `prod-fermigrid/fg1/lists/events-<sample>.tsv.gz`.
4. **Beam-off and beam-on data exist only on the FNAL side** (fg1 beam-off 49,917 events, open FixedDev Run 1 data 89,921 events, Run 3 109,415 events). Cosmic-only and real-data checks therefore belong to sbndgpvm.
5. **Open data note.** The open FixedDev Run 1 production has no tracking-pr.root; the per-event trees are inside its merged CAFs (2,283 branches, a strict subset of fg1's).
6. **The schema hash is written in each unit's `merge.json`**, with the CAF source path and event counts. Any table builder should record it.

## What was not checked

- Only one BNB unit was opened for the r1/r2 comparison. The r2 nueCC and NC lists were counted, not opened.
- The catalog's 2026-11-09 status checks are yuhw's, not ours.

## Update 2026-11-10 (productions.js re-fetched from Aurora, diffed against the 11-09 copy)

- New: `fg1-beamoff-train` (odd runs 18253-18591, 80 runs, 572 pairs: 569 OK, 3 PARTIAL, 27,076 events, 27,067 with WC, 4.40 GB) under `/pnfs/sbnd/persistent/users/yuhw/prod-fermigrid/fg1/beam-off-train/`, symlinks into `../beam-off/`. Side: training.
- New: `fg1-beamoff-eval` (even runs 18250-18592, 76 runs, 486 pairs: 485 OK, 1 PARTIAL, 22,894 events, 22,850 with WC, 3.72 GB) under `.../fg1/beam-off-eval/`. Side: evaluation.
- Changed: `fg1-beamoff` (full, 1,058 files) status `TruthLeakRisk`: contains both halves; never use whole, never evaluate a model trained on the odd half on it.
- New: `ncdr` (fg1 MC NC Delta-radiative, SAM sample, 15,776 pairs: 9,150 OK, 6,626 EMPTY, 13,772 events, 32.4 GB) under `.../prod-fermigrid/ncdr/mc-ncdr/`. Side: evaluation. Not the Aurora NC sample.
- Note added to the catalog: 212 r8 beam-off files (= r7 beam-off events) fall in both halves (123 train, 89 eval); a model trained on r7/r8 beam-off events has seen part of the evaluation half.
- Every beam-off file has art subrun 1, so a subrun split was impossible; the split is by run parity.
