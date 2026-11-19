# SBND Wire-Cell νe CC BDT, round 4, rule A (frozen 2026-11-19)

A retrained version of the Wire-Cell νe tagger BDT for SBND: the 15 per-object "small" BDTs (level 1) and the final
candidate BDT (level 2), both retrained on SBND simulation reconstructed with Wire-Cell toolkit 0.38.0 (production
WCLocal2026Nov16, split-move vertex tree). It replaces the MicroBooNE weights shipped with the toolkit.

| | |
|---|---|
| Status | frozen release candidate; standalone scoring verified; Wire-Cell chain conversion and validation in progress |
| Main model | `l2/nue_l2.json` (258 inputs) on the 15 small models in `l1/` |
| Alternative | `l2/nue_l2_nogap.json`: the same without `gap_n_bad` (257 inputs), see "Known issues" |
| Output | raw margin (xgboost `output_margin`): higher is more νe-like, on the scale of the shipped `nue_score` |
| Operating point | cut **3.01** on the main model = 82 % purity on the SBND evaluation samples (2.53 for the alternative) |
| Format | plain xgboost JSON (written by xgboost 3.4.1; read and scored identically by 2.1.4) |
| Manifest | `manifest.json`: per model the file, sha256, inputs in order, pooling, pre-cut, trees, learning rate |

## Performance (SBND simulation, Fermilab evaluation samples, never used in training)

Evaluation samples: BNB 500k + νe CC 100k simulation from the official Gen2 SAM definitions, reprocessed with the same
Wire-Cell production at Fermilab, mixed by exposure at the BNB POT (νe CC spills of the BNB sample replaced by the νe CC
sample), the beam-off cosmic term included. Efficiency over true νe / anti-νe CC with the true vertex in the fiducial
volume and more than 75 MeV deposited; purity over selected candidates (the collaboration's two-loop definition).

| model | cut | νe CC efficiency at 82 % purity | e500 efficiency / purity at the same cut |
|---|---:|---:|---:|
| shipped MicroBooNE weights | 6.90 | 54.2 % (53.5–55.0) | 62.9 % / 79.2 % |
| **this release** | **3.01** | **56.1 % (55.2–57.0)** | **65.4 % / 79.8 %** |
| this release, without gap_n_bad | 2.53 | 55.8 % (55.0–56.4) | 65.2 % / 79.2 % |

The range is where the efficiency moves when the purity curve is shifted by its own statistical error. e500: true electron
kinetic energy above 500 MeV and a reconstructed shower energy above 500 MeV, the definition used to compare with Pandora
and SPINE. On the same samples the shipped weights at their usual cut of 7 give 53.8 % / 82.7 % (νe CC) and
62.5 % / 79.8 % (e500). Beam-on data agree with the prediction of this release within the statistical error.

## Training

- **Samples** (Aurora, ALCF): BNB CV 627k events, νe CC-filtered 269k, NC Δ-radiative 27k (all simulation, Wire-Cell
  0.38.0 production WCLocal2026Nov16), and the odd-run half of the beam-off data (558 files; the even-run half is used only
  for evaluation; 14 hold-out files excluded).
- **Pool:** candidates with the reconstructed vertex in the fiducial volume and the νe tagger run (171,413).
- **Fiducial volume:** the collaboration νe FV (Pandora / SPINE νe CC): |x| ≤ 190, |y| ≤ 190, 10 ≤ z ≤ 450 cm, excluding the
  cathode band |x| ≤ 5 cm and the corner x < 0, y > 100, z > 250 cm.
- **Label:** signal = candidate charge-matched to a true νe / anti-νe CC, reconstructed vertex within 5 cm of the true vertex
  (space-charge corrected), true vertex in the FV, deposited energy above 75 MeV. Everything else is background.
- **Weights:** each sample at its exposure relative to the BNB POT (BNB 1, νe-filtered 0.00596, NC Δ-rad 0.0106, beam-off
  (1 − f)·G_on/G_off = 4.91 with f = 0.0725); BNB spills holding a νe CC in the filter volume are dropped, since the
  νe-filtered sample stands for them; then the signal is rescaled to half the total weight.
- **Level 1:** per small model, every object of a pool candidate passing the model's pre-cut is a row; label rule A
  (MicroBooNE's): signal objects all kept, background objects only where the model's hand-written rule fired. The same
  exposure weights, then the two classes balanced. xgboost, depth 3, learning rate chosen from 0.05 / 0.1 / 0.3, early
  stopping (patience 50) on a development slice.
- **Level 2:** the shipped 258 inputs with the 15 shipped small-model scores replaced by the retrained ones; xgboost depth
  10, learning rate 0.1, row subsample 0.8, column subsample 0.3 per level and per node, gamma 0.1, binary logistic,
  early stopping on the development slice (109 trees).
- **Development slice:** MC subrun % 10 == 0, beam-off run % 10 == 1 (about 10 %), used only for settings.

## How the model is applied (must be reproduced exactly)

1. Candidates: the νe tagger ran (`br_filled == 1`).
2. **Level 1**, for each of the 15 models in `manifest.json`: score every object of the candidate's list with the model
   (inputs in the listed order, float32; pio_2 also takes the candidate scalar `pio_mip_id`); the candidate's value is
   the **minimum** raw margin over its objects. If the list is empty, or the candidate fails the model's pre-cut
   (only pio_2: `pio_filled == 1 and pio_flag_pio == 0`), the value is **missing (NaN)**, *not* the shipped default.
3. **Level 2:** the 258 inputs in the manifest's order, `l1_<model>` taking the place of `<model>_score`; missing values
   stay missing (xgboost's default direction); the raw margin is the νe score.
4. The trees in the files are exactly the trees to use (already cut to early stopping's best iteration).

Standalone scoring from the builder's tables: `python release/score_release.py releases/nue_round4_ruleA TABLE_DIR SAMPLE`.
Checked on 211,416 BNB and νe candidates: identical, to the last bit, to the scores the evaluation used.

## Validation inside the Wire-Cell chain

`python release/compare_chain.py releases/nue_round4_ruleA TABLE_DIR SAMPLE` compares, candidate by candidate, the scores a
chain run stores (`nue_score`, the 15 `<model>_score`) with the standalone release on the same events (tables built from
the chain's output with `tables/build_tables.py --objs`). Status: the conversion to the toolkit's format is in progress;
results will be recorded here.

## Known issues

- **gap_n_bad** (a level-2 input) disagrees badly between data and simulation on the new production: mean 3.61 in data
  against 1.28 in simulation over the pool, before any cut. The alternative model without it loses at most 0.3 points of
  efficiency. Which one becomes the default is open.
- **NC Δ-radiative weight:** its exposure is taken from the files with events only; the full sample's POT would give a
  weight about 1.7 times smaller. The effect on the background mix is below 0.5 %; kept for comparability with round 3.
- **Gain over the shipped weights is modest:** about 2 points of efficiency at 82 % purity; most of the improvement over
  the previous production comes from the new vertex reconstruction, which lifts the shipped weights by 11 points.

## Provenance

Training: `training/round2/train_l1.py` (rule A) and `train_l2.py` in this repository, job `round4.pbs` on Aurora,
2026-11-19. Frozen with `release/freeze_release.py` (every model checked to give identical margins after cutting the
unused trees). Code of record: this repository; the full record of rounds 0–4 is kept by the authors.
