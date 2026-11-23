# SBND Wire-Cell νμ CC BDT: round 1 (2026-11-20)

> **Superseded on 2026-11-23 by [`../numu_round3af/`](../numu_round3af/README.md)**, which retrains all four νμ models on charge-corrected simulation. This release stays for comparison.

A retrained final νμ BDT for SBND. It replaces the MicroBooNE weights of the toolkit's νμ tagger
(`numu_scalars_scores_0923.xml`). Only the final, candidate-level model is new. The small per-object models (numu_1,
numu_2, cosmict_10) are the shipped ones, so the chain computes every input exactly as before.

Trained on SBND simulation reconstructed with Wire-Cell toolkit 0.38.0 (production WCLocal2026Nov16).

## Performance at 95 % purity

Measured on the SBND evaluation samples at Fermilab, which were not used in training. The efficiency range is where the
efficiency moves when the purity curve is shifted by its own statistical error.

| score | cut | νμ CC efficiency | range | beam-on data / prediction |
|---|---:|---:|---|---:|
| shipped `numu_score` | 1.017 | 75.0 % | 74.7–75.3 | 1.058 ± 0.005 |
| **round 1** | **1.011** | **80.1 %** | 79.8–80.3 | 1.048 ± 0.005 |

- Signal is νμ / anti-νμ CC with the true vertex in the fiducial volume and deposited energy above 100 MeV.
- The gain is largest at low energy. For neutrino energies of 250–500 MeV the efficiency rises from 47 % to 66 %;
  above 1.5 GeV the two models agree within about 1 point.
- Data and prediction agree with this model as well as with the shipped one.

## Use it in Wire-Cell

`wirecell/uboone/weights/numu_scalars_scores_0923.xml` (466 trees, 74 inputs, sha256 `b06ae503…dfba`, full value in
`manifest.json`). It keeps the shipped file name, so no code or configuration change is needed. The SBND
configuration names it in `cfg/pgrapher/experiment/sbnd/clus.jsonnet` (`numu_xgboost_xml`). Either:

- replace `wire-cell-data/uboone/weights/numu_scalars_scores_0923.xml` with this file; or
- put `releases/numu_round1/wirecell` in front of `wire-cell-data` on `WIRECELL_PATH`. All other weight files still
  come from `wire-cell-data`.

The job log confirms the file it loaded: `fast_xgb_forest booked …/numu_scalars_scores_0923.xml (466 trees, 44344
nodes, 74 vars)`. The chain stores `numu_score` on the shipped scale. **Select νμ CC with `numu_score > 1.011`**
for 95 % purity.

Checks done before release:

| check | result |
|---|---|
| converted file against xgboost, 6,251 candidates | max \|diff\| 6.2e-06 on `numu_score` |
| full chain, 104 MC events (45 candidates), against the scorer below | max \|diff\| 3.8e-06; the same 34 candidates pass the cut |
| all 74 inputs in that run, against the same events with the shipped weights | identical |
| toolkit's own C++ scorer, 145 test candidates from the training tables | max \|diff\| 4.4e-06; pass/fail agrees on all 145 |
| the same with the shipped file, against the `numu_score` production stored | identical |

## Use it outside Wire-Cell

`l2/numu_l2.json` is the plain xgboost model (xgboost 3.4.1; it also loads with 2.1).
1. Take the 74 inputs in the order of `manifest.json` (`inputs`, the same as `spec/numu_l2_inputs_74.txt`) as
   float32. Use the shipped `numu_1_score`, `numu_2_score` and `cosmict_10_score`.
2. Set four inputs to 0 where they are missing: `cosmict_4_angle_beam`, `cosmict_7_angle_beam`, `cosmict_7_theta`,
   `cosmict_7_phi`. This mirrors the chain's protection; the production has no missing values.
3. Compute `margin = predict(..., output_margin=True)`. Then `numu_score = 0.868589 × margin`, and the cut is margin 1.164.

From the table builder's candidate tables:

    python training/numu/score_numu_release.py releases/numu_round1 TABLE_DIR SAMPLE OUT.parquet   # column d6_f0 = margin

## How it was trained

`training/numu/train_numu_l2.py`, frozen with `training/numu/freeze_numu_l2.py`.

- **Samples:** BNB simulation; NC events from the NC-enhanced (ncdr) sample as background only; beam-off data for
  cosmic background.
- **Weights:** each sample is weighted to the same exposure.
- **Learner:** binary logistic xgboost, depth 6, learning rate 0.1, subsample 0.8, column sampling 0.5 per level and
  per node, gamma 0.2. Training stopped early on a held-out development slice and kept 466 trees.
- **Selected events (simulation):** 209,294 for training and 23,395 for development.

## Known limitation

A version that also retrained the three small models gained only 0.4 points in simulation and lost about 7 % of the
selected beam-on events relative to prediction. That version is not released. The cause is the muon
dQ/dx input of the numu_1 small model (`numu_cc_1_medium_dQ_dx`): data and simulation disagree in shape because the
charge calibration differs upstream of the tagger. This release does not use any retrained small model.
Round 1 still reads dQ/dx through `cosmict_2/3/7_dQ_dx_end/front` (about 6 % of its total gain). Its data/prediction
ratio matches the shipped model's, but this is being studied.
