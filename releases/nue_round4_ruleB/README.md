# SBND Wire-Cell νe CC BDT, round 4, rule B (frozen 2026-11-19)

One of the three round 4 models; read `../README.md` first (what the three are, how to apply them, performance).

| | |
|---|---|
| Model | `l2/nue_l2.json` (258 inputs, 82 trees, depth 10) on the 15 small models in `l1/` (rule B) |
| Output | raw margin (xgboost `output_margin`), on the scale of the shipped `nue_score` |
| Operating point | cut **3.06** = 82 % purity on the SBND evaluation samples |
| νe CC efficiency at 82 % purity | 54.9 % (54.1–55.7); e500 at the same cut 65.2 % / 79.2 % |
| Manifest | `manifest.json`: per model the file, sha256, inputs in order, pooling, pre-cut, trees, learning rate |

**What differs from rule A:** only the labels of the 15 small models. Under rule B every object of a pool candidate that
passes the model's pre-cut takes its candidate's label, so each small model learns from all background objects (14,501 to
631,983 per model on the training split) instead of only those where its hand-written rule fired (17 to 1,264 under rule A).
Everything else (samples, weights, the final model's settings, the development slice) is the same as rule A.

On Aurora's development slice all 15 rule-B small models are ahead of the shipped ones (rule A: 8 of 15), but at the final
model's level rule B ends about 1.2 points of efficiency below rule A on the evaluation samples, within the shipped model's
range. It is provided for comparison and as a cross-check of the labelling choice.

Standalone scoring (`release/score_release.py releases/nue_round4_ruleB TABLE_DIR SAMPLE`) reproduces the evaluation's
scores exactly on 211,416 BNB and νe candidates. Small models: 31 to 449 trees, depth 3, learning rates 0.05 / 0.1 / 0.3 by
early stopping (see the manifest).
