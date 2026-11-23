# SBND Wire-Cell νe CC BDT: round 4 models for collaborators (2026-11-19)

Three retrained versions of the Wire-Cell νe tagger BDT for SBND, frozen together. Each is the full two-level structure of
the shipped tagger: 15 per-object "small" BDTs (level 1), whose pooled scores feed one candidate-level BDT (level 2). All
three were trained on SBND simulation reconstructed with Wire-Cell toolkit 0.38.0 (production WCLocal2026Nov16, split-move
vertex tree) and replace the MicroBooNE weights shipped with the toolkit.

**Default model (2026-11-20): A v2**, chain files in `nue_round4_ruleA_v2/wirecell/main/`, cut 3.06 on `nue_score` for
82 % purity. A-nogap v2 and B v2 are kept beside it for anyone who wants to compare; all three are frozen, checked in
the chain and evaluated the same way (see "Inside the Wire-Cell chain" below).

> **νμ CC BDT, current release (2026-11-23): `numu_round3af/`.** All four νμ models retrained (numu_1, numu_2, cosmict_10
> and the final model) on simulation whose dQ/dx inputs are scaled by 1.071 towards data, with the numu_1 muon flag
> updated to match: 80.3 % νμ CC efficiency at 95 % purity against 75.0 % for the shipped weights, cut at `numu_score`
> 0.985, beam-on data / prediction 1.063 (shipped 1.058). Drop-in Wire-Cell files under the shipped names, full-chain tested.
> Simulation must be charge-corrected: in the chain with the included toolkit patch (simulation only), or in the tables.
> See [`numu_round3af/README.md`](numu_round3af/README.md).
> The previous νμ release, `numu_round1/` (final model only, cut 1.011, 80.1 %), is kept for comparison.

## The three models

| model | folder / file | small-model labels | final-model inputs | cut for 82 % purity | νe CC efficiency at 82 % purity | e500 eff. / purity |
|---|---|---|---|---:|---:|---:|
| shipped MicroBooNE weights (reference) | in the toolkit | – | 258 | 6.90 | 54.2 % (53.5–55.0) | 62.9 / 79.2 % |
| **A: rule A (main)** | `nue_round4_ruleA/`, `l2/nue_l2.json` | rule A | 258 | **3.01** | **56.1 % (55.2–57.0)** | 65.4 / 79.8 % |
| **A-nogap: rule A without gap_n_bad** | `nue_round4_ruleA/`, `l2/nue_l2_nogap.json` | rule A (same `l1/`) | 257 | 2.53 | 55.8 % (55.0–56.4) | 65.2 / 79.2 % |
| **B: rule B** | `nue_round4_ruleB/`, `l2/nue_l2.json` | rule B | 258 | 3.06 | 54.9 % (54.1–55.7) | 65.2 / 79.2 % |

All numbers on the SBND evaluation samples (Fermilab, not used in training); the range is where the efficiency moves when
the purity curve is shifted by its own statistical error. Beam-on data agree with each model's prediction within the
statistical error.

**Which one to use**
- **A** gives the highest efficiency: about 2 points above the shipped weights at the same purity.
- **A-nogap** leaves out `gap_n_bad`, a final-model input whose distribution disagrees badly between data and simulation
  (mean 3.61 in data against 1.28 in simulation over the pool). It costs at most 0.3 points; it is the safer choice for any
  analysis that relies on data/simulation agreement of the selection.
- **B** differs from A only in how the small models are labelled; it is within the shipped model's range and is provided
  as a cross-check of that choice.

## The two labelling rules for the small models

Each small BDT scores objects inside a candidate (segments, other showers, vertices) and needs a label per object.
- **Rule A** (MicroBooNE's): every object of a signal candidate is signal; an object of a background candidate is used as
  background only where the model's hand-written rule fired on it. Few background objects remain (17 to 1,264 per model).
- **Rule B:** every object takes its candidate's label (14,501 to 631,983 background objects per model).
The 15 hand-written rules are explained, with their physics meaning, in `docs/l1_rules_from_code.md` of the project record.

## How to apply a model (all three)

1. Candidates: the νe tagger ran (`br_filled == 1`).
2. Level 1, per small model in the folder's `manifest.json`: score each object (inputs in the listed order, float32; pio_2
   also takes the candidate's `pio_mip_id`); the candidate value is the **minimum raw margin** over its objects. An empty list,
   or pio_2's pre-cut failing (`pio_filled == 1 and pio_flag_pio == 0`), gives a **missing value (NaN)**, not the shipped
   default.
3. Level 2: the inputs in the manifest's order, `l1_<model>` in the place of the shipped `<model>_score`; missing values
   stay missing; the **raw margin** (xgboost `output_margin`) is the νe score, on the scale of the shipped `nue_score`.
4. Cut at the model's operating point above (or wherever an analysis needs).

The files contain exactly the trees to use. From the builder's tables:

    python release/score_release.py releases/nue_round4_ruleA TABLE_DIR SAMPLE                  # A
    python release/score_release.py releases/nue_round4_ruleA TABLE_DIR SAMPLE --variant nogap   # A-nogap
    python release/score_release.py releases/nue_round4_ruleB TABLE_DIR SAMPLE                  # B

Each reproduces, to the last bit, the scores the evaluation used (checked on 211,416 BNB and νe candidates per model).

## Inside the Wire-Cell chain

The toolkit cannot hold a missing value, so the models above cannot run in the chain as they are. The **v2 folders** are
the versions to use inside wire-cell-toolkit: same small models, final model retrained on the values the chain stores
(each small model's output rescaled, and the toolkit's own "no object" constant for an empty list or pio_2's failed
pre-cut). Their efficiencies agree with v1 within the cut's uncertainty.

| model | folder | chain files | cut for 82 % purity (on `nue_score`) | νe CC efficiency at 82 % purity |
|---|---|---|---:|---:|
| shipped weights (reference) | in the toolkit | – | 6.90 | 54.2 % |
| **A v2** | `nue_round4_ruleA_v2/` | `wirecell/main/` | **3.06** | **56.5 % (55.6–57.3)** |
| A-nogap v2 | `nue_round4_ruleA_v2/` | `wirecell/nogap/` | 2.43 | 54.9 % (53.5–55.8) |
| B v2 | `nue_round4_ruleB_v2/` | `wirecell/main/` | 3.23 | 55.3 % (54.5–56.5) |

Efficiencies from the Fermilab evaluation (bdt-comm `results/round4/eval_fg1_v2/cuts_82.json`); data agree with the
prediction (data/prediction 1.01, 1.06, 1.02).

**To run one:** put the chain folder (for example `releases/nue_round4_ruleA_v2/wirecell/main`) in front of
`wire-cell-data` on `WIRECELL_PATH`. Its `uboone/weights/` holds the 16 νe files under the shipped names (15
`<model>_BDT.weights.xml` and `XGB_nue_seed2_0923.xml`); every other model still resolves to the shipped file. No change
to the toolkit or its configuration. `nue_score` then holds the final model's raw margin, so the cut above applies to it
directly; −15 still means the tagger did not run.

**Do not leave an input empty.** The toolkit's evaluator returns −999 (stored as `nue_score` = −16.2556) if any declared
input is NaN, including one no tree uses. The nogap file still declares `gap_n_bad` (the chain registers 258 inputs); it
must be filled even though it is not used. The chain fills all of them; this matters only for hand-built inputs.

The chain files are written by `release/wirecell/make_chain_weights.py` (from the wcgpu1 session); their checksums are in
each manifest under `wirecell`. Checks:
- full chain on 10 MC files (wcgpu1): level-1 values within 6e-08 of the standalone release, `nue_score` within 7e-07;
- the toolkit's own evaluators on 300 production candidates from Aurora (wcgpu1): within 4e-07 (level 1) and 4e-06
  (final, same level-1 inputs);
- an independent evaluator, `release/wirecell/check_chain_xml.py`, on the same 300 (Aurora): the same bounds, none over 1e-05.

A candidate can move by up to ~0.05 when a last-bit difference in one level-1 value crosses a split of the final model;
this matters only within ~0.05 of the cut. Use the same toolkit version the tables were built with (0.38.0): another
version reconstructs different inputs.

## How they were trained (common to all three)

- **Samples** (Aurora): BNB CV 627k events, νe CC-filtered 269k, NC Δ-radiative 27k (simulation), and the odd-run half of
  the beam-off data (558 files); the even-run beam-off half and the Fermilab simulation are used only for evaluation.
- **Pool:** reconstructed vertex in the collaboration νe fiducial volume (|x| ≤ 190, |y| ≤ 190, 10 ≤ z ≤ 450 cm, excluding the
  cathode band |x| ≤ 5 cm and the corner x < 0, y > 100, z > 250 cm) and the νe tagger run: 171,413 candidates.
- **Label:** signal = charge-matched to a true νe / anti-νe CC, reconstructed vertex within 5 cm of the true vertex, true vertex
  in the FV, deposited energy above 75 MeV; everything else background.
- **Weights:** each sample at its exposure relative to the BNB POT (BNB 1, νe-filtered 0.00596, NC Δ-rad 0.0106, beam-off
  4.91); BNB spills holding a νe CC in the filter volume dropped (the νe-filtered sample stands for them); signal rescaled to
  half the total weight (one third for A-nogap, chosen on the development slice).
- **Learners:** xgboost; small models depth 3, learning rate from 0.05 / 0.1 / 0.3, classes balanced; final model depth 10,
  learning rate 0.1, row subsample 0.8, column subsample 0.3, gamma 0.1; early stopping on a 10 % development slice.
- **Evaluation:** the collaboration's νe CC efficiency / purity (two-loop, mixed by exposure at the BNB POT, beam-off cosmic
  term included) on the Fermilab samples reprocessed with the same production.

## Known issues

- `gap_n_bad` data/simulation disagreement (above); A-nogap avoids it.
- The NC Δ-radiative exposure counts only files with events; the full sample would make its weight about 1.7 times smaller
  (under 0.5 % of the background mix).
- Most of the gain over the previous production comes from the new vertex reconstruction (the shipped weights rise from
  43.1 to 54.2 % at 82 % purity); retraining adds about 2 points on top.

Code of record: this repository (`training/round2/`, `release/`); tag `nue-round4-v1`.
