# SBND Wire-Cell νe CC BDT: round 4 models for collaborators (2026-11-19)

Three retrained versions of the Wire-Cell νe tagger BDT for SBND, frozen together. Each is the full two-level structure of
the shipped tagger: 15 per-object "small" BDTs (level 1), whose pooled scores feed one candidate-level BDT (level 2). All
three were trained on SBND simulation reconstructed with Wire-Cell toolkit 0.38.0 (production WCLocal2026Nov16, split-move
vertex tree) and replace the MicroBooNE weights shipped with the toolkit.

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

The converted versions that wire-cell-toolkit runs, and the check that the chain gives the same scores as the standalone
models on a few files (`release/compare_chain.py`), are being prepared; they will be added here as `<folder>/wirecell/`
with the toolkit commit that reads them. Until then the models are applied to the builder's tables as above.

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
