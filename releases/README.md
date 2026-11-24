# SBND Wire-Cell BDT releases: current models (2026-11-24)

Retrained Wire-Cell tagger BDTs for SBND, trained on SBND simulation reconstructed with Wire-Cell toolkit 0.38.0
(production WCLocal2026Nov16). They replace the MicroBooNE weights shipped with the toolkit.

## Current defaults

| channel | default model | Wire-Cell files | select | efficiency at the operating point | shipped weights | beam-on data / prediction |
|---|---|---|---|---:|---:|---:|
| **νe CC** | **`nue_round4_ruleA_v2`** (A v2) | `nue_round4_ruleA_v2/wirecell/main/` | `nue_score > 3.06` (82 % purity) | **56.5 %** (55.6–57.3) | 54.2 % | 1.01 ± 0.07 (shipped 0.91) |
| **νμ CC** | **`numu_round3af`** | `numu_round3af/wirecell/` | `numu_score > 0.985` (95 % purity) | **80.3 %** (80.0–80.5) | 75.0 % | 1.063 ± 0.005 (shipped 1.058) |

- Efficiencies are on the SBND evaluation samples at Fermilab, which were not used in training. The range is where the
  efficiency moves when the purity curve is shifted by its own statistical error.
- Beam-on data: Run 3 plus open Run 1, 9.57e18 POT; statistical errors only.
- **The νμ model needs a charge correction on simulation** (dQ/dx × 1.071, simulation only). In the chain this takes a
  small toolkit patch. Step-by-step instructions are in [Run both defaults in wire-cell-toolkit](#run-both-defaults-in-wire-cell-toolkit).
  The νe model needs no correction and no toolkit change.

## All models in this folder

### νe CC (two-level: 15 small BDTs + one final BDT)

| model | folder | runs in the chain | cut for 82 % purity | νe CC efficiency | data / prediction | status |
|---|---|---|---:|---:|---:|---|
| shipped MicroBooNE weights | in `wire-cell-data` | yes | 6.90 | 54.2 % (53.5–55.0) | 0.91 | reference |
| **A v2** | `nue_round4_ruleA_v2/` (`wirecell/main/`) | **yes** | **3.06** | **56.5 % (55.6–57.3)** | **1.01** | **default** |
| A-nogap v2 | `nue_round4_ruleA_v2/` (`wirecell/nogap/`) | yes | 2.43 | 54.9 % (53.5–55.8) | 1.06 | alternative: leaves out `gap_n_bad` |
| B v2 | `nue_round4_ruleB_v2/` (`wirecell/main/`) | yes | 3.23 | 55.3 % (54.5–56.5) | 1.02 | cross-check of the labelling rule |
| A v1 | `nue_round4_ruleA/` (`l2/nue_l2.json`) | no (needs missing values) | 3.01 | 56.1 % (55.2–57.0) | – | outside the chain only |
| A-nogap v1 | `nue_round4_ruleA/` (`l2/nue_l2_nogap.json`) | no | 2.53 | 55.8 % (55.0–56.4) | – | outside the chain only |
| B v1 | `nue_round4_ruleB/` (`l2/nue_l2.json`) | no | 3.06 | 54.9 % (54.1–55.7) | – | outside the chain only |

- **v1 against v2:** the same small models. The v2 final model is retrained on the values the chain stores, so it runs
  in wire-cell-toolkit unchanged. The v1 final model expects missing values (NaN), which the toolkit cannot hold.
- **A against B:** how the small models were labelled (see [the νe details](#νe-round-4-details) below).
- **A-nogap:** `gap_n_bad` disagrees badly between data and simulation (mean 3.61 in data against 1.28 in simulation).
  Leaving it out costs at most about 1.6 points in v2.
- Signal: charge-matched to a true νe / anti-νe CC, reconstructed vertex within 5 cm of the true vertex, true vertex in the
  fiducial volume, deposited energy above 75 MeV.

### νμ CC (three small BDTs + one final BDT)

| model | folder | retrained | simulation correction | cut for 95 % purity | νμ CC efficiency | data / prediction | status |
|---|---|---|---|---:|---:|---:|---|
| shipped weights | in `wire-cell-data` | – | none | 1.017 | 75.0 % (74.7–75.3) | 1.058 | reference |
| round 1 | `numu_round1/` | final model only | none | 1.011 | 80.1 % (79.8–80.3) | 1.048 | previous release |
| **round 3af** | `numu_round3af/` | **numu_1, numu_2, cosmict_10 and the final model** | **dQ/dx × 1.071, numu_1 muon flag updated** | **0.985** | **80.3 % (80.0–80.5)** | **1.063** | **default** |

- Signal: νμ / anti-νμ CC with the true vertex in the fiducial volume and deposited energy above 100 MeV.
- Variants trained beside round 3af (3a, 3b, 3bf) were within 0.3 points of it; see `numu_round3af/README.md`.
- Round 2 (small models retrained on uncorrected simulation) is not released: its numu_1 learned a muon dQ/dx edge where
  data and simulation differ, and lost about 7 % of the selected data against prediction.

## Run both defaults in wire-cell-toolkit

Both models are drop-in replacements: the weight files carry the shipped file names, so the toolkit configuration finds
them through `WIRECELL_PATH`. The νe model works with stock toolkit 0.38.0. The νμ model needs one patch, for the
simulation charge correction.

### Step 1. Get the files

```bash
git clone https://github.com/Ningclover/sbnd-wirecell-bdt.git
export BDT=$PWD/sbnd-wirecell-bdt/releases
ls $BDT/nue_round4_ruleA_v2/wirecell/main/uboone/weights   # 16 files: 15 <model>_BDT.weights.xml + XGB_nue_seed2_0923.xml
ls $BDT/numu_round3af/wirecell/uboone/weights              # 4 files: numu_tagger1/2, cos_tagger_10, numu_scalars_scores_0923
```

The sha256 of every file is in `nue_round4_ruleA_v2/manifest.json` (under `wirecell`) and in
`numu_round3af/wirecell/chain_weights.json`.

### Step 2. Patch and rebuild the toolkit (needed for νμ on simulation)

The patch is against wire-cell-toolkit tag **0.38.0** (commit c935e30d), the version the models were trained with.

```bash
cd wire-cell-toolkit
git checkout -b numu-sim-dqdx-scale 0.38.0
git am $BDT/numu_round3af/wirecell/patches/wct-0.38.0-numu_sim_dqdx_scale.patch
# rebuild and install as you normally do, e.g.
./wcb configure --prefix=$PREFIX ... && ./wcb install
```

What the patch changes (5 files, 45 lines; default off, so the output is unchanged unless you set the factor):
- `root/src/UbooneNumuBDTScorer.cxx` and `.h`: a new setting `numu_sim_dqdx_scale` (default 1.0). When it is not 1, the
  scorer multiplies the seven dQ/dx inputs by it, in the copies handed to the νμ BDTs only:
  - `numu_cc_1_medium_dQ_dx` of every numu_1 segment;
  - `cosmict_2_dQ_dx_end/front`, `cosmict_3_dQ_dx_end/front`, `cosmict_7_dQ_dx_end/front`, where that tagger is filled;
  - only values that are finite and above 0.

  It also recomputes the νμ muon flag as the training did. A numu_1 segment keeps its flag only if the scaled dQ/dx is
  still below its `numu_cc_1_dQ_dx_cut`. `numu_cc_flag` is switched off if no numu_1 segment flag survives and neither
  `numu_cc_flag_2` nor `numu_cc_flag_3` is on (about 1 % of flagged segments turn off). The stored tagger values
  (`cosmict_*`, `numu_cc_*`) stay raw. Only the νμ BDT scores (`numu_1_score`, `numu_2_score`, `cosmict_10_score`,
  `numu_score`) change.
- `cfg/pgrapher/common/clus.jsonnet`: passes `numu_sim_dqdx_scale` to the scorer.
- `cfg/pgrapher/experiment/sbnd/clus.jsonnet`: new top-level argument `numu_sim_dqdx_scale`. It reaches the scorer only
  when `reality == 'sim'`; for data the scorer always gets 1.0.
- `cfg/pgrapher/experiment/sbnd/wct-pr.jsonnet`: new TLA `numu_sim_dqdx_scale`, forwarded to `clus.jsonnet`.

The νe scorer is not touched: the νe model was trained on uncorrected simulation and must not get the factor.

### Step 3. Put the two weight folders in front of `wire-cell-data`

```bash
export WIRECELL_PATH=$BDT/nue_round4_ruleA_v2/wirecell/main:$BDT/numu_round3af/wirecell:$WIRECELL_PATH
```

- The SBND configuration loads the BDT files as `uboone/weights/<name>` (`bdt_weights_dir` in
  `cfg/pgrapher/experiment/sbnd/clus.jsonnet`), and the toolkit takes the first match along `WIRECELL_PATH`.
- The two folders hold different file names, so their order does not matter. Every file they do not hold (for example
  `numu_tagger3.weights.xml` and all non-BDT data) still comes from `wire-cell-data`.
- `WIRECELL_PATH` must also still contain your configuration (`cfg`) directory and `wire-cell-data`, as before.

### Step 4. Run simulation with the factor, data without

The pattern-recognition step (`wct-pr.jsonnet`) is the one that scores the BDTs.

**Simulation:**
```bash
wire-cell -l stdout -L debug -c pgrapher/experiment/sbnd/wct-pr.jsonnet \
  --tla-str input=qlpctree.tar.gz --tla-str reality=sim \
  --tla-code numu_sim_dqdx_scale=1.071 \
  <your other TLAs as before>
```

**Beam-on and beam-off data:**
```bash
wire-cell -l stdout -L debug -c pgrapher/experiment/sbnd/wct-pr.jsonnet \
  --tla-str input=qlpctree.tar.gz --tla-str reality=data \
  <your other TLAs as before>
```

- `numu_sim_dqdx_scale` must be passed with `--tla-code` (a number), not `--tla-str`.
- With `reality=data` the factor is ignored even if it is passed, so a data job can never be corrected by mistake.
- With `reality=sim` and no `numu_sim_dqdx_scale`, simulation is **not** corrected and the νμ prediction is wrong (see
  "Why the factor matters" below).
- If you drive the toolkit another way (your own top-level jsonnet, a LArSoft/art job), pass `numu_sim_dqdx_scale=1.071`
  to `pgrapher/experiment/sbnd/clus.jsonnet` together with `reality='sim'`, exactly as `wct-pr.jsonnet` does.

**Aurora production workflow (`sbnd-wirecell-production`):** apply
`numu_round3af/wirecell/patches/step2-numu_sim_dqdx_scale.patch` in that repository (`git apply`). It makes
`aurora/bin/step2.sh` pass the environment variable `NUMU_SIM_DQDX_SCALE` as the TLA, and adds it to the variables
forwarded into the container (`aurora/site/aurora.sh`). Then set it for simulation jobs only:
```bash
export NUMU_SIM_DQDX_SCALE=1.071   # simulation jobs; leave unset for data
```

### Step 5. Check the job

1. **The factor is on (simulation only).** The job log contains
   `UbooneNumuBDTScorer: simulation dQ/dx scale 1.071 on the numu BDT inputs ...`.
   It must appear in simulation logs and must not appear in data logs.
2. **The new weights are found first.** The toolkit does not log which weight file it opened, so check the path
   lookup in the job's environment. Each file must resolve into the release folders:
   ```bash
   for f in XGB_nue_seed2_0923.xml br3_3_BDT.weights.xml numu_scalars_scores_0923.xml numu_tagger1.weights.xml; do
     for d in ${WIRECELL_PATH//:/ }; do [ -f "$d/uboone/weights/$f" ] && { echo "$f -> $d"; break; }; done
   done
   ```
   The first two must print `.../nue_round4_ruleA_v2/wirecell/main`, the last two `.../numu_round3af/wirecell`.
   A changed `nue_score` or `numu_score` against a run with the shipped weights also shows it.
3. **Optional, against our tables.** `training/numu/compare_chain_vs_table.py` compares the chain's stored `numu_score`
   with the release scored from the table builder's tables of the same output. Our full-chain test (2026-11-23) agreed to
   6e-06, with no candidate moving across the cut: 82 BNB simulation candidates on Aurora, 31 at Fermilab, and 34 data
   candidates. Results are in `numu_round3af/checks/chain/`.

### Step 6. Select

| channel | candidates | selection | operating point |
|---|---|---|---|
| νe CC | the νe tagger ran (`br_filled == 1`; `nue_score` = −15 means it did not) | `nue_score > 3.06` | 82 % purity |
| νμ CC | every candidate with a neutrino vertex | `numu_score > 0.985` | 95 % purity |

Both scores are on the shipped scale. Apply the same reconstructed fiducial-volume cut as the evaluation where an analysis
needs it.

### Why the factor matters, and what not to do

- Data and simulation disagree on the muon charge (dQ/dx): data are about 7 % higher. The νμ models were trained on
  simulation with the seven dQ/dx inputs scaled by 1.071 (fitted by gpvm to data / prediction on those inputs).
- Without the factor, the number of selected BNB simulation candidates hardly changes (+0.09 %), but 9,049 of 302,172
  candidates move across the cut. So the efficiency and the data / prediction ratio quoted above no longer hold.
- **Never correct twice.** The chain applies the factor only to the copies handed to the BDTs; the stored `cosmict_*` and
  `numu_cc_*` values stay raw, and the stored `numu_score` is already corrected. If you re-score from tables built on
  chain output, correct the tables first
  (`training/numu/make_corrected_tables.py MERGED OUT releases/numu_round3af/correction.json --samples <simulated samples> --update-flag1`).
  Don't apply the factor again to a `numu_score` the chain has already stored.
- **Never correct data**, beam-on or beam-off.
- **Use toolkit 0.38.0** (the patched copy). Another version reconstructs different inputs.

## Outside the chain

Both models are also plain xgboost files, for scoring the table builder's tables:

```bash
# νe A v2 (or any νe folder)
python release/score_release.py releases/nue_round4_ruleA_v2 TABLE_DIR SAMPLE
# νμ round 3af: correct simulation first, then score (column d6_f0 = margin; numu_score = 0.868589 × margin)
python training/numu/make_corrected_tables.py MERGED CORRECTED releases/numu_round3af/correction.json \
    --samples wcl1116-bnb wcl1116-ncdr --update-flag1
python training/numu/score_numu_release.py releases/numu_round3af CORRECTED SAMPLE OUT.parquet
```

Details are in each folder's README.

## Where to read more

- `numu_round3af/README.md`: νμ training, the correction, file checks, full-chain test, variants, limitations.
- `nue_round4_ruleA_v2/README.md`: νe A v2 chain files, rescaling of the small models, operating point.
- The νe details below: labelling rules, how to apply outside the chain, chain checks, training, known issues.

---

## νe round 4 details

The three νe versions (A, A-nogap, B) were frozen together (2026-11-19). Each is the full two-level structure of the
shipped tagger: 15 per-object "small" BDTs (level 1), whose pooled scores feed one candidate-level BDT (level 2). The
first table above gives their performance; e500 efficiency / purity of v1: shipped 62.9 / 79.2 %, A 65.4 / 79.8 %,
A-nogap 65.2 / 79.2 %, B 65.2 / 79.2 %.

### The two labelling rules for the small models

Each small BDT scores objects inside a candidate (segments, other showers, vertices) and needs a label per object.
- **Rule A** (MicroBooNE's): every object of a signal candidate is signal; an object of a background candidate is used as
  background only where the model's hand-written rule fired on it. Few background objects remain (17 to 1,264 per model).
- **Rule B:** every object takes its candidate's label (14,501 to 631,983 background objects per model).
The 15 hand-written rules are explained, with their physics meaning, in `docs/l1_rules_from_code.md` of the project record.

### How to apply a model (all three)

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

### Inside the Wire-Cell chain (νe v2 files)

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

### How they were trained (common to all three)

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

### Known issues

- `gap_n_bad` data/simulation disagreement (above); A-nogap avoids it.
- The NC Δ-radiative exposure counts only files with events; the full sample would make its weight about 1.7 times smaller
  (under 0.5 % of the background mix).
- Most of the gain over the previous production comes from the new vertex reconstruction (the shipped weights rise from
  43.1 to 54.2 % at 82 % purity); retraining adds about 2 points on top.

Code of record: this repository (`training/round2/`, `release/`); tag `nue-round4-v1`.
