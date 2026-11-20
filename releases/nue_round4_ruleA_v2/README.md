# SBND Wire-Cell νe CC BDT, round 4, rule A, v2: chain version (2026-11-19)

**The default model** (2026-11-20): `wirecell/main/`, cut 3.06. `wirecell/nogap/` (A-nogap v2) and `../nue_round4_ruleB_v2/` are kept as alternatives.

The version of `../nue_round4_ruleA/` that runs in the Wire-Cell chain with **no toolkit change** (issue #22).

- **Small models (`l1/`): byte-identical to v1.** wcgpu converts them to TMVA XML with a per-model rescale
  (`level1.chain_transform` in `manifest.json`), so the chain stores `float32(tanh(scale * margin + offset))` in
  `<model>_score`, and the toolkit's own empty-list constant (0.3 for br3_3, 0.2 for pio_2, …), which no real object can
  reach, marks "no object" in place of v1's missing value.
- **Final model (`l2/nue_l2.json`, 258 inputs, 116 trees): retrained** on exactly those stored values, same samples, weights,
  settings, seed and development slice as v1. It never sees a missing value, so it converts to TMVA XML exactly.
  Also `l2/nue_l2_nogap.json`: the same without `gap_n_bad` (257 inputs, 78 trees, signal share 1/3).
- **Output:** raw margin; the converted file writes it directly into `nue_score`.
- **Operating point (82 % purity, Fermilab evaluation, gpvm 2026-11-19):** main cut **3.06**: νe CC efficiency **56.5 %**
  (55.6–57.3), e500 65.7 % / 79.9 %; without gap_n_bad cut **2.43**: 54.9 % (53.5–55.8). v1 gave 56.1 % / 55.8 %; the
  differences are inside the ~1 point the cut can move. Shipped weights: 54.2 % (cut 6.90). Data agree with the prediction.

Scoring from the builder's tables, with the transform applied from the manifest:
`python release/score_release.py releases/nue_round4_ruleA_v2 TABLE_DIR SAMPLE [--variant nogap]`.
It reproduces the training-time development scores bit for bit (nuecc and BNB).

## Inside the Wire-Cell chain (added 2026-11-20)

`wirecell/<variant>/uboone/weights/` (variants: main, nogap) holds the 16 files wire-cell-toolkit reads, under the shipped names.
Put `wirecell/<variant>` in front of `wire-cell-data` on `WIRECELL_PATH`; nothing else changes. `nue_score` is then the
raw margin, so the operating cut in the manifest applies directly. Every declared input must be filled: one NaN makes
the toolkit return `nue_score` = −16.2556. Written by `release/wirecell/make_chain_weights.py`; checksums and the checks
(chain test, 300 production candidates scored with the toolkit's own evaluators and with an independent one) are in
`manifest.json` under `wirecell`. See `releases/README.md`, "Inside the Wire-Cell chain".
