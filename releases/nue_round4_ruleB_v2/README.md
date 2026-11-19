# SBND Wire-Cell νe CC BDT, round 4, rule B, v2: chain version (2026-11-19)

The version of `../nue_round4_ruleB/` that runs in the Wire-Cell chain with **no toolkit change** (issue #22).

- **Small models (`l1/`): byte-identical to v1.** wcgpu converts them to TMVA XML with a per-model rescale
  (`level1.chain_transform` in `manifest.json`), so the chain stores `float32(tanh(scale * margin + offset))` in
  `<model>_score`, and the toolkit's own empty-list constant (0.3 for br3_3, 0.2 for pio_2, …), which no real object can
  reach, marks "no object" in place of v1's missing value.
- **Final model (`l2/nue_l2.json`, 258 inputs, 103 trees): retrained** on exactly those stored values, same samples, weights,
  settings, seed and development slice as v1. It never sees a missing value, so it converts to TMVA XML exactly.
  
- **Output:** raw margin; the converted file writes it directly into `nue_score`.
- **Operating point:** to be set at 82 % purity by the Fermilab evaluation (pending).

Scoring from the builder's tables, with the transform applied from the manifest:
`python release/score_release.py releases/nue_round4_ruleB_v2 TABLE_DIR SAMPLE [--variant nogap]`.
It reproduces the training-time development scores bit for bit (nuecc and BNB).
