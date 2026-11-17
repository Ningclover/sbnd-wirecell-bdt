# Results so far (ν<sub>e</sub> CC tagger, 2026-11-11)

Samples: aurora-r2 (Avinay's gen2 MC, Wire-Cell on Aurora): BNB CV 627,114 events / 302,146 candidates; ν<sub>e</sub> CC
268,789 / 265,376; NC Δ-radiative 26,744 / 24,030; plus the fg1 beam-off training half 27,067 / 2,556 (pure cosmic).
Collaboration FV, spec_v1 label, even/odd subrun split.

| test half of the ν<sub>e</sub> CC + BNB mix | shipped weights | round 1 (3 seeds) |
|---|---:|---:|
| pool rows / signal / true signal | 80,582 / 58,224 / 97,081 | same |
| AUC | 0.9153 | 0.9695–0.9699 |
| purity at 50 % efficiency | 93.4 % | 97.6–97.8 % |
| purity at 40 % efficiency | 97.7 % | 99.0 % |
| ceiling (tagger reaches the true signal) | 60.0 % | 60.0 % |

Learning curve (AUC on the same test half): 0.931 / 0.946 / 0.958 / 0.965 / 0.970 at 1 / 3 / 10 / 30 / 100 % of the
training events; still rising. Per true-energy bin at matched 50 % efficiency, round 1 gains between 0.5 and 2 GeV and
loses below 500 MeV and above 2.5 GeV; on inclusive BNB alone the purity at 50 % efficiency does not improve
(55.2 → 51.7 %), because 91 % of the training rows are ν<sub>e</sub> CC. Round 2 addresses both.

Round 0 (acceptance): the wcgpu1 small-sample study (r7, 22,000 events) reproduced on an aurora-r2 subset of the same size
with its recipe: AUC gain +0.034 against the reference +0.032 ± 0.005, ceiling 59.7 % against 60.5 ± 0.8 %.

Shipped weights re-scored from the merged-CAF branches with `tools/tmva_grad_forest.py`: bit-identical on 59 of 59
candidates (the CAF holds the complete 258 inputs).

Evaluation standard reproduced: Matteo Vicenzi's nueCC table (fg1, mixed MC at the BNB POT) 94.3 / 91.7 / 42.4 %
efficiency and 0.4 / 0.5 / 83.3 % purity at candidate exists / in FV / nue_score > 7, to the digit with his code on gpvm,
and number for number with `evaluation/eval_standard.py` on a subsample.
