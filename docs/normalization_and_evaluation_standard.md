# Normalization and the evaluation standard (digest, Aurora, 2026-11-11)

Sources: M. Vicenzi, "nueCC update", SBND Wire-Cell meeting 2026-11-10 (pages 2, 3, 5, 8); L. Yates & J. Zennamo,
"Absolute Normalizations for Data and MC Comparisons", SBND physics meeting 2025-04-24; Matteo's repository
`mvicenzi/sbnd-wirecell` (private; `docs/normalization.md`, `docs/sbnd_absolute_normalization.md`, `work/cuts.py`,
`work/effpur.py`, `work/plots.py`, `work/samples.py`), read 2026-11-11. Every number below is copied from those sources.

## 1. Exposure: what each sample is worth

| sample (fg1) | exposure | where it is read | value Matteo uses |
|---|---|---|---|
| BNB MC CV (500k) | POT generated | CAF header `rec.hdr.pot`, first event of each subrun, summed over the files read | 2.715e19 |
| nueCC-filtered MC (100k) | POT generated BEFORE the filter | same header | 2.056e21 |
| beam-on data Run 1 FixedDev | POT = TOR875 summed over spills; gates = spills | `rec.hdr.pot`; `rec.hdr.nbnbinfo` | 4.619e18; 959,609 |
| beam-on data Run 3 FixedDev | same | same | 4.938e18; 1,411,117 |
| beam-off data (Run 1, 50k) | off-beam gates | `rec.hdr.noffbeambnb` | 2,540,692 |

Rules from the recipe: header variables are filled only on the first event of each subrun, sum only those; for data
`TotalPOT` is a default value, never use it; `rec.hdr.spillbnbinfo` is capped at 10 entries, use `rec.hdr.bnbinfo`.
A file that cannot be read contributes no rows and no exposure, so table and exposure stay consistent.
The two beam-on samples are used as one: rows stacked, POT and gates summed (9.557e18 POT, 2,370,726 gates).

## 2. The prediction for beam-on data: three components, two scale rules

| component | sample | weight |
|---|---|---|
| neutrinos in the active volume, and dirt | MC (BNB, or the mixed MC of section 3) | POT_data / POT_MC |
| in-time cosmics | beam-off data | (1 - f) * G_on / G_off |
| data | beam-on | 1 |

f = fraction of beam-on gates that a neutrino triggered (those gates are already in the MC). Matteo uses f = 0.0725
(`plots.F_NU_GATE`); the recipe's default is 0.08 until a data-driven value exists (f = p_BNB - p_off-beam from the
zero-bias streams). The beam-off weight only matters where beam-off candidates survive the selection: at the final
nue cut almost none do (page 9: 1,042 beam-off candidates of 60,800 before the cut, 10 of 540 above score 0).

## 3. The mixed MC: BNB + nueCC-filtered, split by spill

The filtered sample is BNB MC kept only where a spill holds a nue/anti-nue CC interaction in the detector; it covers
76 times the POT of the CV sample. To use both without double counting:

1. Drop from BNB MC every spill whose truth holds a nue/anti-nue CC vertex inside the filter volume
   (`plots.NUECC_FILTER`: the active volume with x widened to +-202.3 cm; inferred from the filtered sample's truth,
   not read from the filter fcl). The whole spill goes, cosmics included.
2. Keep the filtered sample whole.

Weights of the two parts:

| use | BNB part | filtered part |
|---|---|---|
| efficiency, purity, counts | 1 (counts at the BNB POT) | POT_bnb / POT_nuecc = 0.01320 |
| data/MC overlay | POT_data / POT_bnb | POT_data / POT_nuecc |

Fractions are weighted, f = sum(w k) / sum(w n), with the binomial error from sum(w^2) (`plots._wfrac`). The MC
statistical error comes almost entirely from the BNB part (the non-nue background).

## 4. Efficiency and purity: the two-loop definition (medulla / SPINE)

- Truth mode: one row per true neutrino passing the signal definition; follow its best-matched reco object
  (highest point IoU, `nu_ct_*`); efficiency = fraction whose matched candidate passes the selection. No candidate,
  or a cluster no candidate holds, = fail.
- Reco mode: one row per candidate passing the selection; follow its best-matched truth object (`ct_*`);
  purity = fraction whose matched neutrino passes the signal definition. Cosmic, dirt, no partner = not signal.
- Match: the SPINE MatchProcessor charge/point match (the same rule as `T_match`), `MIN_OVERLAP = 0`, the two argmaxes
  independent (not mutual). There is NO 5 cm vertex requirement in the standard metric.
- Intervals: medulla's SpineEfficiency posterior (binomial likelihood, uniform prior, central 68.27 % interval).
- Binning: true E_nu for efficiency, reco E_nu for purity, edges 0, 250, 500, 750, 1000, 1250, 1500, 2000, 2500,
  3000, 6000 MeV (last bin = overflow).

Signal (`cuts.SIGNAL["nueCC"]`): true vertex in the FV, deposited energy > 75 MeV, nue or anti-nue CC.
Selection (`cuts.SELECTED["nueCC"]`), progressive: candidate exists -> reco vertex in the FV -> nue_score > 7.0.
Variant `nueCC_e500` (pages 7-8, for comparison with Pandora and SPINE): signal adds true electron KE > 500 MeV, no
edep cut; selection adds reco electron energy > 500 MeV.

FV (`cuts.FV`, the Pandora/SPINE nueCC FV, SBND technote eq. 6): x in [-190, 190], y in [-190, 190], z in [10, 450]
cm, |x| > 5 cm (cathode band), MINUS the cut-out x < 0, y > 100, z > 250 (top downstream corner of the East TPC).
Same box for the true vertex (signal) and the reco vertex (selection).

## 5. The standard: page 5 of Matteo's report (nueCC, BNB + nueCC-filtered MC, counts at the BNB POT)

| progressive cut | efficiency | purity |
|---|---:|---:|
| candidate exists | 94.3 % | 0.4 % |
| + reco vertex in FV | 91.7 % | 0.5 % |
| + nue_score > 7.0 | **42.4 %** | **83.3 %** |

With the e500 variant (page 8): 94.1 / 91.7 / 53.5 / 50.8 % efficiency and 0.2 / 0.3 / 64.7 / 81.3 % purity,
counts 544.5 / 578.5 signal and 300.1 / 369.1 selected at the final step; Pandora 36.7 / 80.1 %, SPINE 50.2 / 79.3 %.
Cosmic contamination is not included in page 5 (negligible at this selection).

**Decision (user, 2026-11-11): this table, computed this way, is the evaluation standard for every model.** A retrained
model is reported at the same three steps plus its own operating point, with the per-energy-bin curves, on the same
samples with the same weights. Comparisons between models are at matched efficiency as well (spec_v1 section 7).

## 6. How this differs from spec_v1, and what to align

| item | spec_v1 (training) | Matteo (evaluation standard) | action |
|---|---|---|---|
| FV | wcgpu1 box: x +-191.05, y +-189.31, z 10.85-450.15, cathode 5 cm, no cut-out | x +-190, y +-190, z 10-450, cathode 5 cm, cut-out x<0, y>100, z>250 | user to decide: adopt Matteo's FV for the evaluation (required for comparability) and, Aurora's recommendation, for the training too |
| label vertex rule | signal needs reco vertex within 5 cm of the SCE true vertex | no vertex rule; charge match only | keep the 5 cm rule for TRAINING labels (it is what made retraining useful); the evaluation uses Matteo's definition |
| signal energy | edep > 75 MeV | edep > 75 MeV (nueCC); electron KE > 500 MeV (e500) | same; add the e500 variant to the evaluation |
| weights | unweighted | mixed MC by spill, POT weights | evaluation always weighted as in section 3 |
| match | T_match level 2, match_obj[0] | the same SPINE rule on the same points | same |
| operating point | matched efficiency | nue_score > 7.0 | report both: the fixed cut for comparability, matched efficiency for model-to-model |

## 7. Reproduction plan (gpvm, issue #9)

Stage 1: run Matteo's own code unchanged on gpvm (`gh repo clone mvicenzi/sbnd-wirecell`; `work/extract.py part/merge`
on the fg1 lists, then `plots.effpur_with_filtered` / `cut_progression_with_filtered`) and reproduce page 5 and page 8.
Stage 2: compute the same table with our own evaluation code from our tables (spec_v1 columns) and show the two agree
step by step. Only then is our inference "right and comparable with others", and the standard is frozen.
