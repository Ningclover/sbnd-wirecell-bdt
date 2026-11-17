# The evaluation standard: normalization and the efficiency / purity table

`evaluation/eval_standard.py` computes selection efficiency and purity the way the SBND Wire-Cell ν<sub>e</sub> CC study
reports them, following the SBND recipe for absolutely normalized data/MC comparisons (L. Yates and J. Zennamo, SBND
physics meeting, April 2025). This file states the method so the script can be read; the numbers of any particular
production belong with that production's record, not here.

## Exposure per sample

| sample | exposure | read from the CAF header, first event of each subrun only |
|---|---|---|
| BNB MC | POT generated | `rec.hdr.pot` |
| ν<sub>e</sub>CC-filtered MC | POT generated before the filter | `rec.hdr.pot` |
| beam-on data | POT (TOR875 summed over spills); beam gates = spills | `rec.hdr.pot`; `rec.hdr.nbnbinfo` |
| beam-off data | off-beam gates | `rec.hdr.noffbeambnb` |

Never use `TotalPOT` for data, and do not use `rec.hdr.spillbnbinfo` (capped at 10 entries). A file that cannot be read
contributes no rows and no exposure, so tables and exposure stay consistent. The table builder writes these as
`pot_hdr`, `n_bnb_spills`, `n_offbeam_gates`, `ngenevt` in the `files_` table.

## The prediction for beam-on data

Three components, two scale rules:

- neutrinos in the active volume and dirt: MC, weight POT<sub>data</sub> / POT<sub>MC</sub>;
- in-time cosmics: beam-off data, weight (1 − f) · G<sub>on</sub> / G<sub>off</sub>, with f the fraction of beam-on gates
  that a neutrino triggered (those gates are already in the MC); f is measured from the zero-bias streams as
  p<sub>BNB</sub> − p<sub>off-beam</sub>, with 0.08 as the default until then;
- data: weight 1.

## Two MC samples as one, split by spill

The ν<sub>e</sub>CC-filtered sample is BNB MC kept where a spill holds a ν<sub>e</sub>/anti-ν<sub>e</sub> CC interaction in the
detector. To use it with the inclusive BNB MC without double counting: drop from BNB MC every spill whose truth holds such
an interaction inside the filter volume (the TPC box), the whole spill including its cosmics; keep the filtered sample
whole. For efficiency and purity the BNB part has weight 1 and the filtered part POT<sub>bnb</sub> / POT<sub>nuecc</sub>,
so counts are "at the BNB POT". Fractions are weighted, f = Σ w k / Σ w n, with the binomial error from Σ w².

## Efficiency and purity: two loops

- Truth loop: one row per true neutrino passing the signal definition; follow its best-matched reco object by point
  IoU (the SPINE match rule, as written in `T_match`); efficiency = fraction whose matched candidate passes the
  selection. No candidate counts as a failure.
- Reco loop: one row per candidate passing the selection; follow its best-matched truth object; purity = fraction
  whose neutrino passes the signal definition. Cosmic, dirt, no partner = not signal.
- The two directions are independent (not mutual); minimum overlap 0; no vertex-distance requirement.
- Intervals: binomial posterior with a uniform prior, central 68.27 %. Bins: true E<sub>ν</sub> for efficiency, reco
  E<sub>ν</sub> for purity, edges 0, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 6000 MeV.

Signal for the ν<sub>e</sub> CC selection: true vertex in the FV, deposited energy above 75 MeV, ν<sub>e</sub> or
anti-ν<sub>e</sub> CC. Selection, progressive: candidate exists, reco vertex in the FV, `nue_score` above the cut. The
e500 variant (for comparison with other reconstructions) requires true electron kinetic energy above 500 MeV and the
reco electron energy above 500 MeV. FV: `spec_v1.in_fv`. Empty placeholder rows of `T_tagger` are dropped before the
"candidate exists" count (`spec_v1.placeholder_mask`).

`eval_standard.py --score COLUMN` evaluates a retrained model through the same code as the shipped score.
