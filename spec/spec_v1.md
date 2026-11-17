# spec_v1: the shared specification for tables, label, FV, split and scoring

Status: draft by Aurora, 2026-11-10; FV changed to the collaboration nu_e FV on 2026-11-11; review changes of wcgpu and gpvm folded in on 2026-11-11 (sections 1, 3.1, 3.4, 3.8, 5, 6, 7). Frozen when both say so. Becomes the plan of record when both say "no objection"; changes after that get a new version number.

Code: `spec_v1.py` next to this file. Both servers import it; nobody re-implements the rules.

## 1. Inputs

| sample id | role | source | list |
|---|---|---|---|
| `aurora-r2-bnb`, `aurora-r2-nuecc`, `aurora-r2-ncdr` | training MC | Aurora, `dlspace/lists/{bnb,nuecc,ncdr}-merged-cafs.txt` | 11,888 / 61,248 / 17,603 merged CAFs |
| `fg1-beam-off-train` | training data, purely cosmic: every candidate is background. The MC samples all contain a neutrino interaction; real data sometimes has none, and this sample supplies those events (user, 2026-11-10) | Fermilab odd-run half, copied to Aurora `data/fg1-beam-off-train/`, verified | 572 files; 14 hold-out files dropped -> 558; 2,556 candidates, 119 with tagger inputs |
| `fg1-mc-cv`, `fg1-mc-nuecc`, `fg1-mc-ncdr`, `fg1-beam-off-eval`, `run3-bnb-data`, `open-run1-bnb-data` | inference and evaluation | the inference-host file lists | hold-out files dropped; `r8` flagged. Tables built on gpvm with the SAME builder (`tables/build_tables.py`), user decision 2026-11-11; the evaluation standard also needs his event set once, hold-out and r8 included |

Rules: hold-out files (Xin's list, by file id for MC, by event for data) are never trained on and are dropped from every list. The full `fg1/beam-off/` sample is never used. The blinded gen2 BNB Run 1 data is never used. Files flagged `r8` are kept but excluded from any comparison against the wcgpu model.

## 2. Key

Every row of every table carries `sample`, `file_id` (the merged-CAF file name; the aurora-r2 unit id is its `reco2-...` stem), `run`, `subrun`, `event`, `cand` (= `wc.T_tagger.nu_index`). (run, subrun, event) alone is not unique in fg1 nu_e CC (19 collisions), so `file_id` is always part of the key. Per file, `wct_version` (`wc.Trun.wct_version`) is stored.

## 3. Label (per candidate), from `wc.T_match` level 2

1. Reco row: the level-2, side-0 row whose `nu_index` equals the candidate's `nu_index`. If there is none, or its `nmatch == 0`: **no partner -> cosmic background**. Data files have an empty `T_match` (Run 3, beam-off) or no `T_match` tree at all (open Run 1, 2,283 branches): both mean "no partner" for every candidate, never an error and never `label_unreadable`. Every beam-off candidate is cosmic background by this rule, which is its purpose in the training. A partner row that is referenced but not found (`partner_kind = 'missing'`) is classed as cosmic too and its count is reported per sample next to `label_unreadable`; above zero it points to a reading problem.
2. Partner: `match_obj[0]` of that row (the partner with the largest point IoU `match_iou_n`; ties broken by `match_iou_q` by the writer). Look it up among the level-2, side-1 rows by `obj`. If that row has `truth_nu_idx >= 0`: **neutrino partner**, truth taken from the `T_truth_nu` row with that `nu_idx`. Otherwise (`truth_nu_idx < 0`, a G4 prong) or if the row is not found: **cosmic background**.
3. Stored match-quality columns: `match_iou_n`, `match_iou_q`, `match_shared_n`, `match_shared_q` of the best partner, `mutual`, `nmatch`, and the second partner's `match_iou_n` and `match_iou_q` (`iou_n_2`, `iou_q_2`; null when `nmatch < 2`). A quality cut or an alternative partner rule is then a column change.
4. Vertex: `dist_sce_cm` = distance between the tagger vertex (`wc.T_tagger.nu_x/y/z`) and the SCE-shifted true vertex of the partner neutrino, `wc.T_dlvtx_call.truth_all_reco_x/y/z` of the first call, entry `truth_all_nu_idx == nu_idx`. When the file has no shifted vertex for that neutrino, `dist_sce_cm` falls back to the raw distance and `sce_fallback = 1` is set (wcgpu's rule; 0 of 5,384 pool rows in its study). Also stored: `dist_raw_cm` (against `T_truth_nu.vtx_x/y/z`) and the 1 cm / 10 cm flags for variants.
5. Signal: neutrino partner, `dist_sce_cm < 5`, |pdg| == 12 and `ccnc == 0`, true vertex (raw) in FV, `edep > 75 MeV` (`T_truth_nu.edep` is in GeV; multiply by 1000). Everything else in the pool is background, classed as in `spec_v1.background_class`.
6. Pool (training rows and purity denominator): `br_filled == 1` and reco vertex in FV. Rows outside the pool are kept in the tables with `in_pool = 0`.
7. Efficiency denominator: all `T_truth_nu` rows with |pdg| == 12, CC, true vertex in FV, `edep > 75 MeV`, counted per (sample, file_id, run, subrun, event, nu_idx), independent of any candidate. Candidates with the -15 sentinel stay in the denominator (they are neutrinos without a usable candidate).
8. `label_unreadable`: if ANY `T_match` branch that section 3 reads cannot be read by uproot (on Aurora the five partner lists fail, on gpvm's fg1 NC files the other 29 branches fail), every candidate of that file is written with `label_unreadable = 1` and null label and match-quality columns, and the truth table of that file gets null `nu_ct_*` columns. They are scored at inference and left out of efficiency and purity. The count is reported per sample. Measured on aurora-r2: 43 NC, 10 nu_e CC, 0 BNB files. This is distinct from a missing or empty `T_match` tree on data (3.1).

9. Placeholder rows: `T_tagger` holds empty rows that are not candidates (`kine_reco_Enu == 0`, `neutrino_type == 0`, vertex at the origin; Matteo's `wcreco.py`). The builder flags them `is_placeholder = 1` (`spec_v1.placeholder_mask`); they are dropped before any count of "candidate exists" and never enter the pool. Found by gpvm: 23 of 1,540 rows in a BNB subsample, 3 of 1,300 in nu_e CC.
10. The `T_match` branches the label reads are `spec_v1.MATCH_BRANCHES`; the truth -> reco score of the evaluation loop is the point IoU `nu_ct_iou_n` (Matteo's `nu_ct_iou`), and the reco -> truth score is `iou_n` (his `ct_iou`); the charge IoU is stored, not used for matching.

11. `truth_pf_unreadable`: when uproot cannot read `T_truth_pf` (its `E`/`KE` baskets, the same limit as 3.8; 45 of 11,888 aurora-r2 BNB files), the file is still built in full; only `true_lep_pdg` / `true_lep_ke_mev` are null and the truth rows and the files row carry `truth_pf_unreadable = 1`. Such neutrinos are left out of the e500 signal count only.

## 4. Fiducial volume

The collaboration's nu_e FV (the Pandora/SPINE nueCC FV, SBND technote eq. 6, Matteo's `cuts.FV`; user decision
2026-11-11): x in [-190, 190], y in [-190, 190], z in [10, 450] cm, minus the cathode band |x| <= 5 cm, minus
the cut-out x < 0 and y > 100 and z > 250 (top downstream corner of the East TPC). One function, `spec_v1.in_fv`, for
the true vertex (signal), the reco vertex (pre-cut) and the evaluation, identical for data and MC. FV flags are computed
when a table is read from stored coordinates (`reco_x/y/z`, `m_vx/vy/vz` raw, `m_sce_x/y/z`), never baked in.
`in_wcgpu1_box` (the small-sample study's box, no cut-out) and `in_toolkit_box` are reporting columns only; round 0
reproduces the wcgpu numbers with the wcgpu1 box and is then re-reported in the collaboration FV.

## 5. Tables

Per sample, parquet, one file per table, keyed as in section 2.

- `cands_<sample>.parquet`: one row per candidate. Columns: key; `wct_version`; `reco_e_mev` (the largest `kine_energy_particle` among the candidate's `T_kine` particles with `kine_particle_pool == 0` and |`kine_particle_type`| == 11, null if none; the e500 selection); `sce_fallback`; `br_filled`, `nue_score`, `numu_score` (stored shipped scores); the 258 final-model inputs under their `T_tagger` names (257) plus `kine_reco_Enu` (from `wc.T_kine`); every other scalar `wc.T_tagger` branch (1,246 in all, so later rounds need no rebuild); `reco_x/y/z`; label and match-quality columns of section 3; `partner_kind`, `m_nu_idx`, `m_pdg`, `m_ccnc`, `m_E`, `m_edep_mev`, `m_vx/vy/vz`, `m_sce_x/y/z`, `dist_sce_cm`, `dist_raw_cm`, `in_pool`, `is_signal`, `bkg_class`, `label_unreadable`, `split`; DL-vertex columns from `wc.T_dlvtx_call` for this candidate: `dl_x/y/z`, `final_x/y/z`, `final_valid`, `rerank_x/y/z`, `rerank_valid`, `trad_x/y/z`, `trad_valid`, `hint_valid`, `accepted`, `pass`, `status`, `n_points`, `n_space`, `two_end_veto`.
- `objs_<sample>.parquet`: one row per object of every jagged `wc.T_tagger` list (205 list branches), columns key + `list_name` + `index` + value, so every pooling idea (summary statistics, multiple-instance learning) reads this file.
- `truth_<sample>.parquet`: one row per `T_truth_nu` interaction, none filtered out (the efficiency denominator and the spill removal of the evaluation standard need all of them): key without `cand`, `nu_idx`, `pdg`, `ccnc`, `mode`, `E`, `edep_mev`, `vtx_x/y/z`, `sce_x/y/z`, `is_matchable` (edep > 0 and vertex in the active volume x +-201.3, y +-203.73, z 0-501 cm, bounds included), `is_true_signal`, the true lepton (`true_lep_pdg`, `true_lep_ke_mev`: among the `T_truth_pf` rows of this neutrino with `parent_trackid == 0` and |pdg| 11 or 13, the largest KE), and the truth -> reco match for the efficiency loop: `nu_ct_cand_index` (the `nu_index` of the neutrino's best reco object from its level-2 side-1 row, -1 when that object is not a candidate or there is no partner), `nu_ct_iou_n`, `nu_ct_iou_q`, `nu_ct_mutual`.
- `files_<sample>.parquet`: one row per file, written even when the file has no candidate and no truth row: `file_id` (the merged-CAF file name, unique within every sample; the builder's default since 2026-11-11, after the parent-folder rule gave 'merged' for every aurora-r2 file), `path` (relative to the sample root given by `--root`, otherwise as given; the comparison of section 8 skips it), `events`, `wc_events`, `candidates`, `label_unreadable` (0/1 for the file), `partner_missing` (count), `holdout`, `r8`, `wct_version`, and the exposure as the evaluation standard needs it: `pot_hdr` (sum of `rec.hdr.pot` over events with `rec.hdr.first_in_subrun == 1`), `pot_total_hist` (the `TotalPOT` histogram, MC cross-check only, never used for data), `n_bnb_spills` (sum of `rec.hdr.nbnbinfo` on first-in-subrun events), `n_offbeam_gates` (sum of `rec.hdr.noffbeambnb`, same rule), `ngenevt` (sum of `rec.hdr.ngenevt`, same rule).

Reading: uproot with `library='np'`, one process per file, 100 processes per Aurora node. Every number quoted from a table names the script that made it.

## 6. Split

MC, per sample: even `subrun` = train, odd = test; every candidate of an event stays on one side. Beam-off training half: all files have subrun 1, so inside it `run % 4 == 1` is the test quarter and the rest is train (`spec_v1.split`). The evaluation half is never trained on. Round 0 uses the aurora-r2 subset of the same size as the wcgpu study, chosen by file order in the lists: the first 2,020 files of `nuecc` and the first 250 of `bnb` (about 8,900 and 13,100 events), the two samples combined and unweighted, and NOTHING else: no beam-off, no NC Delta-radiative (wcgpu's study had only CV and nu_e CC). Beam-off and NC enter from round 1 on, so their effect is measured on its own.

## 7. Scoring and model export

- Inference is post-processing of the merged CAF. Model: xgboost, trained with `base_score=0.5` fixed, exported as JSON; readers also patch `base_score` from the JSON. Score written as the raw margin (`output_margin=True`); the shipped `nue_score` is kept next to it in every score table.
- Sentinel: `nue_score = -15` when `br_filled == 0`; the model is not evaluated and the new score is also -15. Ties at -15 collapse into one point on every curve.
- Operating point: matched efficiency (50 % of the true signal, section 3.7), never a fixed cut carried between models. Every comparison is per true-energy bin at matched purity, paired per true neutrino against the baseline.
- Round-0 training recipe (wcgpu small-sample study), used exactly: inputs = the 258 names in the shipped order as float32, the 15 small-model scores as STORED in the file; no input protections, NaN left as xgboost missing; `XGBClassifier(max_depth=30, n_estimators=600, learning_rate=0.1, objective='binary:logitraw', subsample=0.8, colsample_bytree=1, colsample_bylevel=0.3, colsample_bynode=0.3, gamma=0.1, random_state=5)` plus `base_score=0.5` (the one intended difference: wcgpu let xgboost fit it, 0.7355; a constant margin offset); no class or event weights; fit on the train-half pool rows with `label_unreadable == 0`, label `is_signal`.
- Acceptance test for round 0 (wcgpu small-sample study), gated on the gains, not on absolute values, since r7 and aurora-r2 are different reconstructions. Definitions as in wcgpu's `stage2/evalnue.py`: AUC over the test-half pool rows of both samples together; efficiency = selected signal over the section-3.7 denominator; purity = selected `is_signal` over selected pool rows at the tightest cut reaching 50 % efficiency. wcgpu's reference with its bootstrap errors (500 resamples by event): AUC 0.9160 +- 0.0055 shipped, 0.9479 +- 0.0046 retrained, gain +0.0319 +- 0.0048; purity at 50 % 95.38 +- 0.83 shipped, 96.91 +- 0.49 retrained, gain +1.53 +- 0.69 points; ceiling 60.5 +- 0.8 %. Pass rule: AUC gain within 0.032 +- 0.014 (2 sigma combined), ceiling within 60.5 +- 2.4 %, purity gain positive (a sign check only). If the AUC gain or the ceiling fails, the label is the first suspect: rebuild the pooled-cosmic rule from `T_overlap` before changing anything else.

## 8. The shared-file check

Before step 4, Aurora publishes the table code and both servers run it on the same one file (an aurora-r2 nu_e CC unit shipped to gpvm, 5 events) and compare every column value by value (`compare_tables.py`, max absolute difference 0 for integers and flags, 1e-6 for floats). Only then do the full jobs run.

Cosmic-class rates: once beam-off is in the training (round 1 on) the cosmic class grows from data while MC has almost none (0.0004 per event), so compare cosmic-class rates between rounds per sample only (wcgpu).

## 9. What is not in v1

POT and flux weights (training stays unweighted apart from the sample mix; purities for the beam need per-species weights, step 9); the nu_mu selection; any new input family (round 2+, see the extra-information discussion).
