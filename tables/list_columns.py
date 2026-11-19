"""Write the catalogue of what the table builder saves: every table, every column, its type, how often it is filled, and one
example value, read from a built table directory (so the list always matches a real build). Family descriptions are the
short texts below; the per-column rows come from the files.

  python tables/list_columns.py TABLE_DIR SAMPLE [--tag TAG] [--out docs/tables_v12.md]

TABLE_DIR/SAMPLE/TAG name the files as build_tables.py wrote them (cands_<sample>_<tag>.parquet, ...).
"""
import argparse, glob, os, re, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'spec'))
from l1_models import L1_MODELS

# (regex on the column name, family name, plain description). First match wins; order matters.
FAMILIES = [
 (r'^(sample|file_id|run|subrun|event|cand|nu_idx|index)$', 'key', 'Which sample, file, run, subrun, event; which candidate (`cand`) or object (`index`) or true neutrino (`nu_idx`) within it.'),
 (r'^(wct_version|reco_e_mev|sce_fallback|reco_[xyz]|is_placeholder|photon_flag|sel_cluster_id)$', 'candidate basics', 'Software version of the file; the electron energy used by the e500 selection; the reconstructed vertex; whether the space-charge fallback was used; the placeholder flag of a neutrino without a usable candidate.'),
 (r'^(partner_kind|m_|dist_|in_pool|is_signal|bkg_class|label_unreadable|split|match_|nu_ct_|is_matchable|is_true_signal|iou_|shared_|mutual$|nmatch$)', 'label and match', 'Truth-to-candidate match: what the candidate\'s true partner is (neutrino, cosmic, missing), that neutrino\'s properties (`m_*`), vertex distances, the signal label, the background class, the train/test split.'),
 (r'^(pdg|ccnc|mode|E|edep_mev|vx|vy|vz|sce_[xyz]|true_lep_pdg|true_lep_ke_mev|truth_pf_unreadable)$', 'true neutrino', 'What the simulated neutrino was (type, charged- or neutral-current, mode, energy, deposited energy), where it interacted (raw and space-charge-shifted vertex), its outgoing lepton.'),
 (r'^(dl_|final_|rerank_|trad_|hint_|accepted$|pass$|status$|n_points$|n_space$|two_end_veto$)', 'vertex finder (v1, this candidate)', 'Outputs of the deep-learning vertex finder for this candidate as stored in v1: the traditional, re-ranked, hint, DL and final vertices with validity flags.'),
 (r'^dlv[01]_|^n_dl_calls$', 'vertex finder (v1.2, first two calls)', 'The same vertex-finder outputs for the first (`dlv0_`) and second (`dlv1_`) call on this candidate, plus the number of calls.'),
 (r'^kine_', 'kinematics (v1.2)', 'Per-candidate energy reconstruction: reconstructed neutrino energy, corrected vertex, added energy, pi0 mass and the two photon kinematics, excluded energies, multiple-scattering energy, cluster and flash ids.'),
 (r'^n_obj_', 'object counts (v1.2)', 'How many objects each level-1 small model has for this candidate; the row count of this candidate in the matching object table.'),
 (r'^(cluster_id|flash_|matched_flash_gid|has_vertex|nu_index|nu_[xyz]|vertex_moved)$|^(cluster_id|flash_|matched_flash_gid|has_vertex|nu_index|nu_[xyz]|vertex_moved)', 'candidate vertex and flash', 'The candidate\'s cluster id, its reconstructed vertex (`nu_x/y/z`), and the optical flash matched to it (time, light yield, TPC, group).'),
 (r'^(nue_score|numu_score|br_filled|neutrino_type|flag_|event_type)', 'shipped decisions', 'The shipped software\'s own outputs: the final nu_e and nu_mu scores, the pre-selection flag `br_filled`, the neutrino-type guess and the overall rule flags.'),
 (r'_score$', 'level-1 pooled scores (shipped)', 'The 18 pooled small-model scores as the shipped software computed them, one number per candidate; the level-2 model reads them.'),
 (r'^(path|events|wc_events|candidates|partner_missing|holdout|r8|pot_|n_bnb_spills|n_offbeam_gates|ngenevt|ismc|error)', 'file bookkeeping', 'Per input file: path, event and candidate counts, exposure (POT, spills, gates, generated events), simulation or data, the error text when the file failed.'),
 (r'^flag$', 'object rule flag', 'The model\'s hand-written rule evaluated on this object (1 = fired), used to label objects at training time.'),
]
TAGGER = [  # the tagger families of the shipped nu_e / nu_mu taggers, by column prefix (Wire-Cell naming)
 ('mip_', 'MIP check: is the main track a minimum-ionising particle'), ('gap_', 'gap between the vertex and the shower start'), ('cme_', 'cosmic-muon energy'), ('anc_', 'angular consistency'),
 ('mgo_', 'multiple gamma overlap'), ('mgt_', 'multiple gamma, tracks'), ('stem_', 'stem of the shower'), ('br1_', 'bad reconstruction 1'), ('br3_', 'bad reconstruction 3'), ('br4_', 'bad reconstruction 4'),
 ('br2_', 'bad reconstruction 2'), ('hol_', 'high-overlap'), ('lem_', 'low-energy michel'), ('vis_', 'vertex inside shower'), ('pio_', 'pi0 reconstruction'), ('stw_', 'single track with shower'),
 ('sig_', 'signal-like single shower'), ('lol_', 'long overlap'), ('tro_', 'track overlap'), ('cosmict_', 'cosmic tagger'), ('numu_cc_', 'nu_mu charged-current tagger'), ('mip_quality_', 'MIP quality'),
 ('shw_sp_', 'shower space points'), ('ssm_', 'single short muon tagger (nu_mu)'), ('ssmsp_', 'single short muon, space points'), ('spt_', 'single photon tagger'), ('cosmic_', 'cosmic counts and energies'), ('act_', 'active-volume containment'), ('kine_', 'kinematics'), ('stem_dir_', 'stem direction'), ('stem_len_', 'stem length'), ('brm_', 'bad reconstruction, misc'),
]

def family(col):
    for rx, name, desc in FAMILIES:
        if re.search(rx, col): return name, desc
    for pre, desc in sorted(TAGGER, key=lambda t: -len(t[0])):
        if col.startswith(pre): return 'tagger inputs: ' + pre.rstrip('_'), 'Shipped tagger family "%s" (%s): its scalar outputs per candidate; every scalar `wc.T_tagger` branch is stored.' % (pre.rstrip('_'), desc)
    return 'other', 'Other scalar `wc.T_tagger` branches.'

def ex(s):
    s = s.dropna()
    if s.empty: return ''
    v = s.iloc[0]
    if isinstance(v, float): return '%.4g' % v
    return str(v)[:40]

def table_section(name, df, per_row, out):
    out.append('\n## `%s`: one row per %s\n' % (name, per_row))
    out.append('%d rows x %d columns in the example build.\n' % df.shape)
    fams = {}
    for c in df.columns:
        f, d = family(c); fams.setdefault(f, (d, []))[1].append(c)
    out.append('| family | columns | what it holds |\n|---|---:|---|')
    for f, (d, cols) in fams.items(): out.append('| %s | %d | %s |' % (f, len(cols), d))
    out.append('\n<details><summary>Every column (name, type, filled fraction, example)</summary>\n')
    for f, (d, cols) in fams.items():
        out.append('\n**%s**\n\n| column | type | filled | example |\n|---|---|---:|---|' % f)
        for c in cols:
            s = df[c]; out.append('| `%s` | %s | %.2f | %s |' % (c, s.dtype, s.notna().mean() if len(s) else 0, ex(s)))
    out.append('\n</details>')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('table_dir'); ap.add_argument('sample'); ap.add_argument('--tag', default=''); ap.add_argument('--out', default='docs/tables_v12.md')
    a = ap.parse_args(); sfx = '_' + a.tag if a.tag else ''
    P = lambda t: os.path.join(a.table_dir, '%s_%s%s.parquet' % (t, a.sample, sfx))
    out = ['# What the table builder saves (spec v1.2)\n',
           'Generated by `tables/list_columns.py` from a build of sample `%s`; every count below is read from the files. Table names carry `_<sample>` and the build tag.\n' % a.sample,
           'The builder (`tables/build_tables.py`) reads the merged CAF files and writes, per sample: a file table, a candidate table, a truth table, and with `--objs` one object table per level-1 small model. A summary `build_<sample>.json` records files, errors, timeouts, candidates and object rows per model.\n']
    table_section('files_<sample>.parquet', pd.read_parquet(P('files')), 'input file (written even when the file failed or has no candidate)', out)
    table_section('cands_<sample>.parquet', pd.read_parquet(P('cands')), 'reconstructed neutrino candidate', out)
    table_section('truth_<sample>.parquet', pd.read_parquet(P('truth')), 'simulated neutrino interaction (empty for data)', out)
    out.append('\n## `objs_<model>_<sample>.parquet`: one row per object of one level-1 small model\n')
    out.append('18 tables, one per model of `spec/l1_models.py`. Columns: the key and `cand`, `index` (the object\'s position in the candidate\'s list), the model\'s list inputs under their shipped names, its scalar input copied to every object (only `pio_2` has one, `pio_mip_id`), and `flag`, the hand-written rule for that object. A model that found no object in a build still gets a table with these columns and no rows.\n')
    out.append('| model | tagger | inputs | rows | objects per candidate with any | candidates with none | rule fired |\n|---|---|---:|---:|---:|---:|---:|')
    C = pd.read_parquet(P('cands'))
    for m, md in L1_MODELS.items():
        f = os.path.join(a.table_dir, 'objs_%s_%s%s.parquet' % (m, a.sample, sfx))
        if not os.path.exists(f): out.append('| %s | %s | %d | (no table) | | | |' % (m, md['flavour'], len(md['lists']) + len(md['scalars']))); continue
        O = pd.read_parquet(f); n = C['n_obj_' + m]
        out.append('| %s | %s | %d | %d | %.1f | %.2f | %.2f |' % (m, md['flavour'], len(md['lists']) + len(md['scalars']), len(O), n[n > 0].mean() if (n > 0).any() else 0, (n == 0).mean(), (O.flag == 1).mean() if len(O) else float('nan')))
    out.append('\n<details><summary>Inputs of every model</summary>\n\n| model | list inputs | scalar | flag | pooled by | empty value |\n|---|---|---|---|---|---|')
    for m, md in L1_MODELS.items(): out.append('| %s | %s | %s | `%s` | %s | %g |' % (m, ', '.join('`%s`' % x for x in md['lists']), ', '.join('`%s`' % x for x in md['scalars']) or '-', md['flag'], md['pool'], md['empty']))
    out.append('\n</details>\n')
    out.append('\n## Special cases the builder records\n')
    out.append('- A file that cannot be opened or read: `error` in the file table starts with `OPEN_FAIL` or `READ_FAIL`; the file contributes nothing; counted in the summary.\n- A file over the time limit (`--timeout`, default 900 s): `error` is `TIMEOUT after ...`; same treatment; `timeouts` in the summary.\n- Truth-match tree unreadable: candidates are written without label columns, `label_unreadable = 1` on the file and its rows.\n- True particle list unreadable: only `true_lep_pdg` / `true_lep_ke_mev` are lost, `truth_pf_unreadable = 1`.\n- Candidate whose true partner is not found: `partner_kind = missing`, counted in `partner_missing`.\n- No space-charge-shifted true vertex: `dist_sce_cm` falls back to the raw distance, `sce_fallback = 1`.\n- Data files: no truth tree, so the truth table is empty, label columns absent, `ismc = 0`.\n- Object lists of unequal length inside one model, or a missing flag branch: the missing entries are NaN. A model whose branches are absent from a file gets no rows from that file.\n- Objects are stored for every candidate, also where the shipped software did not run the model (its `pre_cut` in `l1_models.py`); training applies the pre-cut itself. `pio_2` rows come in pairs, one per photon of each pair, as the shipped software fills them.\n')
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); open(a.out, 'w').write('\n'.join(out) + '\n'); print('wrote', a.out)

if __name__ == '__main__': main()
