"""Per-energy-bin comparison of round 1 against the shipped weights (wcgpu lesson 2: judge per true-energy bin at matched
overall efficiency, paired per true neutrino). Test half, collaboration FV. Edges = Matteo's (MeV). For each model the cut
giving 50 % overall efficiency; then per true-E bin: efficiency (selected signal / true signal in the bin), and paired counts of
true signal neutrinos selected by one model and not the other. Per reco-E bin: purity of the selected pool rows.
Usage: python energy_bins.py <tables_dir> <model.json> <out.json>"""
import sys, os, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, '/lus/flare/projects/neutrinoGPU/xning/bdt-comm/steps/step3_spec'); import spec_v1 as S
HERE = os.path.dirname(os.path.abspath(__file__)); NAMES = [l.strip() for l in open(os.path.join(HERE, 'nue_l2_inputs_258.txt')) if l.strip()]
tdir, model_path, outp = sys.argv[1:4]; EDGES = np.array([0, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 6000.0])
COLS = list(dict.fromkeys(['sample', 'file_id', 'run', 'subrun', 'event', 'cand', 'br_filled', 'nue_score', 'reco_x', 'reco_y', 'reco_z', 'partner_kind', 'dist_sce_cm', 'm_pdg', 'm_ccnc', 'm_vx', 'm_vy', 'm_vz', 'm_edep_mev', 'm_E', 'm_nu_idx', 'kine_reco_Enu', 'label_unreadable', 'is_placeholder'] + NAMES))
C, T = [], []
for s in ['aurora-r2-nuecc', 'aurora-r2-bnb', 'aurora-r2-ncdr', 'fg1-beam-off-train']:
    c = pd.read_parquet(os.path.join(tdir, f'cands_{s}.parquet'), columns=COLS); c = c[c.is_placeholder == 0]
    c = c[S.pool_mask(c.br_filled, c.reco_x, c.reco_y, c.reco_z) & (c.label_unreadable == 0)].copy()
    c['is_signal'] = S.signal_mask(c.partner_kind, c.label_unreadable, c.dist_sce_cm, c.m_pdg, c.m_ccnc, c.m_vx, c.m_vy, c.m_vz, c.m_edep_mev); c['split'] = S.split(c['sample'], c.run, c.subrun); C.append(c[c.split == 'test'])
    t = pd.read_parquet(os.path.join(tdir, f'truth_{s}.parquet'))
    if len(t):
        t = t[S.true_signal_mask(t.pdg, t.ccnc, t.vx, t.vy, t.vz, t.edep_mev) & (t.label_unreadable == 0)].copy(); t['split'] = S.split(t['sample'], t.run, t.subrun); T.append(t[t.split == 'test'])
C = pd.concat(C, ignore_index=True); T = pd.concat(T, ignore_index=True); T['E_mev'] = 1000 * T.E
m = xgb.XGBClassifier(); m.load_model(model_path); C['score_new'] = m.predict(C[NAMES].to_numpy(np.float32), output_margin=True)
n_true = len(T); res = dict(true_signal=n_true, edges=EDGES.tolist(), models={})
key = ['sample', 'file_id', 'run', 'subrun', 'event', 'm_nu_idx']
sel = {}
for name, col in (('shipped', 'nue_score'), ('round1', 'score_new')):
    f = C.sort_values(col, ascending=False); cs = np.cumsum(f.is_signal.to_numpy()); k = int(np.searchsorted(cs, 0.5 * n_true)); cut = float(f[col].iloc[k])
    s_ = C[(C[col] > cut)]; sig = s_[s_.is_signal].drop_duplicates(key); sel[name] = set(map(tuple, sig[key].to_numpy()))
    tsel = T.merge(sig[key].rename(columns={'m_nu_idx': 'nu_idx'}), on=['sample', 'file_id', 'run', 'subrun', 'event', 'nu_idx'], how='left', indicator=True)
    tsel['sel'] = tsel._merge == 'both'; tsel['bin'] = pd.cut(tsel.E_mev, EDGES)
    eff = tsel.groupby('bin', observed=False).sel.agg(['sum', 'size'])
    s_['bin'] = pd.cut(s_.kine_reco_Enu, EDGES); pur = s_.groupby('bin', observed=False).is_signal.agg(['sum', 'size'])
    res['models'][name] = dict(cut=cut, overall_efficiency=float(tsel.sel.mean()), purity=float(s_.is_signal.mean()), selected=int(len(s_)),
                               eff_bins=[dict(lo=float(EDGES[i]), hi=float(EDGES[i + 1]), selected=int(eff['sum'].iloc[i]), true=int(eff['size'].iloc[i]), eff=float(eff['sum'].iloc[i] / eff['size'].iloc[i]) if eff['size'].iloc[i] else None) for i in range(len(EDGES) - 1)],
                               pur_bins=[dict(lo=float(EDGES[i]), hi=float(EDGES[i + 1]), signal=int(pur['sum'].iloc[i]), selected=int(pur['size'].iloc[i]), purity=float(pur['sum'].iloc[i] / pur['size'].iloc[i]) if pur['size'].iloc[i] else None) for i in range(len(EDGES) - 1)])
only_new = sel['round1'] - sel['shipped']; only_old = sel['shipped'] - sel['round1']
def by_bin(keys):
    if not keys: return [0] * (len(EDGES) - 1)
    d = pd.DataFrame(list(keys), columns=key).rename(columns={'m_nu_idx': 'nu_idx'}).merge(T, on=['sample', 'file_id', 'run', 'subrun', 'event', 'nu_idx']); return pd.cut(d.E_mev, EDGES).value_counts(sort=False).tolist()
res['paired'] = dict(gained_by_round1=len(only_new), lost_by_round1=len(only_old), both=len(sel['round1'] & sel['shipped']), gained_by_bin=by_bin(only_new), lost_by_bin=by_bin(only_old))
json.dump(res, open(outp, 'w'), indent=1, default=float)
for name in ('shipped', 'round1'): print(name, 'cut', round(res['models'][name]['cut'], 3), 'eff', round(res['models'][name]['overall_efficiency'], 4), 'purity', round(res['models'][name]['purity'], 4), 'eff by bin', [round(b['eff'], 3) if b['eff'] is not None else None for b in res['models'][name]['eff_bins']])
print('paired: gained', res['paired']['gained_by_round1'], 'lost', res['paired']['lost_by_round1'], 'gained by bin', res['paired']['gained_by_bin'], 'lost by bin', res['paired']['lost_by_bin'])
