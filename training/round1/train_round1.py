"""Round 1 (spec_v1 section 7): the shipped 258-input structure retrained on ALL training samples at full statistics.
Collaboration FV, spec_v1 label and pool, split by spec_v1.split, unweighted. Learning curve over fractions of the training
half (by event), the full fit with several seeds, metrics on the test half: AUC, purity at 50 % / 40 % efficiency,
efficiency and purity at the cut of 7 for the shipped score, ceiling. Usage: python train_round1.py <tables_dir> <out_dir>"""
import sys, os, json, time
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'spec'); sys.path.insert(0, SPEC); import spec_v1 as S
NAMES = [l.strip() for l in open(os.path.join(SPEC, 'nue_l2_inputs_258.txt')) if l.strip()]
tdir, outd = sys.argv[1], sys.argv[2]; os.makedirs(outd, exist_ok=True)
SAMPLES = ['aurora-r2-nuecc', 'aurora-r2-bnb', 'aurora-r2-ncdr', 'fg1-beam-off-train']
COLS = ['sample', 'file_id', 'run', 'subrun', 'event', 'cand', 'br_filled', 'nue_score', 'reco_x', 'reco_y', 'reco_z', 'partner_kind', 'dist_sce_cm',
        'm_pdg', 'm_ccnc', 'm_vx', 'm_vy', 'm_vz', 'm_edep_mev', 'label_unreadable', 'is_placeholder'] + NAMES
t0 = time.time(); C = []; T = []
for s in SAMPLES:
    c = pd.read_parquet(os.path.join(tdir, f'cands_{s}.parquet'), columns=COLS); c = c[c.is_placeholder == 0]
    c = c[S.pool_mask(c.br_filled, c.reco_x, c.reco_y, c.reco_z) & (c.label_unreadable == 0)].copy()
    c['is_signal'] = S.signal_mask(c.partner_kind, c.label_unreadable, c.dist_sce_cm, c.m_pdg, c.m_ccnc, c.m_vx, c.m_vy, c.m_vz, c.m_edep_mev)
    c['split'] = S.split(c['sample'], c.run, c.subrun); C.append(c)
    t = pd.read_parquet(os.path.join(tdir, f'truth_{s}.parquet'))
    if len(t):
        t = t[S.true_signal_mask(t.pdg, t.ccnc, t.vx, t.vy, t.vz, t.edep_mev) & (t.label_unreadable == 0)].copy(); t['split'] = S.split(t['sample'], t.run, t.subrun); T.append(t)
C = pd.concat(C, ignore_index=True); T = pd.concat(T, ignore_index=True)
tr, te = C[C.split == 'train'], C[C.split == 'test']; n_true = int((T.split == 'test').sum())
print(f'load {time.time()-t0:.0f}s: pool train {len(tr)} ({int(tr.is_signal.sum())} signal) test {len(te)} ({int(te.is_signal.sum())} signal), true signal test {n_true}', flush=True)
Xte, yte = te[NAMES].to_numpy(np.float32), te.is_signal.to_numpy(int)
def metrics(score, n_true_sig):
    auc = roc_auc_score(yte, score); o = np.argsort(-score); cs = np.cumsum(yte[o]); n = np.arange(1, len(o) + 1); out = dict(auc=float(auc))
    for eff in (0.5, 0.4):
        k = min(int(np.searchsorted(cs, eff * n_true_sig)), len(o) - 1); out[f'purity_at_{int(eff*100)}'] = float(cs[k] / n[k]); out[f'cut_at_{int(eff*100)}'] = float(score[o][k])
    out['ceiling'] = float(yte.sum() / n_true_sig); return out
PARAMS = dict(max_depth=30, n_estimators=600, learning_rate=0.1, objective='binary:logitraw', subsample=0.8, colsample_bytree=1, colsample_bylevel=0.3, colsample_bynode=0.3, gamma=0.1, base_score=0.5, n_jobs=96, tree_method='hist')
res = dict(pool=dict(train=len(tr), test=len(te)), signal=dict(train=int(tr.is_signal.sum()), test=int(te.is_signal.sum())), true_signal_test=n_true,
           per_sample_train=tr.groupby('sample').is_signal.agg(['size', 'sum']).rename(columns={'size': 'pool', 'sum': 'signal'}).astype(int).to_dict('index'),
           shipped=metrics(te.nue_score.to_numpy(), n_true), learning_curve=[], full=[])
sel7 = te.nue_score > 7; res['shipped']['cut7_efficiency'] = float(te.is_signal[sel7].sum() / n_true); res['shipped']['cut7_purity'] = float(te.is_signal[sel7].mean())
ev = tr[['sample', 'file_id', 'run', 'subrun', 'event']].drop_duplicates().reset_index(drop=True); rng = np.random.default_rng(5); ev['r'] = rng.random(len(ev))
trk = tr.merge(ev, on=['sample', 'file_id', 'run', 'subrun', 'event'])
for frac in (0.01, 0.03, 0.1, 0.3, 1.0):
    sub = trk[trk.r < frac]; t1 = time.time(); m = xgb.XGBClassifier(**PARAMS, random_state=5); m.fit(sub[NAMES].to_numpy(np.float32), sub.is_signal.to_numpy(int))
    r = metrics(m.predict(Xte, output_margin=True), n_true); r.update(fraction=frac, rows=len(sub), signal=int(sub.is_signal.sum()), fit_s=round(time.time() - t1, 1)); res['learning_curve'].append(r); print('curve', r, flush=True)
    if frac == 1.0: m.save_model(os.path.join(outd, 'model_round1_seed5.json')); res['full'].append(dict(seed=5, **r))
for seed in (6, 7):
    t1 = time.time(); m = xgb.XGBClassifier(**PARAMS, random_state=seed); m.fit(tr[NAMES].to_numpy(np.float32), tr.is_signal.to_numpy(int))
    r = metrics(m.predict(Xte, output_margin=True), n_true); r.update(seed=seed, fit_s=round(time.time() - t1, 1)); res['full'].append(r); print('seed', r, flush=True)
    m.save_model(os.path.join(outd, f'model_round1_seed{seed}.json'))
json.dump(res, open(os.path.join(outd, 'round1.json'), 'w'), indent=1, default=float); print(json.dumps({k: v for k, v in res.items() if k != 'learning_curve'}, indent=1, default=float))
