"""Score spec_v1 tables with a saved xgboost model and report the development metrics per sample, for the stored
shipped score and for the model: pool and signal counts, AUC, purity at 50 % and 40 % efficiency, ceiling, cut-7 point
of the shipped score. Same definitions as training/round1/train_round1.py, so numbers are directly comparable.

    python score_model.py TABLE_DIR MODEL.json --samples s1 s2 ... [--split test|train|dev|all] [--json OUT] [--scores OUT.parquet]

--split: which spec_v1 split to evaluate ('test' = odd subruns under v1; 'all' = every row, for an independent production
such as fg1; 'dev' = the v1.1 development slice). The efficiency denominator is the true-signal count of the same split.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'spec'); sys.path.insert(0, SPEC); import spec_v1 as S
NAMES = [l.strip() for l in open(os.path.join(SPEC, 'nue_l2_inputs_258.txt')) if l.strip()]
COLS = ['sample', 'file_id', 'run', 'subrun', 'event', 'cand', 'br_filled', 'nue_score', 'reco_x', 'reco_y', 'reco_z', 'partner_kind', 'dist_sce_cm',
        'm_pdg', 'm_ccnc', 'm_vx', 'm_vy', 'm_vz', 'm_edep_mev', 'label_unreadable', 'is_placeholder'] + NAMES

def which_split(sample, run, subrun, scheme):
    return S.split_v11(sample, run, subrun) if scheme in ('dev',) or scheme == 'train_v11' else S.split(sample, run, subrun)

def load(tdir, samples, split):
    C, T = [], []
    for s in samples:
        c = pd.read_parquet(os.path.join(tdir, f'cands_{s}.parquet'), columns=COLS); c = c[c.is_placeholder == 0]
        c = c[S.pool_mask(c.br_filled, c.reco_x, c.reco_y, c.reco_z) & (c.label_unreadable == 0)].copy()
        c['is_signal'] = S.signal_mask(c.partner_kind, c.label_unreadable, c.dist_sce_cm, c.m_pdg, c.m_ccnc, c.m_vx, c.m_vy, c.m_vz, c.m_edep_mev)
        t = pd.read_parquet(os.path.join(tdir, f'truth_{s}.parquet'))
        if len(t): t = t[S.true_signal_mask(t.pdg, t.ccnc, t.vx, t.vy, t.vz, t.edep_mev) & (t.label_unreadable == 0)].copy()
        if split != 'all':
            c = c[which_split(c['sample'], c.run, c.subrun, split) == split]
            if len(t): t = t[which_split(t['sample'], t.run, t.subrun, split) == split]
        C.append(c); T.append(t)
    return pd.concat(C, ignore_index=True), pd.concat(T, ignore_index=True)

def metrics(y, score, n_true):
    out = dict(pool=int(len(y)), signal=int(y.sum()), true_signal=int(n_true))
    if n_true == 0 or y.sum() == 0 or y.sum() == len(y): return out
    o = np.argsort(-score); cs = np.cumsum(y[o]); n = np.arange(1, len(o) + 1); out['auc'] = float(roc_auc_score(y, score))
    for eff in (0.5, 0.4):
        k = min(int(np.searchsorted(cs, eff * n_true)), len(o) - 1); out[f'purity_at_{int(eff*100)}'] = float(cs[k] / n[k]); out[f'cut_at_{int(eff*100)}'] = float(score[o][k])
    out['ceiling'] = float(y.sum() / n_true); return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0]); ap.add_argument('table_dir'); ap.add_argument('model')
    ap.add_argument('--samples', nargs='+', required=True); ap.add_argument('--split', default='test'); ap.add_argument('--json'); ap.add_argument('--scores')
    a = ap.parse_args(); t0 = time.time()
    C, T = load(a.table_dir, a.samples, a.split)
    m = xgb.XGBClassifier(); m.load_model(a.model)
    C['model_score'] = m.predict(C[NAMES].to_numpy(np.float32), output_margin=True)
    res = dict(model=os.path.basename(a.model), split=a.split, samples=a.samples, xgboost=xgb.__version__, per_sample={}, all={})
    groups = [('all', C, T)] + [(s, C[C['sample'] == s], T[T['sample'] == s] if len(T) else T) for s in a.samples]
    for name, c, t in groups:
        y = c.is_signal.to_numpy(int); n_true = len(t)
        r = dict(shipped=metrics(y, c.nue_score.to_numpy(), n_true), model=metrics(y, c.model_score.to_numpy(), n_true))
        sel7 = c.nue_score > 7
        if n_true: r['shipped']['cut7_efficiency'] = float(c.is_signal[sel7].sum() / n_true); r['shipped']['cut7_purity'] = float(c.is_signal[sel7].mean()) if sel7.any() else None
        (res['all'] if name == 'all' else res['per_sample']).update({name: r} if name != 'all' else r)
    print(f'{time.time()-t0:.0f}s'); print(json.dumps(res, indent=1, default=float))
    if a.json: json.dump(res, open(a.json, 'w'), indent=1, default=float)
    if a.scores: C[['sample', 'file_id', 'run', 'subrun', 'event', 'cand', 'is_signal', 'nue_score', 'model_score']].to_parquet(a.scores, index=False)
