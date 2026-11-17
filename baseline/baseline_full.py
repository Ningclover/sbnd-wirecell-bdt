"""Step 5 on Aurora: the shipped nue weights on the full aurora-r2 tables, in the collaboration FV (spec_v1), hold-out none.
Per sample and for the training mix (nuecc + bnb, unweighted; ncdr and beam-off reported separately): pool, signal, background
classes, AUC of nue_score on the pool, efficiency against the true-signal denominator at cuts 0, 4.5, 7, purity at those cuts and
at matched efficiency 50 % / 40 %, ceiling; all on the test half and on the whole. Usage: python baseline_full.py <tables_dir> <out.json>"""
import sys, os, json
import numpy as np, pandas as pd
SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'spec'); sys.path.insert(0, SPEC); import spec_v1 as S
from sklearn.metrics import roc_auc_score
tdir, outp = sys.argv[1], sys.argv[2]
COLS = ['sample', 'file_id', 'run', 'subrun', 'event', 'cand', 'br_filled', 'nue_score', 'reco_x', 'reco_y', 'reco_z', 'partner_kind', 'dist_sce_cm',
        'm_pdg', 'm_ccnc', 'm_vx', 'm_vy', 'm_vz', 'm_edep_mev', 'label_unreadable', 'is_placeholder']
def load(sample):
    C = pd.read_parquet(os.path.join(tdir, f'cands_{sample}.parquet'), columns=COLS); T = pd.read_parquet(os.path.join(tdir, f'truth_{sample}.parquet'))
    C = C[C.is_placeholder == 0].copy()
    C['in_pool'] = S.pool_mask(C.br_filled, C.reco_x, C.reco_y, C.reco_z) & (C.label_unreadable == 0)
    C['is_signal'] = S.signal_mask(C.partner_kind, C.label_unreadable, C.dist_sce_cm, C.m_pdg, C.m_ccnc, C.m_vx, C.m_vy, C.m_vz, C.m_edep_mev)
    C['bkg_class'] = S.background_class(C.partner_kind, C.label_unreadable, C.dist_sce_cm, C.m_pdg, C.m_ccnc)
    C['split'] = S.split(C['sample'], C.run, C.subrun)
    if len(T):
        T['is_true_signal'] = S.true_signal_mask(T.pdg, T.ccnc, T.vx, T.vy, T.vz, T.edep_mev) & (T.label_unreadable == 0)
        T['split'] = S.split(T['sample'], T.run, T.subrun)
    else:
        T = pd.DataFrame(columns=['is_true_signal', 'split'])
    return C, T
def metrics(P, n_true):
    out = dict(pool=int(len(P)), signal=int(P.is_signal.sum()), background=int((~P.is_signal).sum()), true_signal=int(n_true),
               background_classes=P[~P.is_signal].bkg_class.value_counts().to_dict(), ceiling=float(P.is_signal.sum() / n_true) if n_true else np.nan)
    if P.is_signal.nunique() == 2: out['auc'] = float(roc_auc_score(P.is_signal, P.nue_score))
    for cut in (0.0, 4.5, 7.0):
        sel = P[P.nue_score > cut]; out[f'cut_{cut}'] = dict(selected=int(len(sel)), signal=int(sel.is_signal.sum()), efficiency=float(sel.is_signal.sum() / n_true) if n_true else np.nan, purity=float(sel.is_signal.mean()) if len(sel) else np.nan)
    f = P.sort_values('nue_score', ascending=False); cs = np.cumsum(f.is_signal.to_numpy()); n = np.arange(1, len(f) + 1)
    for eff in (0.5, 0.4):
        if n_true and cs[-1] >= eff * n_true:
            k = int(np.searchsorted(cs, eff * n_true)); out[f'purity_at_{int(eff*100)}'] = float(cs[k] / n[k]); out[f'cut_at_{int(eff*100)}'] = float(f.nue_score.iloc[k])
    return out
res = {}
data = {s: load(s) for s in ('aurora-r2-nuecc', 'aurora-r2-bnb', 'aurora-r2-ncdr', 'fg1-beam-off-train') if os.path.exists(os.path.join(tdir, f'cands_{s}.parquet'))}
for s, (C, T) in data.items():
    for half in ('all', 'test'):
        P = C[C.in_pool & ((C.split == 'test') if half == 'test' else True)]; TT = T[T.is_true_signal & ((T.split == 'test') if half == 'test' else True)]
        res[f'{s}/{half}'] = metrics(P, len(TT))
if 'aurora-r2-nuecc' in data and 'aurora-r2-bnb' in data:
    for half in ('all', 'test'):
        P = pd.concat([data[s][0] for s in ('aurora-r2-nuecc', 'aurora-r2-bnb')]); T = pd.concat([data[s][1] for s in ('aurora-r2-nuecc', 'aurora-r2-bnb')])
        P = P[P.in_pool & ((P.split == 'test') if half == 'test' else True)]; TT = T[T.is_true_signal & ((T.split == 'test') if half == 'test' else True)]
        res[f'mix nuecc+bnb/{half}'] = metrics(P, len(TT))
json.dump(res, open(outp, 'w'), indent=1, default=float)
for k, v in res.items(): print(k, 'pool', v['pool'], 'sig', v['signal'], 'bkg', v['background'], 'true', v['true_signal'], 'auc', round(v.get('auc', float('nan')), 4), 'ceiling', round(v['ceiling'], 4), 'pur@50', round(v.get('purity_at_50', float('nan')), 4), 'eff@7', round(v['cut_7.0']['efficiency'], 4), 'pur@7', round(v['cut_7.0']['purity'], 4))
