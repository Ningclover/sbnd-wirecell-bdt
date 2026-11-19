"""Round 2, stage 2: retrain the final (level-2) nu_e model on the v1.2 tables, with the 15 level-1 scores replaced by the
retrained ones (variant "two-level") and, for comparison, with the shipped level-1 scores (variant "level2-only"), both with
the decided weights and early stopping.

Rows: pool candidates (spec_v1.pool_mask), label spec_v1.signal_mask, split spec_v1.split_v11 (train / dev). Inputs: the 258
of spec/nue_l2_inputs_258.txt; in the two-level variant the 15 <model>_score columns are replaced by l1_<model> from the
stage-1 scores file (NaN where the candidate has no object or fails the model's pre-cut; xgboost treats NaN as missing).

Weights (user decision 2026-11-19): each row carries its sample's exposure weight at the BNB POT (--weights); then the signal
rows are rescaled so that signal holds the fraction --floor of the total weight (the floor on the nu_e signal share;
scanned). Dev rows are weighted the same way for the early-stopping loss.

Learner: xgboost hist, binary:logistic (the same trees as the shipped binary:logitraw; the score reported and saved is the raw
margin, output_margin=True, so cuts live on the shipped scale), learning rate --lr, depths in --depth, early stopping on the weighted dev
log-loss (--patience, --max-trees); round 1's column subsampling kept (colsample_bylevel / bynode 0.3, subsample 0.8).

Reported per fit, on the dev slice: unweighted AUC overall and per sample; the exposure-weighted Aurora proxy of the target
metric: the shipped score at cut 7 gives a weighted signal efficiency e0 and purity p0 on the dev pool; each model's cut is
set where its weighted efficiency equals e0, and its weighted purity there is reported (plus the per-sample composition of
what passes). Written: <out>/l2_<variant>_d<depth>_f<floor>.json (xgboost), <out>/train_l2.json, <out>/dev_scores.parquet.

  python training/round2/train_l2.py MERGED_DIR L1_SCORES OUT_DIR --samples ... --weights S=w ... [--variants two-level level2-only]
         [--depth 10 30] [--floor 0.333 0.5] [--lr 0.1] [--max-trees 2000] [--patience 50] [--nthread 96] [--seed 5]
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); SPEC = os.path.join(HERE, '..', '..', 'spec'); sys.path.insert(0, SPEC)
import spec_v1
from l1_models import NUE_L1, L1_MODELS
KEY = ['file_id', 'run', 'subrun', 'event', 'cand']
NAMES = [l.strip() for l in open(os.path.join(SPEC, 'nue_l2_inputs_258.txt')) if l.strip()]
SHIPPED_SCORES = [L1_MODELS[m]['score'] for m in NUE_L1]

def load(d, samples, W):
    need = KEY + ['br_filled', 'reco_x', 'reco_y', 'reco_z', 'partner_kind', 'label_unreadable', 'dist_sce_cm', 'm_pdg', 'm_ccnc', 'm_vx', 'm_vy', 'm_vz', 'm_edep_mev', 'is_placeholder', 'nue_score'] + NAMES
    out = []
    for s in samples:
        C = pd.read_parquet(os.path.join(d, 'cands_%s.parquet' % s), columns=list(dict.fromkeys(need)))
        C = C[(C.is_placeholder == 0) & (C.label_unreadable == 0)]
        C = C[spec_v1.pool_mask(C.br_filled, C.reco_x, C.reco_y, C.reco_z)].copy()
        C['y'] = spec_v1.signal_mask(C.partner_kind, C.label_unreadable, C.dist_sce_cm, C.m_pdg, C.m_ccnc, C.m_vx, C.m_vy, C.m_vz, C.m_edep_mev).astype(int)
        C['split'] = [str(x) for x in spec_v1.split_v11(np.full(len(C), s, dtype=object), C.run.to_numpy(), C.subrun.to_numpy())]
        C['w_exp'] = W[s]; C['sample'] = s; out.append(C)
    return pd.concat(out, ignore_index=True)

def floored(w, y, floor):
    w = w.astype(float).copy(); s = y == 1; tot = w.sum()
    w[s] *= floor * tot / w[s].sum(); w[~s] *= (1 - floor) * tot / w[~s].sum(); return w

def weighted_eff_pur(score, y, w, cut):
    sel = score > cut; ws = w * (y == 1)
    eff = ws[sel].sum() / ws.sum(); pur = ws[sel].sum() / max(w[sel].sum(), 1e-12); return float(eff), float(pur), sel

def cut_at_eff(score, y, w, e0):
    s = score[y == 1]; ws = w[y == 1]; o = np.argsort(-s); c = np.cumsum(ws[o]) / ws.sum()
    k = int(np.searchsorted(c, e0)); return float(s[o][min(k, len(s) - 1)])

def dev_report(score, D, e0, cut=None):
    y, w = D.y.to_numpy(), D.w_exp.to_numpy()
    r = dict(auc=float(roc_auc_score(y, score)), auc_by_sample={s: (float(roc_auc_score(D.y[D['sample'] == s], score[(D['sample'] == s).to_numpy()])) if D.y[D['sample'] == s].nunique() == 2 else None) for s in D['sample'].unique()})
    c = cut if cut is not None else cut_at_eff(score, y, w, e0); eff, pur, sel = weighted_eff_pur(score, y, w, c)
    r.update(cut=c, weighted_eff=eff, weighted_purity=pur, passing_weighted_by_sample={s: float(w[sel & (D['sample'] == s).to_numpy()].sum()) for s in D['sample'].unique()},
             passing_signal_weighted=float(w[sel & (y == 1)].sum()), passing_background_weighted=float(w[sel & (y == 0)].sum()))
    return r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('merged_dir'); ap.add_argument('l1_scores'); ap.add_argument('out_dir'); ap.add_argument('--samples', nargs='+', required=True); ap.add_argument('--weights', nargs='*', default=[])
    ap.add_argument('--variants', nargs='*', default=['two-level', 'level2-only']); ap.add_argument('--depth', nargs='*', type=int, default=[10, 30]); ap.add_argument('--floor', nargs='*', type=float, default=[0.333, 0.5])
    ap.add_argument('--lr', type=float, default=0.1); ap.add_argument('--max-trees', type=int, default=2000); ap.add_argument('--patience', type=int, default=50); ap.add_argument('--nthread', type=int, default=96); ap.add_argument('--seed', type=int, default=5)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    W = {s: 1.0 for s in a.samples}
    for kv in a.weights: k, v = kv.split('='); W[k] = float(v)
    t0 = time.time(); C = load(a.merged_dir, a.samples, W)
    L1 = pd.read_parquet(a.l1_scores); C = C.merge(L1[KEY + ['l1_' + m for m in NUE_L1]], on=KEY, how='left')
    tr, dv = C[C.split == 'train'], C[C.split == 'dev']
    print('pool train %d (%d signal) dev %d (%d signal); l1 scores present for %.1f %% of pool rows (any model); %.0f s' % (len(tr), tr.y.sum(), len(dv), dv.y.sum(), 100 * C[['l1_' + m for m in NUE_L1]].notna().any(axis=1).mean(), time.time() - t0), flush=True)
    e0, p0, _ = weighted_eff_pur(dv.nue_score.to_numpy(), dv.y.to_numpy(), dv.w_exp.to_numpy(), 7.0)
    res = dict(weights=W, lr=a.lr, max_trees=a.max_trees, patience=a.patience, seed=a.seed, xgboost=xgb.__version__, rows=dict(train=dict(n=int(len(tr)), s=int(tr.y.sum())), dev=dict(n=int(len(dv)), s=int(dv.y.sum()))),
               shipped_dev=dev_report(dv.nue_score.to_numpy(), dv, e0, cut=7.0), fits={})
    print('shipped nue_score at cut 7 on dev: weighted efficiency %.3f purity %.3f; AUC %.4f' % (e0, p0, res['shipped_dev']['auc']), flush=True)
    dev_scores = dv[KEY + ['sample', 'split', 'y', 'w_exp', 'nue_score']].copy()
    for var in a.variants:
        names = list(NAMES)
        if var == 'two-level':
            names = [('l1_' + NUE_L1[SHIPPED_SCORES.index(n)] if n in SHIPPED_SCORES else n) for n in NAMES]
        Xtr, Xdv = tr[names].to_numpy(np.float32), dv[names].to_numpy(np.float32)
        for depth in a.depth:
            for floor in a.floor:
                tag = '%s_d%d_f%g' % (var, depth, floor); t1 = time.time()
                wtr, wdv = floored(tr.w_exp.to_numpy(), tr.y.to_numpy(), floor), floored(dv.w_exp.to_numpy(), dv.y.to_numpy(), floor)
                dtr = xgb.DMatrix(Xtr, label=tr.y.to_numpy(), weight=wtr, feature_names=names, missing=np.nan); ddv = xgb.DMatrix(Xdv, label=dv.y.to_numpy(), weight=wdv, feature_names=names, missing=np.nan)
                params = dict(max_depth=depth, eta=a.lr, objective='binary:logistic', eval_metric='logloss', tree_method='hist', nthread=a.nthread, seed=a.seed, subsample=0.8, colsample_bylevel=0.3, colsample_bynode=0.3, gamma=0.1, base_score=0.5)
                ev = {}; bst = xgb.train(params, dtr, num_boost_round=a.max_trees, evals=[(dtr, 'train'), (ddv, 'dev')], early_stopping_rounds=a.patience, evals_result=ev, verbose_eval=False)
                nt = bst.best_iteration + 1; sc = bst.predict(ddv, iteration_range=(0, nt), output_margin=True)
                f = dict(variant=var, depth=depth, floor=floor, trees=int(nt), train_logloss=float(ev['train']['logloss'][nt - 1]), dev_logloss=float(ev['dev']['logloss'][nt - 1]), fit_s=round(time.time() - t1, 1), inputs=names)
                f['dev'] = dev_report(sc, dv, e0); res['fits'][tag] = f; dev_scores[tag] = sc
                bst.save_model(os.path.join(a.out_dir, 'l2_%s.json' % tag))
                print('%-26s trees %5d  logloss train %.4f dev %.4f  AUC %.4f  at eff %.3f: purity %.3f (shipped %.3f)  bnb AUC %s  (%.0f s)' % (
                    tag, nt, f['train_logloss'], f['dev_logloss'], f['dev']['auc'], f['dev']['weighted_eff'], f['dev']['weighted_purity'], p0, '%.4f' % next((v for k, v in f['dev']['auc_by_sample'].items() if 'bnb' in k and v), float('nan')), f['fit_s']), flush=True)
                json.dump(res, open(os.path.join(a.out_dir, 'train_l2.json'), 'w'), indent=1)
    dev_scores.to_parquet(os.path.join(a.out_dir, 'dev_scores.parquet')); res['wall_s'] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out_dir, 'train_l2.json'), 'w'), indent=1); print('done in %.0f s' % res['wall_s'])

if __name__ == '__main__': main()
