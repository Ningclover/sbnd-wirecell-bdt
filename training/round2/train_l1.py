"""Round 2, stage 1: retrain the level-1 small per-object models of the nu_e tagger on the v1.2 object tables.

Rule A, MicroBooNE's (user decision 2026-11-19; polarity as wcgpu1 read it off the MicroBooNE trainers): every object of a
signal candidate is a signal example; an object of a background candidate is a background example only where the model's
hand-written flag fired (flag == 0 for the nu_e models); other background objects are left out. Pool candidates only
(spec_v1.pool_mask), the model's pre-cut applied (spec/l1_models.py), rows split by spec_v1.split_v11 (train / dev).

Weights: each row carries its sample's exposure weight at the BNB POT (--weights), then the two classes are rescaled to
equal total weight, as MicroBooNE's trainers did (TMVA NormMode=NumEvents); the dev rows are weighted the same way.

Learner: xgboost, depth 3, hist, binary:logistic; one fit per learning rate in --lr with early stopping on the weighted dev
log-loss (--patience trees without improvement, at most --max-trees). The learning rate kept per model is the one with the
lowest dev log-loss. Reported per model and learning rate: rows and weights per class, trees, train / dev log-loss, object
AUC train / dev, and on dev candidates with at least one object the AUC of the pooled score (min over the candidate's
objects, as the shipped software pools the nu_e models; ALL objects of the candidate, not only the rule-selected ones)
against the candidate label, next to the shipped pooled score's AUC on the same candidates. Written: <out>/l1_<model>_lr<lr>.json (xgboost), <out>/train_l1.json (everything above),
<out>/l1_scores_<label>.parquet (key + one pooled column per model from the kept fit, NaN where the list is empty; the
input of stage 2).

  python training/round2/train_l1.py MERGED_DIR OUT_DIR --samples S1 S2 ... --weights S1=w1 ... [--models m1 m2 ...]
         [--label A|B] [--lr 0.05 0.1 0.3] [--depth 3] [--max-trees 2000] [--patience 50] [--nthread 96] [--seed 5]
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'spec'))
import spec_v1
from l1_models import L1_MODELS, NUE_L1
KEY = ['file_id', 'run', 'subrun', 'event', 'cand']

def load_cands(d, samples, W):
    need = KEY + ['br_filled', 'reco_x', 'reco_y', 'reco_z', 'partner_kind', 'label_unreadable', 'dist_sce_cm', 'm_pdg', 'm_ccnc', 'm_vx', 'm_vy', 'm_vz', 'm_edep_mev',
                  'pio_filled', 'pio_flag_pio', 'cosmict_flag_1'] + [L1_MODELS[m]['score'] for m in L1_MODELS]
    out = []
    for s in samples:
        C = pd.read_parquet(os.path.join(d, 'cands_%s.parquet' % s), columns=need)
        C = C[C.label_unreadable == 0]
        C = C[spec_v1.pool_mask(C.br_filled, C.reco_x, C.reco_y, C.reco_z)].copy()
        C['sig'] = spec_v1.signal_mask(C.partner_kind, C.label_unreadable, C.dist_sce_cm, C.m_pdg, C.m_ccnc, C.m_vx, C.m_vy, C.m_vz, C.m_edep_mev)
        C['split'] = [str(x) for x in spec_v1.split_v11(np.full(len(C), s, dtype=object), C.run.to_numpy(), C.subrun.to_numpy())]
        C['w_exp'] = W[s]; C['sample'] = s; out.append(C)
    return pd.concat(out, ignore_index=True)

def rows_for(C, d, m, samples, label):
    md = L1_MODELS[m]; cols = md['lists'] + md['scalars']
    keep = C[C.eval(md['pre_cut']).to_numpy(bool)] if md['pre_cut'] else C
    keep = keep[KEY + ['sig', 'split', 'w_exp', 'sample', md['score']]]
    O = pd.concat([pd.read_parquet(os.path.join(d, 'objs_%s_%s.parquet' % (m, s)), columns=KEY + cols + ['flag']) for s in samples], ignore_index=True)
    O = O.merge(keep, on=KEY, how='inner')
    sgn = O.sig.to_numpy(bool); ra = md['rule_a']; fired = (O.flag.to_numpy() == ra['fired'])
    if label == 'B': use = np.ones(len(O), bool)
    elif ra['narrows'] == 'signal': use = (~sgn) | fired      # rule narrows the signal side (numu_1, numu_2)
    else: use = sgn | fired                                  # rule narrows the background side (the 15 nu_e models, cosmict_10)
    O['use'] = use; O['y'] = O.sig.astype(int)
    return O, cols, keep

def balanced(w, y):
    w = w.astype(float).copy(); s = y == 1
    tot = w.sum(); w[s] *= 0.5 * tot / w[s].sum(); w[~s] *= 0.5 * tot / w[~s].sum(); return w

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('merged_dir'); ap.add_argument('out_dir'); ap.add_argument('--samples', nargs='+', required=True); ap.add_argument('--weights', nargs='*', default=[])
    ap.add_argument('--models', nargs='*', default=NUE_L1); ap.add_argument('--label', choices=['A', 'B'], default='A'); ap.add_argument('--lr', nargs='*', type=float, default=[0.05, 0.1, 0.3])
    ap.add_argument('--depth', type=int, default=3); ap.add_argument('--max-trees', type=int, default=2000); ap.add_argument('--patience', type=int, default=50); ap.add_argument('--nthread', type=int, default=96); ap.add_argument('--seed', type=int, default=5)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    W = {s: 1.0 for s in a.samples}
    for kv in a.weights: k, v = kv.split('='); W[k] = float(v)
    t0 = time.time(); C = load_cands(a.merged_dir, a.samples, W); print('candidates in the pool: %d (%d signal), %.0f s' % (len(C), C.sig.sum(), time.time() - t0), flush=True)
    res = dict(label=a.label, weights=W, depth=a.depth, lrs=a.lr, max_trees=a.max_trees, patience=a.patience, seed=a.seed, xgboost=xgb.__version__, models={})
    scores = C[KEY + ['split', 'sig']].copy()
    for m in a.models:
        md = L1_MODELS[m]; t1 = time.time(); Oall, cols, keep = rows_for(C, a.merged_dir, m, a.samples, a.label); O = Oall[Oall.use]
        tr, dv = O[O.split == 'train'], O[O.split == 'dev']
        dall = Oall[Oall.split == 'dev']; dmall = xgb.DMatrix(dall[cols].to_numpy(np.float32), feature_names=cols)
        wtr, wdv = balanced(tr.w_exp.to_numpy(), tr.y.to_numpy()), balanced(dv.w_exp.to_numpy(), dv.y.to_numpy())
        r = dict(inputs=cols, rows=dict(train=dict(s=int(tr.y.sum()), b=int((tr.y == 0).sum())), dev=dict(s=int(dv.y.sum()), b=int((dv.y == 0).sum()))),
                 exposure_weight=dict(train=dict(s=float(tr.w_exp[tr.y == 1].sum()), b=float(tr.w_exp[tr.y == 0].sum()))), fits={}, load_s=round(time.time() - t1, 1))
        if min(r['rows']['train']['s'], r['rows']['train']['b']) < 2 or min(r['rows']['dev']['s'], r['rows']['dev']['b']) < 2:
            r['skipped'] = 'too few rows'; res['models'][m] = r; print(m, 'skipped: too few rows', r['rows'], flush=True); continue
        dtr = xgb.DMatrix(tr[cols].to_numpy(np.float32), label=tr.y.to_numpy(), weight=wtr, feature_names=cols)
        ddv = xgb.DMatrix(dv[cols].to_numpy(np.float32), label=dv.y.to_numpy(), weight=wdv, feature_names=cols)
        best = None
        for lr in a.lr:
            params = dict(max_depth=a.depth, eta=lr, objective='binary:logistic', eval_metric='logloss', tree_method='hist', nthread=a.nthread, seed=a.seed)
            ev = {}; t2 = time.time()
            bst = xgb.train(params, dtr, num_boost_round=a.max_trees, evals=[(dtr, 'train'), (ddv, 'dev')], early_stopping_rounds=a.patience, evals_result=ev, verbose_eval=False)
            nt = bst.best_iteration + 1
            ptr = bst.predict(dtr, iteration_range=(0, nt)); pdv = bst.predict(ddv, iteration_range=(0, nt))
            f = dict(trees=int(nt), train_logloss=float(ev['train']['logloss'][nt - 1]), dev_logloss=float(ev['dev']['logloss'][nt - 1]),
                     object_auc_train=float(roc_auc_score(tr.y, ptr, sample_weight=wtr)), object_auc_dev=float(roc_auc_score(dv.y, pdv, sample_weight=wdv)), fit_s=round(time.time() - t2, 1))
            # pooled per candidate on dev (min over objects, like the shipped nu_e pooling), against the candidate label; shipped pooled score on the same candidates
            dvp = dall[KEY + ['sig', md['score'], 'w_exp']].copy(); dvp['p'] = bst.predict(dmall, iteration_range=(0, nt), output_margin=True)
            g = dvp.groupby(KEY, sort=False).agg(sig=('sig', 'first'), new=('p', 'min'), shipped=(md['score'], 'first'), w=('w_exp', 'first')).reset_index()
            if g.sig.nunique() == 2:
                f['pooled_auc_dev_new'] = float(roc_auc_score(g.sig, g.new, sample_weight=g.w)); f['pooled_auc_dev_shipped'] = float(roc_auc_score(g.sig, g.shipped, sample_weight=g.w))
                f['pooled_candidates_dev'] = int(len(g)); f['pooled_signal_dev'] = int(g.sig.sum())
            r['fits']['%g' % lr] = f; bst.save_model(os.path.join(a.out_dir, 'l1_%s_lr%g.json' % (m, lr)))
            print('%-8s lr %-5g trees %5d  logloss train %.4f dev %.4f  objAUC train %.4f dev %.4f  pooled dev AUC new %s shipped %s  (%.0f s)' % (
                m, lr, nt, f['train_logloss'], f['dev_logloss'], f['object_auc_train'], f['object_auc_dev'], '%.4f' % f['pooled_auc_dev_new'] if 'pooled_auc_dev_new' in f else '-', '%.4f' % f['pooled_auc_dev_shipped'] if 'pooled_auc_dev_shipped' in f else '-', f['fit_s']), flush=True)
            if best is None or f['dev_logloss'] < best[1]: best = (lr, f['dev_logloss'], bst, nt)
        r['kept_lr'] = best[0]; res['models'][m] = r
        # pooled score of the kept fit for every pool candidate that passes the pre-cut (train and dev), NaN where the list is empty
        allo = Oall[KEY].copy(); allo['p'] = best[2].predict(xgb.DMatrix(Oall[cols].to_numpy(np.float32), feature_names=cols), iteration_range=(0, best[3]), output_margin=True)
        pooled = allo.groupby(KEY, sort=False).p.min().rename('l1_' + m).reset_index()
        scores = scores.merge(pooled, on=KEY, how='left')
        json.dump(res, open(os.path.join(a.out_dir, 'train_l1.json'), 'w'), indent=1)
    scores.to_parquet(os.path.join(a.out_dir, 'l1_scores_%s.parquet' % a.label)); res['wall_s'] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out_dir, 'train_l1.json'), 'w'), indent=1); print('done in %.0f s' % res['wall_s'])

if __name__ == '__main__': main()
