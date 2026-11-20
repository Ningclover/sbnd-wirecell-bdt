"""νμ round 2, stage 1: retrain the three small models that feed the final νμ model (numu_1, numu_2, cosmict_10) and write,
for every pool candidate, the level-1 value exactly as the Wire-Cell chain will store it with the converted model.

Rows: the νμ pool of train_numu_l2.load (reco vertex in FV, no gate; ncdr NC-partner rows only), label numu_signal,
split split_v11, exposure weights. Objects from objs_<model>_<sample>.parquet, joined on sample + key.
Training rows (wcgpu #25): MicroBooNE's training cut cosmict_flag_1 == 0 on both sides, then
  rule A: numu_1 / numu_2 keep signal objects only where the model's flag == 1 (every background object);
          cosmict_10 keeps background objects only where its flag == 1 (every signal object);
  rule B: every object takes its candidate's label.
Input protections as in the chain: numu_cc_1_dQ_dx_cut inf -> 10, cosmict_10_angle_beam NaN -> 0.
Weights: exposure, then both classes rescaled to equal total weight (TMVA NormMode=NumEvents, as MicroBooNE).
Learner: xgboost depth 3, hist, binary:logistic; one fit per learning rate in --lr, early stopping on the weighted dev
log-loss; the rate with the lowest dev log-loss is kept and cut to its best iteration (l1_<model>.json).

Chain value (no gate: the chain runs the three models on every candidate): margin pooled over ALL the candidate's objects
(max for numu_1/2, min for cosmict_10), then value = float32(tanh(scale * margin + offset)), with scale / offset from
wcgpu's l1_chain_transform recipe (affine_for_range(forest, empty + 0.05, 0.99), EMPTY_CONST); the empty constant where the
candidate has no object. Written: <out>/l1_<model>.json, <out>/l1_chain_transform.json, <out>/l1_values_<label>.parquet
(sample + key + l1_<model>), <out>/train_numu_l1.json.

  python training/numu/train_numu_l1.py MERGED OUT --samples ... --weights S=w ... --nc-only S --label A|B
         [--lr 0.05 0.1 0.3] [--max-trees 3000] [--patience 50] [--nthread 8] --wcgpu-scripts DIR
"""
import argparse, json, math, os, sys, time
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'spec'))
import train_numu_l2 as T
from l1_models import L1_MODELS, RULE_A
MODELS = ['numu_1', 'numu_2', 'cosmict_10']
SK = ['sample'] + T.KEY
PROTECT = {'numu_cc_1_dQ_dx_cut': ('inf', 10.0), 'cosmict_10_angle_beam': ('nan', 0.0)}
VHI, MARGIN_ABOVE = 0.99, 0.05

def balanced(w, y):
    w = w.astype(float).copy(); s = y == 1; tot = w.sum(); w[s] *= 0.5 * tot / w[s].sum(); w[~s] *= 0.5 * tot / w[~s].sum(); return w

def objects(d, m, samples, cols):
    O = pd.concat([pd.read_parquet(os.path.join(d, 'objs_%s_%s.parquet' % (m, s)), columns=SK + cols + ['flag']) for s in samples], ignore_index=True)
    for c in cols:
        if c in PROTECT:
            kind, v = PROTECT[c]; x = O[c].to_numpy(float)
            O[c] = np.where(np.isinf(x) if kind == 'inf' else np.isnan(x), v, x)
    return O

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('merged_dir'); ap.add_argument('out_dir'); ap.add_argument('--samples', nargs='+', required=True)
    ap.add_argument('--weights', nargs='*', default=[]); ap.add_argument('--nc-only', nargs='*', default=[]); ap.add_argument('--label', choices=['A', 'B'], default='A')
    ap.add_argument('--lr', nargs='*', type=float, default=[0.05, 0.1, 0.3]); ap.add_argument('--max-trees', type=int, default=3000); ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--nthread', type=int, default=8); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--wcgpu-scripts', required=True)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    sys.path.insert(0, a.wcgpu_scripts)
    from xgb_json_forest import XgbJsonForest
    from xgb_to_tmva import affine_for_range
    from check_tmva_conversion import EMPTY_CONST
    W = {s: 1.0 for s in a.samples}
    for kv in a.weights: k, v = kv.split('='); W[k] = float(v)
    t0 = time.time(); C, _ = T.load(a.merged_dir, a.samples, W, a.nc_only)
    res = dict(label=a.label, weights=W, nc_only=a.nc_only, lrs=a.lr, xgboost=xgb.__version__, models={}); out = C[SK].copy(); TR = {}
    for m in MODELS:
        md = L1_MODELS[m]; cols = md['lists']; ra = RULE_A[m]; t1 = time.time()
        O = objects(a.merged_dir, m, a.samples, cols).merge(C[SK + ['y', 'split', 'w_exp', 'cosmict_flag_1', md['score']]], on=SK, how='inner')
        sig = O.y.to_numpy() == 1; fired = O.flag.to_numpy() == ra['fired']
        use = np.ones(len(O), bool) if a.label == 'B' else ((~sig) | fired if ra['narrows'] == 'signal' else sig | fired)
        use &= (O.cosmict_flag_1.to_numpy() == 0)
        U = O[use]; tr, dv = U[U.split == 'train'], U[U.split == 'dev']
        r = dict(inputs=cols, objects_in_pool=int(len(O)), rows=dict(train=dict(s=int(tr.y.sum()), b=int((tr.y == 0).sum())), dev=dict(s=int(dv.y.sum()), b=int((dv.y == 0).sum()))), fits={})
        print('%s: %d objects on pool candidates; training rows train %d s / %d b, dev %d s / %d b' % (m, len(O), r['rows']['train']['s'], r['rows']['train']['b'], r['rows']['dev']['s'], r['rows']['dev']['b']), flush=True)
        dtr = xgb.DMatrix(tr[cols].to_numpy(np.float32), label=tr.y.to_numpy(), weight=balanced(tr.w_exp.to_numpy(), tr.y.to_numpy()), feature_names=cols)
        ddv = xgb.DMatrix(dv[cols].to_numpy(np.float32), label=dv.y.to_numpy(), weight=balanced(dv.w_exp.to_numpy(), dv.y.to_numpy()), feature_names=cols)
        best = None
        for lr in a.lr:
            params = dict(max_depth=3, eta=lr, objective='binary:logistic', eval_metric='logloss', tree_method='hist', nthread=a.nthread, seed=a.seed, base_score=0.5)
            ev = {}; bst = xgb.train(params, dtr, num_boost_round=a.max_trees, evals=[(dtr, 'train'), (ddv, 'dev')], early_stopping_rounds=a.patience, evals_result=ev, verbose_eval=False)
            nt = bst.best_iteration + 1; f = dict(trees=int(nt), train_logloss=float(ev['train']['logloss'][nt - 1]), dev_logloss=float(ev['dev']['logloss'][nt - 1]))
            r['fits']['%g' % lr] = f; print('  lr %-5g trees %4d logloss train %.4f dev %.4f' % (lr, nt, f['train_logloss'], f['dev_logloss']), flush=True)
            if best is None or f['dev_logloss'] < best[1]: best = (lr, f['dev_logloss'], bst, nt)
        r['kept_lr'] = best[0]; r['kept_trees'] = best[3]
        p = os.path.join(a.out_dir, 'l1_%s.json' % m); best[2][:best[3]].save_model(p)
        forest = XgbJsonForest(p); const = EMPTY_CONST[m]; scale, offset = affine_for_range(forest, const + MARGIN_ABOVE, VHI); mlo, mhi = forest.margin_bounds()
        TR[m] = dict(file=os.path.basename(p), scale=scale, offset=offset, empty_value=const, pool=md['pool'], margin_min=mlo, margin_max=mhi)
        bst = xgb.Booster(); bst.load_model(p)
        O['p'] = bst.predict(xgb.DMatrix(O[cols].to_numpy(np.float32), feature_names=cols), output_margin=True)
        pooled = O.groupby(SK, sort=False).p.agg(md['pool']).rename('margin').reset_index()
        out = out.merge(pooled, on=SK, how='left'); x = out.pop('margin').to_numpy(np.float64)
        out['l1_' + m] = np.where(np.isnan(x), np.float32(const), np.tanh(scale * x + offset).astype(np.float32)).astype(np.float32)
        # pooled candidate AUC on dev, all candidates (empty -> constant), against the shipped score
        dmask = (C.split == 'dev').to_numpy(); yv = C.y.to_numpy()[dmask]; wv = C.w_exp.to_numpy()[dmask]
        r['dev_candidate_auc'] = dict(new=float(roc_auc_score(yv, out['l1_' + m].to_numpy()[dmask], sample_weight=wv)), shipped=float(roc_auc_score(yv, C[md['score']].to_numpy()[dmask], sample_weight=wv)))
        r['empty_share'] = float(np.isnan(x).mean()); r['fit_s'] = round(time.time() - t1, 1); res['models'][m] = r
        print('  kept lr %g, %d trees; chain range %.3f..%.3f (empty %g, %.1f %% of pool); dev candidate AUC new %.4f shipped %.4f (%.0f s)' % (
            best[0], best[3], math.tanh(scale * mlo + offset), math.tanh(scale * mhi + offset), const, 100 * r['empty_share'], r['dev_candidate_auc']['new'], r['dev_candidate_auc']['shipped'], r['fit_s']), flush=True)
    json.dump(dict(models=TR, recipe='affine_for_range(forest, EMPTY_CONST + %g, %g) from wcgpu/release/scripts' % (MARGIN_ABOVE, VHI)), open(os.path.join(a.out_dir, 'l1_chain_transform.json'), 'w'), indent=1)
    out.to_parquet(os.path.join(a.out_dir, 'l1_values_%s.parquet' % a.label)); res['wall_s'] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out_dir, 'train_numu_l1.json'), 'w'), indent=1); print('done in %.0f s' % res['wall_s'])

if __name__ == '__main__': main()
