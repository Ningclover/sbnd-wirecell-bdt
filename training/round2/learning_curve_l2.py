"""Does more training data help? The final model of round 3 (two-level on the rule-B level-1 scores, depth 10, signal share
1/2, exposure weights, early stopping) retrained on random fractions of the training rows, each fraction with several seeds,
measured on the development slice: AUC (all, BNB, nu_e), weighted purity at the shipped score's efficiency (the dev stand-in),
trees chosen. The level-1 scores are the full-data ones (stage 1 is not refit per fraction).

  python training/round2/learning_curve_l2.py MERGED_DIR L1_SCORES OUT_DIR --samples ... --weights S=w ...
         [--fractions 0.05 0.1 0.2 0.4 0.6 0.8 1.0] [--seeds 5 6 7] [--depth 10] [--floor 0.5] [--nthread 96]
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, xgboost as xgb
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from train_l2 import load, floored, weighted_eff_pur, dev_report, NAMES, SHIPPED_SCORES, KEY
from l1_models import NUE_L1

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('merged_dir'); ap.add_argument('l1_scores'); ap.add_argument('out_dir'); ap.add_argument('--samples', nargs='+', required=True); ap.add_argument('--weights', nargs='*', default=[])
    ap.add_argument('--fractions', nargs='*', type=float, default=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]); ap.add_argument('--seeds', nargs='*', type=int, default=[5, 6, 7]); ap.add_argument('--depth', type=int, default=10); ap.add_argument('--floor', type=float, default=0.5)
    ap.add_argument('--lr', type=float, default=0.1); ap.add_argument('--max-trees', type=int, default=2000); ap.add_argument('--patience', type=int, default=50); ap.add_argument('--nthread', type=int, default=96)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    W = {s: 1.0 for s in a.samples}
    for kv in a.weights: k, v = kv.split('='); W[k] = float(v)
    t0 = time.time(); C = load(a.merged_dir, a.samples, W); L1 = pd.read_parquet(a.l1_scores); C = C.merge(L1[KEY + ['l1_' + m for m in NUE_L1]], on=KEY, how='left')
    tr, dv = C[C.split == 'train'], C[C.split == 'dev']
    names = [('l1_' + NUE_L1[SHIPPED_SCORES.index(n)] if n in SHIPPED_SCORES else n) for n in NAMES]
    Xdv = dv[names].to_numpy(np.float32); wdv = floored(dv.w_exp.to_numpy(), dv.y.to_numpy(), a.floor); ddv = xgb.DMatrix(Xdv, label=dv.y.to_numpy(), weight=wdv, feature_names=names, missing=np.nan)
    e0, p0, _ = weighted_eff_pur(dv.nue_score.to_numpy(), dv.y.to_numpy(), dv.w_exp.to_numpy(), 7.0)
    res = dict(weights=W, depth=a.depth, floor=a.floor, lr=a.lr, fractions=a.fractions, seeds=a.seeds, train_rows=int(len(tr)), dev_rows=int(len(dv)), shipped_dev=dev_report(dv.nue_score.to_numpy(), dv, e0, cut=7.0), points=[])
    print('train %d rows (%d signal), dev %d; shipped at cut 7 on dev: eff %.3f purity %.3f AUC %.4f; %.0f s' % (len(tr), tr.y.sum(), len(dv), e0, p0, res['shipped_dev']['auc'], time.time() - t0), flush=True)
    for frac in a.fractions:
        for seed in a.seeds:
            rng = np.random.default_rng(seed); sub = tr if frac >= 1.0 else tr.iloc[rng.random(len(tr)) < frac]
            wtr = floored(sub.w_exp.to_numpy(), sub.y.to_numpy(), a.floor); dtr = xgb.DMatrix(sub[names].to_numpy(np.float32), label=sub.y.to_numpy(), weight=wtr, feature_names=names, missing=np.nan)
            params = dict(max_depth=a.depth, eta=a.lr, objective='binary:logistic', eval_metric='logloss', tree_method='hist', nthread=a.nthread, seed=seed, subsample=0.8, colsample_bylevel=0.3, colsample_bynode=0.3, gamma=0.1, base_score=0.5)
            t1 = time.time(); ev = {}; bst = xgb.train(params, dtr, num_boost_round=a.max_trees, evals=[(dtr, 'train'), (ddv, 'dev')], early_stopping_rounds=a.patience, evals_result=ev, verbose_eval=False)
            nt = bst.best_iteration + 1; sc = bst.predict(ddv, iteration_range=(0, nt), output_margin=True); r = dev_report(sc, dv, e0)
            pt = dict(fraction=frac, seed=seed, rows=int(len(sub)), signal=int(sub.y.sum()), trees=int(nt), train_logloss=float(ev['train']['logloss'][nt - 1]), dev_logloss=float(ev['dev']['logloss'][nt - 1]), auc=r['auc'], auc_bnb=next((v for k, v in r['auc_by_sample'].items() if 'bnb' in k), None), auc_nuecc=next((v for k, v in r['auc_by_sample'].items() if 'nuecc' in k), None), purity_at_e0=r['weighted_purity'], fit_s=round(time.time() - t1, 1))
            res['points'].append(pt); json.dump(res, open(os.path.join(a.out_dir, 'learning_curve_l2.json'), 'w'), indent=1)
            print('frac %.2f seed %d rows %6d trees %4d  logloss train %.4f dev %.4f  AUC %.4f bnb %.4f nue %.4f  purity@e0 %.3f  (%.0f s)' % (frac, seed, len(sub), nt, pt['train_logloss'], pt['dev_logloss'], pt['auc'], pt['auc_bnb'] or 0, pt['auc_nuecc'] or 0, pt['purity_at_e0'], pt['fit_s']), flush=True)
    res['wall_s'] = round(time.time() - t0, 1); json.dump(res, open(os.path.join(a.out_dir, 'learning_curve_l2.json'), 'w'), indent=1); print('done in %.0f s' % res['wall_s'])

if __name__ == '__main__': main()
