"""Paired comparison of νμ models on the Aurora dev slice at a weighted-purity anchor.
python compare_rounds_dev.py NAME=DEV_SCORES.parquet:COLUMN ... [--purity 0.95]   (first one is the reference)"""
import argparse, sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import train_numu_l2 as T
ap = argparse.ArgumentParser(); ap.add_argument('models', nargs='+'); ap.add_argument('--purity', type=float, default=0.95); a = ap.parse_args()
base = None; res = {}
for spec in a.models:
    name, rest = spec.split('='); path, col = rest.rsplit(':', 1)
    D = pd.read_parquet(path).set_index(['sample'] + T.KEY).sort_index()
    if base is None: base = D
    assert D.index.equals(base.index)
    op, sel = T.at_purity(D[col].to_numpy(), D.y.to_numpy(), D.w_exp.to_numpy(), a.purity)
    s = D.y.to_numpy() == 1; e = D.m_E.to_numpy() * 1000
    bins = [round(100 * D.w_exp[(s & (e >= lo) & (e < hi))].to_numpy()[sel[s & (e >= lo) & (e < hi)]].sum() / D.w_exp[(s & (e >= lo) & (e < hi))].sum(), 1) for lo, hi in zip(T.E_BINS[1:-1], T.E_BINS[2:])]
    res[name] = dict(sel=sel, eff=op['eff'], err=op['eff_err'], bins=bins)
names = list(res); ref = res[names[0]]; s = base.y.to_numpy() == 1
print('efficiency at %.0f %% purity; true-energy bins %s MeV' % (100 * a.purity, ['%g-%g' % (lo, hi) for lo, hi in zip(T.E_BINS[1:-1], T.E_BINS[2:])]))
for n in names:
    r = res[n]; print('%-22s %.4f ± %.4f  paired signal vs %s: only this %5d, only ref %5d   bins %s' % (n, r['eff'], r['err'], names[0], (r['sel'] & ~ref['sel'] & s).sum(), (~r['sel'] & ref['sel'] & s).sum(), r['bins']))
