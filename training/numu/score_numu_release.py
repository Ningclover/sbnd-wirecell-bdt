"""Score candidate tables with a frozen νμ release folder (manifest.json from freeze_numu_l2.py), alone.

Level 1 (if the manifest has one): per model, the objects of objs_<model>_<sample>.parquet with the chain's protections,
margin of l1_<model>.json, pooled per candidate (max / min), float32(tanh(scale * margin + offset)), the empty constant
where the candidate has no object; the value replaces <model>_score. Final: the 74 inputs in order, NaN -> 0 on the four
protected ones, margin of each frozen final model. Writes sample + key + one column per final model (+ the l1 values).

  python score_numu_release.py RELEASE_DIR TABLE_DIR SAMPLE OUT.parquet
"""
import json, os, sys
import numpy as np, pandas as pd, xgboost as xgb
KEY = ['sample', 'file_id', 'run', 'subrun', 'event', 'cand']

def main():
    rel, d, s, out = sys.argv[1:5]; man = json.load(open(os.path.join(rel, 'manifest.json'))); names = man['inputs']
    C = pd.read_parquet(os.path.join(d, 'cands_%s.parquet' % s), columns=list(dict.fromkeys(KEY + names)))
    for n in man['nan_to_0']: C[n] = C[n].fillna(0.0)
    if man.get('level1'):
        for m, t in man['level1']['models'].items():
            O = pd.read_parquet(os.path.join(d, 'objs_%s_%s.parquet' % (m, s)), columns=KEY + t['inputs'])
            for c, (kind, v) in t['protect'].items():
                x = O[c].to_numpy(float); O[c] = np.where(np.isinf(x) if kind == 'inf' else np.isnan(x), v, x)
            b = xgb.Booster(); b.load_model(os.path.join(rel, t['file']))
            O['p'] = b.predict(xgb.DMatrix(O[t['inputs']].to_numpy(np.float32), feature_names=t['inputs']), output_margin=True)
            g = O.groupby(KEY, sort=False).p.agg(t['pool']).rename('margin').reset_index()
            C = C.merge(g, on=KEY, how='left'); x = C.pop('margin').to_numpy(np.float64)
            C['l1_' + m] = np.where(np.isnan(x), np.float32(t['empty_value']), np.tanh(t['scale'] * x + t['offset']).astype(np.float32)).astype(np.float32)
            C[m + '_score'] = C['l1_' + m]
    X = xgb.DMatrix(C[names].to_numpy(np.float32), feature_names=names)
    R = C[KEY + [c for c in C.columns if c.startswith('l1_')]].copy()
    for tag, mm in man['models'].items():
        b = xgb.Booster(); b.load_model(os.path.join(rel, mm['file'])); R[tag] = b.predict(X, output_margin=True)
    R.to_parquet(out); print('%s: %d candidates scored with %s' % (s, len(R), ', '.join(man['models'])))

if __name__ == '__main__': main()
