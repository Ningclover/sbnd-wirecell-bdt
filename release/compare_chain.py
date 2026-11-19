"""Check that a model running inside the Wire-Cell chain gives the same scores as the standalone release.

Run the chain with the converted model on a few files, build tables from its output with tables/build_tables.py --objs
(the chain's scores land in the usual columns: nue_score and the 15 <model>_score), then:

  python release/compare_chain.py RELEASE_DIR TABLE_DIR SAMPLE [--variant main] [--tol 1e-4] [--out compare.json]

For every candidate with the tagger run, the standalone release scores the same tables (release/score_release.py) and the
chain's stored values are compared: nue_score against the level-2 margin, <model>_score against the pooled level-1
margin (NaN must stay NaN: an empty list or a failed pre-cut gives a missing value, not the shipped default). Prints the
largest differences and the candidates outside the tolerance; exit code 1 if any.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from score_release import score, KEY

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('release'); ap.add_argument('table_dir'); ap.add_argument('sample')
    ap.add_argument('--variant', default='main'); ap.add_argument('--tol', type=float, default=1e-4); ap.add_argument('--out')
    a = ap.parse_args(); man = json.load(open(os.path.join(a.release, 'manifest.json')))
    S = score(a.release, man, a.table_dir, a.sample, a.variant)
    pairs = [('nue_score', 'score')] + [(md['shipped_score_name'], md['level2_input_name']) for md in man['level1']['models'].values()]
    chain = pd.read_parquet(os.path.join(a.table_dir, 'cands_%s.parquet' % a.sample), columns=KEY + [p for p, _ in pairs])
    M = S[KEY + [q for _, q in pairs]].merge(chain, on=KEY, how='left')
    res = dict(sample=a.sample, candidates=int(len(M)), tol=a.tol, columns={}); bad = 0
    for p, q in pairs:
        x, y = M[p].to_numpy(float), M[q].to_numpy(float)
        nan_mismatch = int(np.sum(np.isnan(x) != np.isnan(y))); both = ~np.isnan(x) & ~np.isnan(y)
        d = np.abs(x[both] - y[both]); n_out = int(np.sum(d > a.tol)) + nan_mismatch; bad += n_out
        res['columns'][p] = dict(compared=int(both.sum()), nan_both=int(np.sum(np.isnan(x) & np.isnan(y))), nan_mismatch=nan_mismatch,
                                 max_abs_diff=float(d.max()) if len(d) else 0.0, outside_tol=n_out)
        print('%-12s vs %-10s compared %7d  NaN both %6d  NaN mismatch %5d  max |diff| %.3g  outside tol %d' % (p, q, both.sum(), res['columns'][p]['nan_both'], nan_mismatch, res['columns'][p]['max_abs_diff'], n_out))
    res['pass'] = bad == 0; print('PASS' if bad == 0 else 'FAIL: %d values outside tolerance' % bad)
    if a.out: json.dump(res, open(a.out, 'w'), indent=1)
    sys.exit(0 if bad == 0 else 1)

if __name__ == '__main__':
    main()
