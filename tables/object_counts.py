"""Object-row counts of the v1.2 tables per sample and per level-1 model, in the form of wcgpu1's r7 audit (project record,
wcgpu/level1/output/l1_objects_audit.json): candidates, candidates with objects, empty fraction, objects per candidate,
rule-flag rate; over all candidates and over the pool (br_filled == 1). Optionally the reference audit for a side-by-side.

  python tables/object_counts.py MERGED_DIR --samples S1 S2 ... [--reference l1_objects_audit.json] [--json out.json]
"""
import argparse, json, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'spec'))
from l1_models import L1_MODELS, NUE_L1

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('merged_dir'); ap.add_argument('--samples', nargs='+', required=True); ap.add_argument('--reference'); ap.add_argument('--json')
    a = ap.parse_args(); ref = json.load(open(a.reference))['models'] if a.reference else {}
    out = {}
    for s in a.samples:
        C = pd.read_parquet(os.path.join(a.merged_dir, 'cands_%s.parquet' % s), columns=['br_filled'] + ['n_obj_' + m for m in L1_MODELS])
        P = C[C.br_filled == 1]; r = dict(candidates=len(C), pool=len(P), models={})
        for m in L1_MODELS:
            O = pd.read_parquet(os.path.join(a.merged_dir, 'objs_%s_%s.parquet' % (m, s)), columns=['flag'])
            n, npool = C['n_obj_' + m], P['n_obj_' + m]
            r['models'][m] = dict(objects=int(n.sum()), objects_per_candidate=round(float(n.mean()), 3), empty_fraction=round(float((n == 0).mean()), 4),
                                  pool_objects_per_candidate=round(float(npool.mean()), 3), pool_empty_fraction=round(float((npool == 0).mean()), 4),
                                  objects_per_nonempty=round(float(n[n > 0].mean()), 3) if (n > 0).any() else 0.0, max_length=int(n.max()), rule_fired_fraction=round(float((O.flag == 1).mean()), 4) if len(O) else None)
        nue = sum(r['models'][m]['objects'] for m in NUE_L1)
        r['nue_objects_per_pool_candidate'] = round(sum(r['models'][m]['pool_objects_per_candidate'] for m in NUE_L1), 2)
        r['pio_2_share_of_nue_objects'] = round(r['models']['pio_2']['objects'] / nue, 3) if nue else None
        out[s] = r
        print('%s: %d candidates, %d in the pool; nu_e objects per pool candidate %.1f (pio_2 share %.2f)' % (s, len(C), len(P), r['nue_objects_per_pool_candidate'], r['pio_2_share_of_nue_objects'] or 0))
        print('  %-11s %9s %8s %7s %8s | %8s %7s | %s' % ('model', 'objects', 'per cand', 'empty', 'nonempty', 'pool/cd', 'p.empty', 'wcgpu1 r7: per cand, empty, per nonempty' if ref else ''))
        for m, d in r['models'].items():
            print('  %-11s %9d %8.2f %7.3f %8.2f | %8.2f %7.3f | %s' % (m, d['objects'], d['objects_per_candidate'], d['empty_fraction'], d['objects_per_nonempty'], d['pool_objects_per_candidate'], d['pool_empty_fraction'], ('%.2f  %.3f  %.2f' % (ref[m]['objects_per_candidate'], ref[m]['empty_fraction'], ref[m]['objects_per_nonempty'])) if m in ref else ''))
    if a.json: json.dump(out, open(a.json, 'w'), indent=1); print('wrote', a.json)

if __name__ == '__main__': main()
