"""How many signal and background objects each level-1 small model has to learn from, on the v1.2 tables, under
MicroBooNE's labelling A and labelling B (every object inherits its candidate's label), as wcgpu1 read them off the
MicroBooNE trainers (project record, wcgpu/level1/scripts/make_labels.py): for the nu_e models the rule "fires" where the
flag is 0 (the object was flagged bad), and A keeps every signal object and only the background objects where the rule
fired; for the nu_mu models the rule fires where the flag is 1 and A keeps every background object and only the signal
objects where the rule fired. Objects A leaves out are counted as unused. Pool
candidates only (spec_v1.pool_mask), the model's pre-cut applied (spec/l1_models.py), per sample and per v1.1 split,
raw and with the exposure weights at the BNB POT.

  python tables/l1_label_counts.py MERGED_DIR --samples S1 S2 ... [--weights S1=w1 S2=w2 ...] [--json out.json]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'spec'))
import spec_v1
from l1_models import L1_MODELS, NUE_L1
KEY = ['file_id', 'run', 'subrun', 'event', 'cand']

def pre_cut(C, expr):
    if not expr: return np.ones(len(C), bool)
    return C.eval(expr).to_numpy(bool)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('merged_dir'); ap.add_argument('--samples', nargs='+', required=True); ap.add_argument('--weights', nargs='*', default=[]); ap.add_argument('--json')
    a = ap.parse_args(); W = {s: 1.0 for s in a.samples}
    for kv in a.weights: k, v = kv.split('='); W[k] = float(v)
    need = KEY + ['br_filled', 'reco_x', 'reco_y', 'reco_z', 'partner_kind', 'label_unreadable', 'dist_sce_cm', 'm_pdg', 'm_ccnc', 'm_vx', 'm_vy', 'm_vz', 'm_edep_mev',
                  'pio_filled', 'pio_flag_pio', 'cosmict_flag_1'] + ['n_obj_' + m for m in L1_MODELS]
    out = {'weights': W, 'samples': {}, 'models': {}}
    acc = {m: {lab: {sp: {k: 0.0 for k in ('s', 'b', 'x', 's_w', 'b_w')} for sp in ('train', 'dev')} for lab in ('A', 'B')} for m in L1_MODELS}
    for s in a.samples:
        C = pd.read_parquet(os.path.join(a.merged_dir, 'cands_%s.parquet' % s), columns=need)
        C = C[C.label_unreadable == 0]
        C = C[spec_v1.pool_mask(C.br_filled, C.reco_x, C.reco_y, C.reco_z)].copy()
        C['sig'] = spec_v1.signal_mask(C.partner_kind, C.label_unreadable, C.dist_sce_cm, C.m_pdg, C.m_ccnc, C.m_vx, C.m_vy, C.m_vz, C.m_edep_mev)
        C['split'] = [spec_v1.split_v11(s, r, sr) for r, sr in zip(C.run, C.subrun)]
        out['samples'][s] = dict(pool=int(len(C)), signal=int(C.sig.sum()), train=int((C.split == 'train').sum()), dev=int((C.split == 'dev').sum()))
        for m, md in L1_MODELS.items():
            keep = C[pre_cut(C, md['pre_cut'])][KEY + ['sig', 'split']]
            O = pd.read_parquet(os.path.join(a.merged_dir, 'objs_%s_%s.parquet' % (m, s)), columns=KEY + ['flag'])
            O = O.merge(keep, on=KEY, how='inner')
            for sp in ('train', 'dev'):
                o = O[O.split == sp]
                for lab in ('A', 'B'):
                    sgn = o.sig.to_numpy(bool); ra = md['rule_a']; fired = (o.flag.to_numpy() == ra['fired'])
                    if lab == 'B': is_s, is_b = sgn, ~sgn
                    elif ra['narrows'] == 'signal': is_s, is_b = sgn & fired, ~sgn        # numu_1, numu_2
                    else:                           is_s, is_b = sgn, (~sgn) & fired      # the 15 nu_e models, cosmict_10
                    is_x = ~(is_s | is_b)
                    d = acc[m][lab][sp]; d['s'] += int(is_s.sum()); d['b'] += int(is_b.sum()); d['x'] += int(is_x.sum()); d['s_w'] += W[s] * is_s.sum(); d['b_w'] += W[s] * is_b.sum()
                    out['models'].setdefault(m, {}).setdefault(s, {}).setdefault(lab, {})[sp] = dict(s=int(is_s.sum()), b=int(is_b.sum()), x=int(is_x.sum()))
    out['totals'] = acc
    print('samples:', json.dumps(out['samples']))
    print('%-11s | %9s %9s %9s | %9s %9s | %11s %11s | %9s %9s' % ('model', 'A sig', 'A bkg', 'A unused', 'B sig', 'B bkg', 'A sig w', 'A bkg w', 'dev A sig', 'dev A bkg'))
    for m in L1_MODELS:
        t = acc[m]; A, B, D = t['A']['train'], t['B']['train'], t['A']['dev']
        print('%-11s | %9d %9d %9d | %9d %9d | %11.1f %11.1f | %9d %9d' % (m, A['s'], A['b'], A['x'], B['s'], B['b'], A['s_w'], A['b_w'], D['s'], D['b']))
    if a.json: json.dump(out, open(a.json, 'w'), indent=1); print('wrote', a.json)

if __name__ == '__main__': main()
