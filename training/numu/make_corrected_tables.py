"""νμ round 3: a copy of the merged tables with the dQ/dx inputs of the SIMULATED samples corrected towards data.
Every other file is linked unchanged, so the round 2 training scripts run on OUT exactly as on MERGED.

Corrected inputs (all in MIP units; only filled, positive values):
  objs_numu_1_<s>.parquet : numu_cc_1_medium_dQ_dx                      (per vertex segment)
  cands_<s>.parquet       : cosmict_{2,3,7}_dQ_dx_{end,front}           (where cosmict_N_filled == 1)
Correction (gpvm #41, JSON):
  {"mode": "uniform", "k": 1.07}                                         q -> k q
  {"mode": "linear", "a": .., "b": .., "q_min": .., "q_max": ..}         q -> q (a + b clip(q, q_min, q_max))
Data samples (beam-off) are never corrected.

  python make_corrected_tables.py MERGED OUT CORRECTION.json --samples wcl1116-bnb wcl1116-ncdr [--update-flag1]
--update-flag1: per numu_1 segment numu_cc_flag_1 stays on only if the corrected dQ/dx < numu_cc_1_dQ_dx_cut; numu_cc_flag recomputed.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

COS = ['cosmict_%d_dQ_dx_%s' % (n, e) for n in (2, 3, 7) for e in ('end', 'front')]

def factor(q, c):
    if c['mode'] == 'uniform':
        return np.full_like(q, c['k'])
    if c['mode'] == 'linear':
        return c['a'] + c['b'] * np.clip(q, c['q_min'], c['q_max'])
    raise ValueError(c['mode'])

def correct(col, mask, c):
    q = col.to_numpy(zero_copy_only=False).astype(np.float64)
    ok = mask & np.isfinite(q) & (q > 0)
    out = q.copy(); out[ok] = q[ok] * factor(q[ok], c)
    return pa.array(out.astype(col.type.to_pandas_dtype()), type=col.type), int(ok.sum()), float(np.median(out[ok] / q[ok])) if ok.any() else None

K = ['file_id', 'run', 'subrun', 'event', 'cand']

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('merged'); ap.add_argument('out'); ap.add_argument('correction')
    ap.add_argument('--samples', nargs='+', required=True)
    ap.add_argument('--update-flag1', action='store_true',
                    help='also make the numu_1 muon flag follow the corrected dQ/dx (gpvm #41): per segment '
                         'flag = flag AND (medium_dQ_dx_corr < numu_cc_1_dQ_dx_cut); per candidate '
                         'numu_cc_flag recomputed as any segment flag_1 OR any numu_cc_2 flag OR numu_cc_flag_3')
    a = ap.parse_args(); c = json.load(open(a.correction)); os.makedirs(a.out, exist_ok=True)
    rep = dict(correction=c, merged=os.path.abspath(a.merged), samples=a.samples, update_flag1=a.update_flag1, columns={}, flags={})
    targets = {'cands_%s.parquet' % s for s in a.samples} | {'objs_numu_1_%s.parquet' % s for s in a.samples}
    off = {}   # sample -> candidate keys whose flag_1 turned off
    # objects before candidates, so the candidate flags can follow the segment flags
    for f in sorted(os.listdir(a.merged), key=lambda f: (not f.startswith('objs_numu_1_'), f)):
        src, dst = os.path.join(a.merged, f), os.path.join(a.out, f)
        if os.path.lexists(dst): os.remove(dst)
        if f not in targets:
            os.symlink(src, dst); continue
        t = pq.read_table(src)
        if f.startswith('objs_numu_1_'):
            cols = {'numu_cc_1_medium_dQ_dx': np.ones(t.num_rows, bool)}
        else:
            cols = {c_: (t.column('cosmict_%s_filled' % c_.split('_')[1]).to_numpy(zero_copy_only=False) == 1) for c_ in COS}
        for name, mask in cols.items():
            i = t.schema.get_field_index(name)
            new, n, med = correct(t.column(name).combine_chunks(), mask, c)
            t = t.set_column(i, name, new)
            rep['columns']['%s:%s' % (f, name)] = dict(n_corrected=n, median_factor=med)
            print('%-40s %-26s corrected %8d  median factor %s' % (f, name, n, med), flush=True)
        if a.update_flag1 and f.startswith('objs_numu_1_'):
            s_ = f[len('objs_numu_1_'):-len('.parquet')]
            fl = t.column('flag').to_numpy(zero_copy_only=False).astype(np.float64)
            q = t.column('numu_cc_1_medium_dQ_dx').to_numpy(zero_copy_only=False).astype(np.float64)
            cut = t.column('numu_cc_1_dQ_dx_cut').to_numpy(zero_copy_only=False).astype(np.float64)
            newfl = np.where((fl == 1) & ~(q < cut), 0.0, fl)
            t = t.set_column(t.schema.get_field_index('flag'), 'flag', pa.array(newfl.astype(t.schema.field('flag').type.to_pandas_dtype()), type=t.schema.field('flag').type))
            df = t.select(K).to_pandas(); df['old'] = fl == 1; df['new'] = newfl == 1
            g = df.groupby(K)[['old', 'new']].any()
            off[s_] = g[g.old & ~g.new].reset_index()[K]
            rep['flags'][s_] = dict(segments_flag_on=int((fl == 1).sum()), segments_turned_off=int(((fl == 1) & (newfl != 1)).sum()),
                                    candidates_flag1_turned_off=int(len(off[s_])))
            print('%-40s flag 1: %d of %d flagged segments off; %d candidates lose flag_1' % (f, rep['flags'][s_]['segments_turned_off'], rep['flags'][s_]['segments_flag_on'], len(off[s_])), flush=True)
        if a.update_flag1 and f.startswith('cands_'):
            # numu_cc_flag = any segment flag_1 OR any numu_cc_2 flag OR numu_cc_flag_3 (NeutrinoTaggerNuMu.cxx, checked exact on
            # every non-placeholder bnb candidate); the candidate column numu_cc_flag_1 is not a tagger value and not an input.
            s_ = f[len('cands_'):-len('.parquet')]
            df = t.select(K + ['numu_cc_flag', 'numu_cc_flag_3']).to_pandas()
            o2 = pd.read_parquet(os.path.join(a.merged, 'objs_numu_2_%s.parquet' % s_), columns=K + ['flag'])
            a2 = o2.assign(a2=o2.flag == 1).groupby(K).a2.any().reset_index()
            m = df[K].merge(off[s_].assign(_off=True), on=K, how='left').merge(a2, on=K, how='left')
            hit = m._off.fillna(False).to_numpy(bool) & ~m.a2.fillna(False).to_numpy(bool) & (df.numu_cc_flag_3.to_numpy() != 1)
            f0 = np.where(hit, 0.0, df.numu_cc_flag.to_numpy(np.float64))
            fld = t.schema.field('numu_cc_flag'); t = t.set_column(t.schema.get_field_index('numu_cc_flag'), 'numu_cc_flag', pa.array(f0.astype(fld.type.to_pandas_dtype()), type=fld.type))
            rep['flags'][s_]['candidates_numu_cc_flag_turned_off'] = int(((df.numu_cc_flag.to_numpy() == 1) & (f0 != 1)).sum())
            print('%-40s numu_cc_flag off on %d candidates (flag_1 lost on %d)' % (f, rep['flags'][s_]['candidates_numu_cc_flag_turned_off'], len(off[s_])), flush=True)
        pq.write_table(t, dst)
    json.dump(rep, open(os.path.join(a.out, 'correction_applied.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
