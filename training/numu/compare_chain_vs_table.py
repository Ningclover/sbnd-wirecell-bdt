#!/usr/bin/env python3
"""νμ release in the chain against the same release scored from the builder's tables, candidate by candidate.

Chain side: the scores the chain stored in the merged CAF (candidate table columns numu_1_score, numu_2_score,
cosmict_10_score, numu_score). Table side: training/numu/score_numu_release.py on the same tables (simulation:
corrected first with make_corrected_tables.py --update-flag1; data: as built), written to SCORES.parquet.

  python compare_chain_vs_table.py RELEASE_DIR TABLE_DIR SAMPLE SCORES.parquet [--events 10] [--json OUT.json]

Prints the first N events (run, subrun, event order) that hold a candidate with numu_cc_flag == 1, every candidate of
those events, and the largest differences over all candidates.
"""
import argparse, json, math, os
import numpy as np, pandas as pd

K = ['file_id', 'run', 'subrun', 'event', 'cand']
L1 = ['numu_1', 'numu_2', 'cosmict_10']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('release'); ap.add_argument('tables'); ap.add_argument('sample'); ap.add_argument('scores')
    ap.add_argument('--events', type=int, default=10); ap.add_argument('--json')
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.release, 'manifest.json'))); cut = man['operating_point']['cut_margin']; k = 2 / math.log(10)
    C = pd.read_parquet(os.path.join(a.tables, 'cands_%s.parquet' % a.sample), columns=K + ['numu_cc_flag', 'numu_score'] + [m + '_score' for m in L1])
    S = pd.read_parquet(a.scores)
    M = C.merge(S[K + ['l1_' + m for m in L1] + ['d6_f0']], on=K, how='inner')
    assert len(M) == len(C) == len(S), (len(C), len(S), len(M))
    M['table_numu_score'] = (k * M.d6_f0).astype(np.float32)
    for m in L1: M['d_' + m] = (M[m + '_score'] - M['l1_' + m]).abs()
    M['d_numu_score'] = (M.numu_score - M.table_numu_score).abs()
    M['pass_chain'] = M.numu_score > k * cut; M['pass_table'] = M.d6_f0 > cut
    ev = M[M.numu_cc_flag == 1][['file_id', 'run', 'subrun', 'event']].drop_duplicates().sort_values(['run', 'subrun', 'event']).head(a.events)
    T = M.merge(ev, on=['file_id', 'run', 'subrun', 'event']).sort_values(K)
    cols = ['run', 'subrun', 'event', 'cand', 'numu_cc_flag'] + sum([[m + '_score', 'l1_' + m] for m in L1], []) + ['numu_score', 'table_numu_score', 'pass_chain', 'pass_table']
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(T[cols].to_string(index=False, float_format=lambda x: '%.6f' % x))
    summ = dict(candidates=len(M), events_shown=len(ev), candidates_shown=len(T),
                max_abs_diff={c: float(M['d_' + c].max()) for c in L1 + ['numu_score']},
                shown_max_abs_diff={c: float(T['d_' + c].max()) for c in L1 + ['numu_score']},
                pass_chain=int(M.pass_chain.sum()), pass_table=int(M.pass_table.sum()), pass_disagree=int((M.pass_chain != M.pass_table).sum()),
                cut_numu_score=k * cut)
    print(json.dumps(summ, indent=1))
    if a.json:
        json.dump(dict(summary=summ, rows=T[cols].to_dict('records')), open(a.json, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


if __name__ == '__main__':
    main()
