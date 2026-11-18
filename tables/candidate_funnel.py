"""Candidate funnel per sample, the way the small-sample study reported it:
events -> events with Wire-Cell output -> all candidates -> reco vertex in FV -> tagger ran (br_filled == 1)
-> train half / test half, plus the signal rows among the pool. Reads the merged spec_v1 tables.

    python candidate_funnel.py TABLE_DIR [--samples s1 s2 ...] [--json OUT]
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'spec'))
import spec_v1 as S

COLS = ['sample', 'file_id', 'run', 'subrun', 'event', 'cand', 'reco_x', 'reco_y', 'reco_z', 'kine_reco_Enu',
        'partner_kind', 'label_unreadable', 'dist_sce_cm', 'm_pdg', 'm_ccnc', 'm_vx', 'm_vy', 'm_vz', 'm_edep_mev',
        'br_filled', 'neutrino_type']

def funnel(table_dir, sample):
    c = pq.read_table(os.path.join(table_dir, f'cands_{sample}.parquet'), columns=COLS).to_pandas()
    f = pq.read_table(os.path.join(table_dir, f'files_{sample}.parquet')).to_pandas()
    br = c['br_filled'].to_numpy()
    placeholder = S.placeholder_mask(c['kine_reco_Enu'].to_numpy(), c['neutrino_type'].to_numpy(),
                                     c['reco_x'].to_numpy(), c['reco_y'].to_numpy(), c['reco_z'].to_numpy())
    real = ~placeholder
    fv = S.in_fv(c['reco_x'].to_numpy(), c['reco_y'].to_numpy(), c['reco_z'].to_numpy()) & real
    filled = fv & (br == 1)
    pool = S.pool_mask(br, c['reco_x'].to_numpy(), c['reco_y'].to_numpy(), c['reco_z'].to_numpy())
    assert (pool == filled).all(), 'pool_mask differs from FV & br_filled'
    sig = S.signal_mask(c['partner_kind'].to_numpy(), c['label_unreadable'].to_numpy(), c['dist_sce_cm'].to_numpy(),
                        c['m_pdg'].to_numpy(), c['m_ccnc'].to_numpy(), c['m_vx'].to_numpy(), c['m_vy'].to_numpy(),
                        c['m_vz'].to_numpy(), c['m_edep_mev'].to_numpy()) & pool
    train = S.split(c['sample'].to_numpy(), c['run'].to_numpy(), c['subrun'].to_numpy()) == 'train'
    usable = pool & (c['label_unreadable'].to_numpy() == 0)
    ev = c.loc[real, ['file_id', 'run', 'subrun', 'event']].drop_duplicates()
    return dict(
        sample=sample, files=int(len(f)), events=int(f['events'].sum()), wc_events=int(f['wc_events'].sum()),
        events_with_candidate=int(len(ev)),
        candidates=int(real.sum()), placeholder_rows=int(placeholder.sum()),
        vertex_in_fv=int(fv.sum()), br_filled=int(filled.sum()),
        br_filled_label_unreadable=int((pool & ~usable).sum()),
        train_rows=int((usable & train).sum()), test_rows=int((usable & ~train).sum()),
        train_signal=int((sig & usable & train).sum()), test_signal=int((sig & usable & ~train).sum()),
        br_filled_fraction_of_candidates=float(filled.sum() / max(real.sum(), 1)),
        candidates_per_wc_event=float(real.sum() / max(f['wc_events'].sum(), 1)))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('table_dir'); ap.add_argument('--samples', nargs='*'); ap.add_argument('--json')
    a = ap.parse_args()
    samples = a.samples or sorted(os.path.basename(p)[6:-8] for p in glob.glob(os.path.join(a.table_dir, 'cands_*.parquet')))
    rows = [funnel(a.table_dir, s) for s in samples]
    tot = {k: sum(r[k] for r in rows) for k in rows[0] if isinstance(rows[0][k], int)}
    tot['sample'] = 'all'
    rows.append(tot)
    keys = ['sample', 'files', 'events', 'wc_events', 'events_with_candidate', 'candidates', 'vertex_in_fv', 'br_filled',
            'br_filled_label_unreadable', 'train_rows', 'test_rows', 'train_signal', 'test_signal']
    print('\t'.join(keys))
    for r in rows: print('\t'.join(f'{r.get(k, "")}' for k in keys))
    if a.json: json.dump(rows, open(a.json, 'w'), indent=1)
