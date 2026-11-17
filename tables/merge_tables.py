"""Merge the per-job table slices of one sample into one parquet per table, and bring older slices up to the
current builder: integer key columns, is_placeholder, truth_pf_unreadable. Files that were rebuilt later (a *_retry* slice)
replace the failed rows of the same file_id. Usage: python merge_tables.py <in_dir> <sample> <out_dir>"""
import sys, os, glob, json
import numpy as np, pandas as pd
sys.path.insert(0, '/lus/flare/projects/neutrinoGPU/xning/bdt-comm/steps/step3_spec'); import spec_v1 as S
ind, sample, outd = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(outd, exist_ok=True)
KEYS = ['run', 'subrun', 'event', 'cand', 'nu_idx']
summary = {}
for table in ('files', 'cands', 'truth'):
    parts = sorted(glob.glob(os.path.join(ind, f'{table}_{sample}_*.parquet')))
    retry = [p for p in parts if '_retry' in p]; main = [p for p in parts if '_retry' not in p]
    frames = [pd.read_parquet(p) for p in main]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if retry:
        R = pd.concat([pd.read_parquet(p) for p in retry], ignore_index=True)
        redo = set(R.file_id); df = df[~df.file_id.isin(redo)] if len(df) else df; df = pd.concat([df, R], ignore_index=True)
    for k in KEYS:
        if k in df.columns and pd.api.types.is_numeric_dtype(df[k]): df[k] = df[k].astype('int64')
    if table == 'cands' and 'is_placeholder' not in df.columns:
        df['is_placeholder'] = S.placeholder_mask(df.kine_reco_Enu, df.neutrino_type, df.reco_x, df.reco_y, df.reco_z).astype(int)
    if table in ('truth', 'files') and 'truth_pf_unreadable' not in df.columns:
        df['truth_pf_unreadable'] = 0
    df.to_parquet(os.path.join(outd, f'{table}_{sample}.parquet'))
    import time; time.sleep(2)   # Lustre: let the write settle before the read-back below
    summary[table] = dict(parts=len(main), retry_parts=len(retry), rows=int(len(df)))
F = pd.read_parquet(os.path.join(outd, f'files_{sample}.parquet'))
summary['files_detail'] = dict(files=int(len(F)), errors=int((F.error != '').sum()), label_unreadable=int(F.label_unreadable.sum()), truth_pf_unreadable=int(F.truth_pf_unreadable.sum()),
                               events=int(F.events.sum()), wc_events=int(F.wc_events.sum()), candidates=int(F.candidates.sum()), pot_hdr=float(F.pot_hdr.sum()))
if 'cands' in summary and summary['cands']['rows']:
    C = pd.read_parquet(os.path.join(outd, f'cands_{sample}.parquet'), columns=['is_placeholder', 'label_unreadable', 'partner_kind', 'br_filled'])
    summary['cands_detail'] = dict(placeholder=int(C.is_placeholder.sum()), label_unreadable=int(C.label_unreadable.sum()), partner=C.partner_kind.value_counts().to_dict(), br_filled=int((C.br_filled == 1).sum()))
json.dump(summary, open(os.path.join(outd, f'merge_{sample}.json'), 'w'), indent=1); print(json.dumps(summary))
