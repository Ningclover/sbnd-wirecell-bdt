#!/usr/bin/env python3
"""spec_v1 section 8, the shared-file check: compare two parquet tables built from the same files on two servers.
Rows are aligned on the key columns; every common column is compared, integers and strings exactly, floats within --atol
(NaN equal to NaN). Prints the columns that differ and exits 1 if any does.
  python compare_tables.py A.parquet B.parquet [--atol 1e-6] [--key sample,file_id,run,subrun,event,cand]"""
import argparse, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument('a'); ap.add_argument('b'); ap.add_argument('--atol', type=float, default=1e-6)
ap.add_argument('--key', default=None, help='comma-separated key columns; default: the KEY columns present in both tables')
args = ap.parse_args()
A, B = pd.read_parquet(args.a), pd.read_parquet(args.b)
key = args.key.split(',') if args.key else [k for k in ['sample', 'file_id', 'run', 'subrun', 'event', 'cand', 'nu_idx', 'list_name', 'index'] if k in A.columns and k in B.columns]
common = [c for c in A.columns if c in B.columns]; only_a = [c for c in A.columns if c not in B.columns]; only_b = [c for c in B.columns if c not in A.columns]
for k in key:   # a key column that is numeric on both sides is compared as integers (older tables carried float keys)
    if pd.api.types.is_numeric_dtype(A[k]) and pd.api.types.is_numeric_dtype(B[k]):
        A[k] = A[k].astype('int64'); B[k] = B[k].astype('int64')
A = A.sort_values(key).reset_index(drop=True); B = B.sort_values(key).reset_index(drop=True)
print(f'rows {len(A)} vs {len(B)}; columns common {len(common)}, only in A {len(only_a)}, only in B {len(only_b)}; key {key}')
if len(A) != len(B) or not A[key].equals(B[key]):
    print('KEY MISMATCH: the two tables do not hold the same rows'); sys.exit(1)
bad = []
for c in common:
    x, y = A[c], B[c]
    if pd.api.types.is_float_dtype(x) or pd.api.types.is_float_dtype(y):
        xv, yv = x.to_numpy(float), y.to_numpy(float); eq = (np.isnan(xv) & np.isnan(yv)) | (np.abs(xv - yv) <= args.atol)
    else:
        eq = (x.astype(str) == y.astype(str)).to_numpy()
    n = int((~eq).sum())
    if n: bad.append((c, n))
for c, n in bad: print(f'DIFF {c}: {n} rows')
print('IDENTICAL' if not bad else f'{len(bad)} columns differ'); sys.exit(1 if bad else 0)
