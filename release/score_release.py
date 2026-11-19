"""Score candidates with a frozen νe release folder (release/freeze_release.py), from its manifest alone.

Reads the builder's tables (cands_<sample>.parquet and objs_<model>_<sample>.parquet), computes the 15 level-1 pooled margins
exactly as in training (minimum over the candidate's objects; NaN when the candidate has no object or fails the pre-cut),
then the level-2 raw margin. A release with level1.chain_transform in its manifest (the chain release, issue #22) feeds the
final model the value the Wire-Cell chain stores instead: float32(tanh(scale * margin + offset)), the model's empty constant
where the margin is NaN; the l1 columns written are then those values (the margins stay in margin_l1_<model>). Candidates are those with is_placeholder == 0 and br_filled == 1 (the tagger ran).

  python release/score_release.py RELEASE_DIR TABLE_DIR SAMPLE [SAMPLE ...] [--variant main] [--file-ids ids.txt]
         [--column score] [--out scores.parquet] [--nthread 16]
"""
import argparse, json, os
import numpy as np, pandas as pd, xgboost as xgb
KEY = ['file_id', 'run', 'subrun', 'event', 'cand']

def booster(path, nthread):
    b = xgb.Booster(); b.load_model(path); b.set_param({'nthread': nthread}); return b

def score(rel, man, table_dir, sample, variant='main', file_ids=None, nthread=16):
    l2m = man['level2'][variant]; l2 = booster(os.path.join(rel, l2m['file']), nthread)
    need = list(dict.fromkeys(KEY + ['is_placeholder', 'br_filled', 'pio_filled', 'pio_flag_pio', 'cosmict_flag_1'] + [n for n in l2m['inputs'] if not n.startswith('l1_')]))
    C = pd.read_parquet(os.path.join(table_dir, 'cands_%s.parquet' % sample), columns=need)
    C = C[(C.is_placeholder == 0) & (C.br_filled == 1)]
    if file_ids is not None: C = C[C.file_id.astype(str).isin(file_ids)]
    C = C.copy()
    for m, md in man['level1']['models'].items():
        keep = C[C.eval(md['pre_cut']).to_numpy(bool)][KEY] if md['pre_cut'] else C[KEY]
        O = pd.read_parquet(os.path.join(table_dir, 'objs_%s_%s.parquet' % (m, sample)), columns=KEY + md['inputs']).merge(keep, on=KEY, how='inner')
        col = md['level2_input_name']
        if len(O) == 0: C[col] = np.nan; continue
        O['p'] = booster(os.path.join(rel, md['file']), nthread).predict(xgb.DMatrix(O[md['inputs']].to_numpy(np.float32), feature_names=md['inputs'], missing=np.nan), output_margin=True)
        C = C.merge(O.groupby(KEY, sort=False).p.min().rename(col).reset_index(), on=KEY, how='left')
    T = man['level1'].get('chain_transform')
    if T:
        for m, md in man['level1']['models'].items():
            col = md['level2_input_name']; t = T['models'][m]; x = C[col].to_numpy(np.float64); C['margin_' + col] = x
            C[col] = np.where(np.isnan(x), np.float32(t['empty_value']), np.tanh(t['scale'] * x + t['offset']).astype(np.float32)).astype(np.float32)
    C['score'] = l2.predict(xgb.DMatrix(C[l2m['inputs']].to_numpy(np.float32), feature_names=l2m['inputs'], missing=np.nan), output_margin=True)
    return C

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('release'); ap.add_argument('table_dir'); ap.add_argument('samples', nargs='+')
    ap.add_argument('--variant', default='main'); ap.add_argument('--file-ids'); ap.add_argument('--column', default='score'); ap.add_argument('--out', default='scores_release.parquet'); ap.add_argument('--nthread', type=int, default=16)
    a = ap.parse_args(); man = json.load(open(os.path.join(a.release, 'manifest.json')))
    ids = set(l.strip() for l in open(a.file_ids) if l.strip()) if a.file_ids else None
    out = []
    for s in a.samples:
        C = score(a.release, man, a.table_dir, s, a.variant, ids, a.nthread)
        l1 = [md['level2_input_name'] for md in man['level1']['models'].values()]
        out.append(C[KEY + l1 + ['score']].rename(columns={'score': a.column}).assign(sample=s)); print('%s: %d candidates scored' % (s, len(C)), flush=True)
    pd.concat(out, ignore_index=True).to_parquet(a.out); print('wrote', a.out)

if __name__ == '__main__':
    main()
