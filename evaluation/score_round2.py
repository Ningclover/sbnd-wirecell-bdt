"""Score candidates with a round 2 model: the two-stage tagger (15 retrained level-1 models on the object tables, pooled by
minimum per candidate, then the final model) or the level-2-only variant (shipped level-1 scores, new final model). The
same code runs on Aurora and at Fermilab; gpvm attaches the output to its tables and runs eval_standard.py --score on it.

  python evaluation/score_round2.py TABLE_DIR MODEL_DIR --l2 l2_two-level_d10_f0.5.json --variant two-level
         --samples S1 S2 ... [--column score_r2] [--out scores.parquet] [--nthread 16]

MODEL_DIR holds the level-1 models as stage 1 wrote them (l1_<model>_lr<lr>.json, with train_l1.json naming the kept
learning rate per model) and the level-2 model file. Rows scored: candidates that are not placeholders and have
br_filled == 1 (the tagger ran); every other candidate gets NaN, which fails any cut, as the shipped sentinel -15 does.
Level-1 pooling: the model's pre-cut (spec/l1_models.py) on the candidate, then the minimum of the model's raw margin over
the candidate's objects; NaN where the candidate has no object or fails the pre-cut (xgboost missing value in level 2).
Output: one parquet with the key columns and the score column (the raw margin of the final model, the shipped scale).
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, xgboost as xgb
HERE = os.path.dirname(os.path.abspath(__file__)); SPEC = os.path.join(HERE, '..', 'spec'); sys.path.insert(0, SPEC)
import spec_v1

def load_booster(path, nthread):
    """A saved booster holds every tree trained, the best early-stopping round included; predict with the trees up to the
    best iteration, as training did (xgboost stores best_iteration as an attribute)."""
    b = xgb.Booster(); b.load_model(path); b.set_param({'nthread': nthread}); bi = b.attr('best_iteration')
    return b, ((0, int(bi) + 1) if bi is not None else None)
from l1_models import L1_MODELS, NUE_L1
KEY = ['file_id', 'run', 'subrun', 'event', 'cand']
NAMES = [l.strip() for l in open(os.path.join(SPEC, 'nue_l2_inputs_258.txt')) if l.strip()]
SHIPPED = [L1_MODELS[m]['score'] for m in NUE_L1]

def level1_scores(table_dir, sample, C, model_dir, kept, nthread):
    """C: the candidates to score (key + pre-cut columns). Returns C's key + l1_<model> columns."""
    out = C[KEY].copy()
    for m in NUE_L1:
        md = L1_MODELS[m]; cols = md['lists'] + md['scalars']
        keep = C[C.eval(md['pre_cut']).to_numpy(bool)][KEY] if md['pre_cut'] else C[KEY]
        O = pd.read_parquet(os.path.join(table_dir, 'objs_%s_%s.parquet' % (m, sample)), columns=KEY + cols).merge(keep, on=KEY, how='inner')
        col = 'l1_' + m
        if len(O) == 0: out[col] = np.nan; continue
        bst, ir = load_booster(os.path.join(model_dir, 'l1_%s_lr%g.json' % (m, kept[m])), nthread)
        O['p'] = bst.predict(xgb.DMatrix(O[cols].to_numpy(np.float32), feature_names=cols), output_margin=True, iteration_range=ir)
        out = out.merge(O.groupby(KEY, sort=False).p.min().rename(col).reset_index(), on=KEY, how='left')
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('table_dir'); ap.add_argument('model_dir'); ap.add_argument('--l2', required=True); ap.add_argument('--variant', choices=['two-level', 'level2-only'], required=True)
    ap.add_argument('--samples', nargs='+', required=True); ap.add_argument('--column', default='score_r2'); ap.add_argument('--out', default='scores_round2.parquet'); ap.add_argument('--nthread', type=int, default=16)
    a = ap.parse_args()
    names = [('l1_' + NUE_L1[SHIPPED.index(n)] if n in SHIPPED else n) for n in NAMES] if a.variant == 'two-level' else list(NAMES)
    kept = {m: float(r['kept_lr']) for m, r in json.load(open(os.path.join(a.model_dir, 'train_l1.json')))['models'].items()} if a.variant == 'two-level' else {}
    l2, ir2 = load_booster(os.path.join(a.model_dir, a.l2), a.nthread)
    if l2.feature_names:   # the inputs the model was trained on (a model may leave some out, train_l2.py --drop-inputs)
        if set(l2.feature_names) - set(names): raise SystemExit('model inputs not in the %s input list: %s' % (a.variant, sorted(set(l2.feature_names) - set(names))))
        names = list(l2.feature_names)
    need = list(dict.fromkeys(KEY + ['br_filled', 'is_placeholder', 'pio_filled', 'pio_flag_pio', 'cosmict_flag_1'] + NAMES))
    out = []
    for s in a.samples:
        C = pd.read_parquet(os.path.join(a.table_dir, 'cands_%s.parquet' % s), columns=need)
        C = C[(C.is_placeholder == 0) & (C.br_filled == 1)].copy()
        if a.variant == 'two-level': C = C.merge(level1_scores(a.table_dir, s, C, a.model_dir, kept, a.nthread), on=KEY, how='left')
        C[a.column] = l2.predict(xgb.DMatrix(C[names].to_numpy(np.float32), feature_names=names, missing=np.nan), output_margin=True, iteration_range=ir2)
        print('%s: %d candidates scored' % (s, len(C)), flush=True); out.append(C[KEY + [a.column]])
    R = pd.concat(out, ignore_index=True); R.to_parquet(a.out); print('wrote', a.out, len(R), 'rows')

if __name__ == '__main__': main()
