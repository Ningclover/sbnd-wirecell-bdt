"""Freeze a trained νe model (level 1 + level 2) into a self-contained release folder.

Each xgboost model is cut to the trees training kept (early stopping's best iteration; the saved files also hold the
patience trees), checked to give identical raw margins on real rows of the tables, and saved as plain xgboost JSON.
A manifest records, per model, the file, its sha256, the input names in order, how the per-object scores are pooled,
the pre-cut, what an empty list gives, and the training settings; release/score_release.py scores tables from it alone.

  python release/freeze_release.py --l1-dir L1_DIR --l2 L2.json [--l2-alt NAME=L2.json ...] --tables MERGED_DIR \
         --sample S --out OUT_DIR [--name NAME] [--operating-cut X] [--notes notes.json] [--l1-chain-transform TABLE]

--l1-chain-transform (chain release, issue #22): the final model was trained on the level-1 values the Wire-Cell chain stores
with the TMVA-converted small models (training/round2/train_l2.py --l1-chain-transform); the table (scale, offset, empty
constant per model, written by wcgpu's l1_chain_transform.py for these exact small-model files) goes into the manifest.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, pandas as pd, xgboost as xgb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'spec'))
from l1_models import L1_MODELS, NUE_L1
KEY = ['file_id', 'run', 'subrun', 'event', 'cand']
NAMES = [l.strip() for l in open(os.path.join(HERE, '..', 'spec', 'nue_l2_inputs_258.txt')) if l.strip()]
TWO_LEVEL = [next(('l1_' + k for k in NUE_L1 if L1_MODELS[k]['score'] == n), n) for n in NAMES]

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def cut_to_best(path):
    b = xgb.Booster(); b.load_model(path); bi = b.attr('best_iteration')
    n = int(bi) + 1 if bi is not None else b.num_boosted_rounds()
    c = b[0:n]; c.feature_names = b.feature_names; c.feature_types = b.feature_types
    c.set_attr(best_iteration=None, best_score=None)
    return b, c, n

def check_same(full, cut, n, X, names):
    d = xgb.DMatrix(X, feature_names=names, missing=np.nan)
    a = full.predict(d, output_margin=True, iteration_range=(0, n)); c = cut.predict(d, output_margin=True)
    return float(np.max(np.abs(a - c))) if len(a) else 0.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--l1-dir', required=True); ap.add_argument('--l2', required=True); ap.add_argument('--l2-alt', nargs='*', default=[])
    ap.add_argument('--tables', required=True); ap.add_argument('--sample', required=True); ap.add_argument('--out', required=True); ap.add_argument('--name', default='nue_bdt')
    ap.add_argument('--operating-cut', type=float); ap.add_argument('--notes'); ap.add_argument('--check-rows', type=int, default=20000); ap.add_argument('--l1-chain-transform')
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, 'l1'), exist_ok=True); os.makedirs(os.path.join(a.out, 'l2'), exist_ok=True)
    rec = json.load(open(os.path.join(a.l1_dir, 'train_l1.json')))
    man = dict(name=a.name, frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), xgboost=xgb.__version__,
               output='raw margin (xgboost output_margin, base_score 0.5): higher is more nu_e-like; the level-1 margins are the level-2 inputs',
               level1=dict(label_rule=rec['label'], pooling='minimum over the candidate\'s objects of the raw margin',
                           empty='NaN (missing) when the candidate has no object of the model or fails its pre-cut; NOT the shipped default value',
                           models={}),
               level2={})
    rng = np.random.default_rng(5)
    for m in NUE_L1:
        md = L1_MODELS[m]; lr = rec['models'][m]['kept_lr']; cols = md['lists'] + md['scalars']
        full, cut, n = cut_to_best(os.path.join(a.l1_dir, 'l1_%s_lr%s.json' % (m, lr)))
        assert list(cut.feature_names) == cols, (m, cut.feature_names, cols)
        O = pd.read_parquet(os.path.join(a.tables, 'objs_%s_%s.parquet' % (m, a.sample)), columns=cols)
        if len(O) > a.check_rows: O = O.iloc[rng.choice(len(O), a.check_rows, replace=False)]
        diff = check_same(full, cut, n, O.to_numpy(np.float32), cols)
        f = os.path.join(a.out, 'l1', '%s.json' % m); cut.save_model(f)
        man['level1']['models'][m] = dict(file='l1/%s.json' % m, sha256=sha256(f), inputs=cols, inputs_per_object=md['lists'], inputs_per_candidate=md['scalars'],
                                          pre_cut=md['pre_cut'] or None, trees=n, learning_rate=lr, depth=rec['depth'], shipped_score_name=md['score'], level2_input_name='l1_' + m,
                                          check=dict(rows=int(len(O)), max_abs_diff_vs_training_file=diff))
        print('%-6s trees %4d  check on %6d rows: max |diff| %.3g' % (m, n, len(O), diff), flush=True)
    C = pd.read_parquet(os.path.join(a.tables, 'cands_%s.parquet' % a.sample))
    C = C[(C.is_placeholder == 0) & (C.br_filled == 1)]
    for tag, path in [('main', a.l2)] + [tuple(x.split('=', 1)) for x in a.l2_alt]:
        full, cut, n = cut_to_best(path); names = list(cut.feature_names)
        X = C.assign(**{'l1_' + m: np.float32(0.0) for m in NUE_L1})  # level-1 columns filled with zeros only for the identity check of the cut
        S = X.iloc[rng.choice(len(X), min(a.check_rows, len(X)), replace=False)] if len(X) > a.check_rows else X
        diff = check_same(full, cut, n, S[names].to_numpy(np.float32), names)
        fn = 'l2/nue_l2%s.json' % ('' if tag == 'main' else '_' + tag); f = os.path.join(a.out, fn); cut.save_model(f)
        shipped_map = {('l1_' + m): L1_MODELS[m]['score'] for m in NUE_L1}
        man['level2'][tag] = dict(file=fn, sha256=sha256(f), trees=n, inputs=names,
                                  inputs_note='in this order; l1_<model> is the retrained small model\'s pooled margin and replaces the shipped <model>_score; every other name is the tagger variable of the same name',
                                  replaces_shipped_inputs=shipped_map, left_out=[x for x in TWO_LEVEL if x not in names],
                                  check=dict(rows=int(len(S)), max_abs_diff_vs_training_file=diff))
        print('level 2 %-8s trees %4d inputs %d  check on %d rows: max |diff| %.3g' % (tag, n, len(names), len(S), diff), flush=True)
    if a.l1_chain_transform:
        T = json.load(open(a.l1_chain_transform))
        for m, md in man['level1']['models'].items():
            assert T['models'][m]['sha256'] == md['sha256'], ('transform table made for another file', m)
        man['level1']['empty'] = 'the model\'s empty constant (empty_value below) when the candidate has no object of the model or fails its pre-cut: the value the toolkit substitutes, which no real object reaches'
        man['level1']['chain_transform'] = dict(rule='level-2 input l1_<model> = float32(tanh(scale * margin + offset)), margin = minimum raw margin over the objects; empty list or failed pre-cut -> empty_value. This is the value the Wire-Cell chain stores in <model>_score with the TMVA-converted small model',
                                                source=os.path.basename(a.l1_chain_transform),
                                                models={m: {k: T['models'][m][k] for k in ('scale', 'offset', 'empty_value', 'value_min', 'value_max')} for m in man['level1']['models']})
        man['output'] = 'raw margin (xgboost output_margin, base_score 0.5) of the final model: higher is more nu_e-like; the final model reads the level-1 values as the chain stores them (level1.chain_transform)'
    if a.operating_cut is not None: man['operating_cut'] = a.operating_cut
    if a.notes: man['notes'] = json.load(open(a.notes))
    json.dump(man, open(os.path.join(a.out, 'manifest.json'), 'w'), indent=1)
    print('wrote', os.path.join(a.out, 'manifest.json'))

if __name__ == '__main__':
    main()
