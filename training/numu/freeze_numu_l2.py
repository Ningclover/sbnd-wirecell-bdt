"""Freeze νμ final models (round 1: final model only, shipped small-model scores) for evaluation at gpvm.

Each model is cut to the trees early stopping kept (the saved file holds the patience trees too), checked to give identical
margins on the dev rows, and saved as plain xgboost JSON with a manifest: input names in order (the chain's 74), the NaN->0
inputs, how to turn the margin into the numu_score scale, the training record and the Aurora dev cut at the purity anchor.

  python training/numu/freeze_numu_l2.py ROUND_DIR OUT_DIR TAG [TAG ...] [--tables MERGED_DIR]
"""
import argparse, hashlib, json, os, sys
import numpy as np, pandas as pd, xgboost as xgb
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import train_numu_l2 as T
import train_numu_l1 as T_L1

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('round_dir'); ap.add_argument('out_dir'); ap.add_argument('tags', nargs='+'); ap.add_argument('--name', default='r1'); ap.add_argument('--l1-dir', help='round 2: folder of train_numu_l1.py (l1_<model>.json + l1_chain_transform.json), copied into the release')
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    R = json.load(open(os.path.join(a.round_dir, 'train_numu_l2.json'))); D = pd.read_parquet(os.path.join(a.round_dir, 'dev_scores.parquet'))
    man = dict(kind='numu final model (level 2), %s' % a.name, level1=None, inputs=T.NAMES, nan_to_0=T.NAN_TO_0,
               score='xgboost margin (output_margin=True); numu_score scale = margin * 2/ln(10) = margin * %.6f' % T.TO_SCORE,
               spec=R['spec'], purity_anchor=R['purity_anchor'], weights=R['weights'], nc_only=R['nc_only'], xgboost=xgb.__version__,
               shipped_dev=R['shipped_dev']['at_purity'], models={})
    for tag in a.tags:
        f = R['fits'][tag]; full = xgb.Booster(); full.load_model(os.path.join(a.round_dir, 'l2_numu_%s.json' % tag))
        cut = full[:f['trees']]; p = os.path.join(a.out_dir, 'numu_l2_%s_%s.json' % (a.name, tag)); cut.save_model(p)
        chk = xgb.Booster(); chk.load_model(p)
        # identical margins on a dev-like matrix: compare the cut model to the full one with iteration_range, on random rows
        X = np.random.default_rng(0).normal(size=(2000, len(T.NAMES))).astype(np.float32)
        dm = xgb.DMatrix(X, feature_names=T.NAMES)
        d = np.max(np.abs(chk.predict(dm, output_margin=True) - full.predict(dm, iteration_range=(0, f['trees']), output_margin=True)))
        if d != 0: raise SystemExit('%s: frozen model differs by %g' % (tag, d))
        man['models'][tag] = dict(file=os.path.basename(p), sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest(), trees=f['trees'],
                                  depth=f['depth'], floor=f['floor'], dev=f['dev'], max_abs_diff_vs_unfrozen=float(d))
        print('%s: %d trees, frozen file identical (max |diff| %g), dev efficiency at %.0f %% purity %.4f' % (tag, f['trees'], d, 100 * R['purity_anchor'], f['dev']['at_purity']['eff']))
    if a.l1_dir:
        import shutil
        TRF = json.load(open(os.path.join(a.l1_dir, 'l1_chain_transform.json')))
        for m, t in TRF['models'].items():
            shutil.copy(os.path.join(a.l1_dir, t['file']), os.path.join(a.out_dir, t['file']))
            t['sha256'] = hashlib.sha256(open(os.path.join(a.out_dir, t['file']), 'rb').read()).hexdigest()
            t['inputs'] = T_L1.L1_MODELS[m]['lists']; t['protect'] = {k: v for k, v in T_L1.PROTECT.items() if k in t['inputs']}
        man['level1'] = dict(TRF, rule=json.load(open(os.path.join(a.l1_dir, 'train_numu_l1.json')))['label'],
                             note='value = float32(tanh(scale * pooled margin + offset)); empty_value when the candidate has no object; the chain runs these on every candidate (no gate). The value replaces <model>_score in the final model inputs.')
    json.dump(man, open(os.path.join(a.out_dir, 'manifest.json'), 'w'), indent=1)

if __name__ == '__main__': main()
