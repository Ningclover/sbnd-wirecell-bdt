"""Round 0: reproduce the wcgpu small-sample study on aurora-r2 with its own recipe.
Label and pool as spec_v1 but with the wcgpu1 box (what wcgpu used); split even subrun = train, odd = test;
model A = XGBoost on the 258 shipped inputs with MicroBooNE's settings; metrics on the test half:
AUC on the pool, purity at 50 % and 40 % efficiency (efficiency against the true signal count), ceiling.
Usage: python train_round0.py <tables_dir> <out_dir> [seed]"""
import sys, os, json, time
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import spec_v1 as S
NAMES = [l.strip() for l in open(os.path.join(HERE, 'nue_l2_inputs_258.txt')) if l.strip()]
tdir, outd = sys.argv[1], sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 5
os.makedirs(outd, exist_ok=True)
C = pd.concat([pd.read_parquet(os.path.join(tdir, f'cands_{s}.parquet')) for s in ('nuecc', 'bnb')], ignore_index=True)
T = pd.concat([pd.read_parquet(os.path.join(tdir, f'truth_{s}.parquet')) for s in ('nuecc', 'bnb')], ignore_index=True)
fv = S.in_wcgpu1_box   # round 0 reproduces wcgpu: its box
lu = np.zeros(len(C), bool)
C['in_pool'] = (C.br_filled == 1) & fv(C.reco_x, C.reco_y, C.reco_z)
vtx_ok = C.dist_sce.notna() & (C.dist_sce < S.VTX_CUT_CM)
C['is_signal'] = (C.partner == 'neutrino') & vtx_ok & (C.m_pdg.abs() == 12) & (C.m_ccnc == 0) & fv(C.m_vx, C.m_vy, C.m_vz) & (C.m_edep_mev > S.EDEP_MIN_MEV)
C['split'] = np.where(C.subrun % 2 == 0, 'train', 'test')
T['is_true_signal'] = (T.pdg.abs() == 12) & (T.ccnc == 0) & fv(T.vx, T.vy, T.vz) & (T.edep_mev > S.EDEP_MIN_MEV)
T['split'] = np.where(T.subrun % 2 == 0, 'train', 'test')
P = C[C.in_pool].copy()
X = {h: P[P.split == h][NAMES].to_numpy(np.float32) for h in ('train', 'test')}
y = {h: P[P.split == h].is_signal.to_numpy(int) for h in ('train', 'test')}
n_true = {h: int(T[(T.split == h) & T.is_true_signal].shape[0]) for h in ('train', 'test')}
print('pool rows train/test', len(y['train']), len(y['test']), 'signal', int(y['train'].sum()), int(y['test'].sum()), 'true signal', n_true)
# MicroBooNE / wcgpu settings, base_score fixed per spec_v1
params = dict(max_depth=30, n_estimators=600, learning_rate=0.1, objective='binary:logitraw', subsample=0.8, colsample_bytree=1,
              colsample_bylevel=0.3, colsample_bynode=0.3, gamma=0.1, random_state=seed, base_score=0.5, n_jobs=32, tree_method='hist')
t0 = time.time(); model = xgb.XGBClassifier(**params); model.fit(X['train'], y['train']); fit_s = time.time() - t0
model.save_model(os.path.join(outd, f'model_A_seed{seed}.json'))
te = P[P.split == 'test'].copy(); te['score_A'] = model.predict(X['test'], output_margin=True)

def metrics(frame, col, n_true_sig):
    """AUC on the pool; purity at the cut where selected signal / true signal = eff; ties at the sentinel collapse."""
    auc = roc_auc_score(frame.is_signal, frame[col])
    f = frame.sort_values(col, ascending=False); cs = np.cumsum(f.is_signal.to_numpy()); n = np.arange(1, len(f) + 1)
    out = dict(auc=float(auc))
    for eff in (0.5, 0.4):
        k = int(np.searchsorted(cs, eff * n_true_sig)); k = min(k, len(f) - 1)
        out[f'purity_at_{int(eff*100)}'] = float(cs[k] / n[k]); out[f'cut_at_{int(eff*100)}'] = float(f[col].iloc[k]); out[f'selected_at_{int(eff*100)}'] = int(n[k])
    out['ceiling'] = float(frame.is_signal.sum() / n_true_sig)
    return out
res = dict(seed=seed, fit_seconds=round(fit_s, 1), pool_rows=dict(train=len(y['train']), test=len(y['test'])), signal=dict(train=int(y['train'].sum()), test=int(y['test'].sum())),
           background=dict(train=int(len(y['train']) - y['train'].sum()), test=int(len(y['test']) - y['test'].sum())), true_signal=n_true,
           shipped=metrics(te, 'nue_score', n_true['test']), model_A=metrics(te, 'score_A', n_true['test']),
           wcgpu_reference=dict(shipped=dict(auc=0.9160, purity_at_50=0.9538, purity_at_40=0.9840), model_A=dict(auc=0.9479, purity_at_50=0.9691, purity_at_40=0.9833), ceiling=0.605,
                                pool=dict(train=2662, test=2722), signal=dict(train=1958, test=2047), background=dict(train=704, test=675)))
# acceptance test as wcgpu defined it: gate on the gains, not on absolute values
ref = res['wcgpu_reference']; gA, gS = res['model_A'], res['shipped']
res['acceptance'] = dict(auc_gain=round(gA['auc'] - gS['auc'], 4), auc_gain_ref=round(ref['model_A']['auc'] - ref['shipped']['auc'], 4), auc_gain_window=[0.032 - 0.014, 0.032 + 0.014],
                         ceiling=round(gA['ceiling'], 4), ceiling_window=[0.605 - 0.024, 0.605 + 0.024], purity_gain_pts=round(100 * (gA['purity_at_50'] - gS['purity_at_50']), 2))
acc = res['acceptance']; acc['pass'] = bool(acc['auc_gain_window'][0] <= acc['auc_gain'] <= acc['auc_gain_window'][1] and acc['ceiling_window'][0] <= acc['ceiling'] <= acc['ceiling_window'][1] and acc['purity_gain_pts'] > 0)
json.dump(res, open(os.path.join(outd, f'round0_seed{seed}.json'), 'w'), indent=1); print(json.dumps(res, indent=1))
