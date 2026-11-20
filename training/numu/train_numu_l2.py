"""νμ round 1: retrain the final νμ CC model only, on the 74 inputs the chain binds by position
(spec/numu_l2_inputs_74.txt, wcgpu #25), with the shipped small-model scores (numu_1, numu_2, cosmict_10) as stored by the
chain. Chain-ready as is: no level-1 transform needed.

Rows (numu/docs/spec_numu_draft.md, issue #27): every real candidate (is_placeholder == 0, label_unreadable == 0) with the
reco vertex in the spec_v1 FV; no br_filled filter. For an NC-only sample (--nc-only, the NC Δ-radiative ncdr) only
candidates whose partner is an NC neutrino are kept.
Label (user 2026-11-19, #26, Matteo's matching): partner is a neutrino with iou_n > 0 (best T_match point IoU, no vertex
cut), |pdg| == 14, CC, true vertex in the FV, deposited energy > 100 MeV.
Inputs: NaN -> 0 for the four the chain protects (cosmict_4_angle_beam, cosmict_7_angle_beam, cosmict_7_theta, cosmict_7_phi);
any other NaN is reported and left missing.
Split: spec_v1.split_v11 (train / dev). Weights: exposure at the BNB POT (--weights); optional signal-share floor (--floor,
0 = none).

Learner: xgboost hist, binary:logistic, saved margin. numu_score scale: numu_score = log10((1+v)/(1-v)), v = tanh(margin)
= margin * 2/ln10; the dev report quotes cuts on that scale.

Operating point (user 2026-11-20): weighted purity 95 % on the dev slice. For each model (and the shipped numu_score) the cut
is the lowest one whose weighted purity of everything above it is >= --purity; weighted efficiency there is reported, with
its binomial error from the effective number of signal rows, plus the efficiency per true-energy bin and the paired counts
(signal passing one model and not the other, unweighted).

  python training/numu/train_numu_l2.py MERGED_DIR OUT_DIR --samples S ... --weights S=w ... [--nc-only S ...]
         [--depth 6 8 12] [--floor 0 0.5] [--lr 0.1] [--max-trees 3000] [--patience 50] [--nthread 8] [--purity 0.95]
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); SPEC = os.path.join(HERE, '..', '..', 'spec'); sys.path.insert(0, SPEC)
import spec_v1
KEY = ['file_id', 'run', 'subrun', 'event', 'cand']
NAMES = [l.strip() for l in open(os.path.join(SPEC, 'numu_l2_inputs_74.txt')) if l.strip()]
NAN_TO_0 = ['cosmict_4_angle_beam', 'cosmict_7_angle_beam', 'cosmict_7_theta', 'cosmict_7_phi']
EDEP_MIN_NUMU = 100.0
TO_SCORE = 2.0 / np.log(10.0)
L1_SLOTS = ['numu_1', 'numu_2', 'cosmict_10']   # small models whose score is a final-model input
E_BINS = [0, 250, 500, 750, 1000, 1500, 2000, 3000, 1e9]   # true neutrino energy, MeV

def numu_signal(C):
    nu = (C.partner_kind == 'neutrino') & (C.iou_n > 0)
    return nu & (C.m_pdg.abs() == 14) & (C.m_ccnc == 0) & spec_v1.in_fv(C.m_vx, C.m_vy, C.m_vz) & (C.m_edep_mev > EDEP_MIN_NUMU)

def load(d, samples, W, nc_only):
    need = KEY + ['is_placeholder', 'label_unreadable', 'reco_x', 'reco_y', 'reco_z', 'partner_kind', 'iou_n', 'm_pdg', 'm_ccnc', 'm_vx', 'm_vy', 'm_vz', 'm_edep_mev', 'm_E', 'numu_score'] + NAMES
    out = []
    for s in samples:
        C = pd.read_parquet(os.path.join(d, 'cands_%s.parquet' % s), columns=list(dict.fromkeys(need)))
        n0 = len(C); C = C[(C.is_placeholder == 0) & (C.label_unreadable == 0)]
        C = C[spec_v1.in_fv(C.reco_x, C.reco_y, C.reco_z)].copy()
        if s in nc_only:
            n1 = len(C); C = C[(C.partner_kind == 'neutrino') & (C.iou_n > 0) & (C.m_ccnc == 1)].copy()
            print('%s: NC-partner rows only, %d of %d pool rows kept' % (s, len(C), n1), flush=True)
        C['y'] = numu_signal(C).astype(int).to_numpy()
        C['split'] = [str(x) for x in spec_v1.split_v11(np.full(len(C), s, dtype=object), C.run.to_numpy(), C.subrun.to_numpy())]
        C['w_exp'] = W[s]; C['sample'] = s; out.append(C)
        print('%s: %d candidates, pool %d (%d signal), weight %g' % (s, n0, len(C), C.y.sum(), W[s]), flush=True)
    C = pd.concat(out, ignore_index=True)
    for n in NAN_TO_0: C[n] = C[n].fillna(0.0)
    nan = {n: int(C[n].isna().sum()) for n in NAMES if C[n].isna().any()}
    print('NaN left in inputs: %s' % (nan or 'none'), flush=True)
    return C, nan

def floored(w, y, floor):
    w = w.astype(float).copy()
    if not floor: return w
    s = y == 1; tot = w.sum(); w[s] *= floor * tot / w[s].sum(); w[~s] *= (1 - floor) * tot / w[~s].sum(); return w

def at_purity(score, y, w, purity):
    """Lowest cut whose weighted purity above it is >= purity. Returns cut, eff, eff_err, pur, selection mask."""
    o = np.argsort(-score); ws = (w * (y == 1))[o]; cs, cw = np.cumsum(ws), np.cumsum(w[o]); pur = cs / cw
    ok = np.nonzero(pur >= purity)[0]
    if len(ok) == 0: return None
    k = ok[-1]; cut = float(score[o][k]); sel = score >= cut
    tot = ws.sum(); eff = float(cs[k] / tot)
    n_eff = tot ** 2 / np.sum((w * (y == 1)) ** 2)   # effective number of signal rows
    return dict(cut=cut, cut_numu_score_scale=None, eff=eff, eff_err=float(np.sqrt(eff * (1 - eff) / n_eff)), purity=float(pur[k]),
                passing_weighted_signal=float(cs[k]), passing_weighted_background=float(cw[k] - cs[k])), sel

def report(score, D, purity, scale):
    y, w = D.y.to_numpy(), D.w_exp.to_numpy()
    r = dict(auc=float(roc_auc_score(y, score)), weighted_auc=float(roc_auc_score(y, score, sample_weight=w)))
    a = at_purity(score, y, w, purity)
    if a is None: r['at_purity'] = None; return r, None
    op, sel = a; op['cut_numu_score_scale'] = op['cut'] * scale
    s = y == 1; e = D.m_E.to_numpy() * 1000.0   # m_E in GeV
    op['eff_by_true_E'] = []
    for lo, hi in zip(E_BINS[:-1], E_BINS[1:]):
        b = s & (e >= lo) & (e < hi); ws = w[b]
        op['eff_by_true_E'].append(dict(lo=lo, hi=hi, n=int(b.sum()), eff=float(ws[sel[b]].sum() / ws.sum()) if b.any() else None))
    op['passing_background_by_sample'] = {k: float(w[sel & ~s & (D['sample'] == k).to_numpy()].sum()) for k in D['sample'].unique()}
    r['at_purity'] = op; return r, sel

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('merged_dir'); ap.add_argument('out_dir'); ap.add_argument('--samples', nargs='+', required=True)
    ap.add_argument('--weights', nargs='*', default=[]); ap.add_argument('--nc-only', nargs='*', default=[])
    ap.add_argument('--depth', nargs='*', type=int, default=[6, 8, 12]); ap.add_argument('--floor', nargs='*', type=float, default=[0.0, 0.5])
    ap.add_argument('--lr', type=float, default=0.1); ap.add_argument('--max-trees', type=int, default=3000); ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--nthread', type=int, default=8); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--purity', type=float, default=0.95)
    ap.add_argument('--l1', metavar='L1_VALUES', help='round 2: parquet from train_numu_l1.py; its l1_<model> chain values replace the shipped <model>_score inputs (same slots)')
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    W = {s: 1.0 for s in a.samples}
    for kv in a.weights: k, v = kv.split('='); W[k] = float(v)
    t0 = time.time(); C, nan = load(a.merged_dir, a.samples, W, a.nc_only)
    if a.l1:
        L = pd.read_parquet(a.l1); n0 = len(C); C = C.merge(L, on=['sample'] + KEY, how='left'); assert len(C) == n0
        for m in L1_SLOTS:
            if C['l1_' + m].isna().any(): raise SystemExit('l1_%s missing for %d pool rows' % (m, C['l1_' + m].isna().sum()))
            C[m + '_score'] = C.pop('l1_' + m).astype(np.float32)
        print('level-1 inputs replaced by the chain values of %s: %s' % (a.l1, ', '.join(m + '_score' for m in L1_SLOTS)), flush=True)
    tr, dv = C[C.split == 'train'], C[C.split == 'dev']
    print('pool train %d (%d signal) dev %d (%d signal); dev weighted signal %.0f background %.0f; %.0f s' % (
        len(tr), tr.y.sum(), len(dv), dv.y.sum(), dv.w_exp[dv.y == 1].sum(), dv.w_exp[dv.y == 0].sum(), time.time() - t0), flush=True)
    res = dict(l1=a.l1, spec='numu/docs/spec_numu_draft.md', purity_anchor=a.purity, weights=W, nc_only=a.nc_only, nan_left=nan, lr=a.lr, seed=a.seed, xgboost=xgb.__version__,
               rows=dict(train=dict(n=int(len(tr)), s=int(tr.y.sum())), dev=dict(n=int(len(dv)), s=int(dv.y.sum()))), fits={})
    shipped, sel0 = report(dv.numu_score.to_numpy(), dv, a.purity, 1.0); res['shipped_dev'] = shipped
    op = shipped['at_purity']
    print('shipped numu_score on dev: AUC %.4f; at %.0f %% purity: cut %.3f, efficiency %.4f ± %.4f' % (shipped['auc'], 100 * a.purity, op['cut'], op['eff'], op['eff_err']), flush=True)
    dev_scores = dv[KEY + ['sample', 'y', 'w_exp', 'm_E', 'numu_score']].copy()
    Xtr, Xdv = tr[NAMES].to_numpy(np.float32), dv[NAMES].to_numpy(np.float32)
    for depth in a.depth:
        for floor in a.floor:
            tag = 'd%d_f%g' % (depth, floor); t1 = time.time()
            dtr = xgb.DMatrix(Xtr, label=tr.y.to_numpy(), weight=floored(tr.w_exp.to_numpy(), tr.y.to_numpy(), floor), feature_names=NAMES, missing=np.nan)
            ddv = xgb.DMatrix(Xdv, label=dv.y.to_numpy(), weight=floored(dv.w_exp.to_numpy(), dv.y.to_numpy(), floor), feature_names=NAMES, missing=np.nan)
            params = dict(max_depth=depth, eta=a.lr, objective='binary:logistic', eval_metric='logloss', tree_method='hist', nthread=a.nthread, seed=a.seed,
                          subsample=0.8, colsample_bylevel=0.5, colsample_bynode=0.5, gamma=0.2, base_score=0.5)
            ev = {}; bst = xgb.train(params, dtr, num_boost_round=a.max_trees, evals=[(dtr, 'train'), (ddv, 'dev')], early_stopping_rounds=a.patience, evals_result=ev, verbose_eval=False)
            nt = bst.best_iteration + 1; sc = bst.predict(ddv, iteration_range=(0, nt), output_margin=True)
            f = dict(depth=depth, floor=floor, trees=int(nt), train_logloss=float(ev['train']['logloss'][nt - 1]), dev_logloss=float(ev['dev']['logloss'][nt - 1]), fit_s=round(time.time() - t1, 1))
            f['dev'], sel = report(sc, dv, a.purity, TO_SCORE)
            if sel is not None:
                s = dv.y.to_numpy() == 1
                f['dev']['paired_signal'] = dict(new_only=int((sel & ~sel0 & s).sum()), shipped_only=int((~sel & sel0 & s).sum()))
            res['fits'][tag] = f; dev_scores[tag] = sc; bst.save_model(os.path.join(a.out_dir, 'l2_numu_%s.json' % tag))
            o = f['dev']['at_purity']
            print('%-8s trees %4d  logloss train %.4f dev %.4f  AUC %.4f  at %.0f %% purity: efficiency %.4f ± %.4f (shipped %.4f)  paired new-only %d shipped-only %d  (%.0f s)' % (
                tag, nt, f['train_logloss'], f['dev_logloss'], f['dev']['auc'], 100 * a.purity, o['eff'], o['eff_err'], op['eff'],
                f['dev']['paired_signal']['new_only'], f['dev']['paired_signal']['shipped_only'], f['fit_s']), flush=True)
            json.dump(res, open(os.path.join(a.out_dir, 'train_numu_l2.json'), 'w'), indent=1)
    dev_scores.to_parquet(os.path.join(a.out_dir, 'dev_scores.parquet')); res['wall_s'] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out_dir, 'train_numu_l2.json'), 'w'), indent=1); print('done in %.0f s' % res['wall_s'])

if __name__ == '__main__': main()
