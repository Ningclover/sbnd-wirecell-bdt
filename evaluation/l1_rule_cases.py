"""What each level-1 small model's hand-written rule does on real objects, and where it fails.

For every nu_e small model (spec/l1_models.py): all objects of the pool candidates (label B rows, train_l1.rows_for), with
the candidate's truth label (spec_v1.signal_mask) and exposure weight. Per model:
  - counts: objects and exposure-weighted sums per class; the share of each class on which the rule flag fired (nu_e
    models: flag == 0 is the fired, background-like value); fired signal objects are the rule's failures on signal.
  - one-variable separation of every input (weighted AUC against the class), to pick the two inputs to draw.
  - the retrained level-1 model (the kept learning rate of <l1-dir>/train_l1.json) scored on every object (raw margin);
    at the rule's weighted background-fired share, the model's cut with the same background share below it, and the
    signal share below that cut, next to the rule's signal share fired: the same background rejection, two signal losses.
  - a figure per model: the inputs' distributions per class (weighted, unit area; background where the rule fired dashed),
    the two best inputs against each other with fired objects marked, and the model score per class.
  - failure cases, written to JSON and Markdown: signal objects the rule fired on (highest deposited energy first),
    background objects the rule missed that the model scores lowest, and signal objects the model scores lowest.
Usage:
  python evaluation/l1_rule_cases.py --merged-dir DIR --l1-dir DIR --out DIR [--models m ...] [--samples ...]
         [--weights SAMPLE=W ...] [--nthread 16] [--n-cases 6] [--scatter-n 4000]
Writes <out>/l1_rule_cases.json, <out>/l1_rule_cases.md and <out>/l1_rule_<model>.png.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, xgboost as xgb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'spec')); sys.path.insert(0, os.path.join(HERE, '..', 'training', 'round2'))
from l1_models import L1_MODELS, NUE_L1
from train_l1 import load_cands, rows_for, KEY
from score_round2 import load_booster
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# The two inputs each hand-written cut turns on, and the thresholds it uses on them (docs/l1_rules_from_code.md in the
# project record, read from wire-cell-pid NeutrinoID_nue_tagger.h); drawn as the scatter and as dashed lines.
RULE_DRAW = {
 'br3_3': (('br3_3_v_angle', [150, 105]), ('br3_3_v_dir_length', [10])),
 'br3_5': (('br3_5_v_angle', [60]), ('br3_5_v_energy', [250])),
 'br3_6': (('br3_6_v_angle', [150]), ('br3_6_v_length', [7.5])),
 'pio_2': (('pio_2_v_angle2', [7.5]), ('pio_2_v_dis2', [36])),
 'stw_2': (('stw_2_v_angle', [15]), ('stw_2_v_dir_length', [40])),
 'stw_3': (('stw_3_v_angle', [15]), ('stw_3_v_dir_length', [40])),
 'stw_4': (('stw_4_v_angle', [30]), ('stw_4_v_dis', [3])),
 'sig_1': (('sig_1_v_angle', [30]), ('sig_1_v_energy', [250])),
 'sig_2': (('sig_2_v_medium_dQ_dx', [1.6, 2.0]), ('sig_2_v_start_dQ_dx', [2.5, 3.6])),
 'lol_1': (('lol_1_v_angle', [36]), ('lol_1_v_energy', [150])),
 'lol_2': (('lol_2_v_angle', [10, 17.5, 30]), ('lol_2_v_length', [7.5, 8, 30])),
 'tro_1': (('tro_1_v_min_dis', [10]), ('tro_1_v_sg1_length', [3.6])),
 'tro_2': (('tro_2_v_max_length', [18, 30, 40]), ('tro_2_v_stem_length', [])),
 'tro_4': (('tro_4_v_dir2_mag', [10, 12]), ('tro_4_v_angle', [15, 20, 30, 60])),
 'tro_5': (('tro_5_v_max_angle', [25]), ('tro_5_v_min_angle', [20])),
}

def wauc(x, y, w):
    """Weighted AUC of score x for class y (1 = signal) with weights w: P(x_sig > x_bkg) with ties counted half."""
    o = np.argsort(x, kind='stable'); x, y, w = x[o], y[o], w[o]
    ws, wb = w * (y == 1), w * (y == 0)
    # cumulative background weight strictly below each value, ties half
    vals, idx = np.unique(x, return_index=True)
    cb = np.add.reduceat(wb, idx); cs = np.add.reduceat(ws, idx)
    below = np.concatenate([[0.0], np.cumsum(cb)[:-1]])
    num = np.sum(cs * (below + 0.5 * cb)); den = ws.sum() * wb.sum()
    return float(num / den) if den > 0 else float('nan')

def share_below(x, w, cut, strict=True):
    m = (x < cut) if strict else (x <= cut)
    return float(w[m].sum() / w.sum()) if w.sum() > 0 else float('nan')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--merged-dir', required=True); ap.add_argument('--l1-dir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--models', nargs='*', default=NUE_L1); ap.add_argument('--samples', nargs='*', default=['aurora-r2-nuecc', 'aurora-r2-bnb', 'aurora-r2-ncdr', 'fg1-beam-off-train'])
    ap.add_argument('--weights', nargs='*', default=None, help='SAMPLE=W; default: the weights of <l1-dir>/train_l1.json')
    ap.add_argument('--nthread', type=int, default=16); ap.add_argument('--n-cases', type=int, default=6); ap.add_argument('--scatter-n', type=int, default=4000); ap.add_argument('--seed', type=int, default=5)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    rec = json.load(open(os.path.join(a.l1_dir, 'train_l1.json')))
    W = dict(rec['weights'])
    if a.weights:
        for kv in a.weights: k, v = kv.split('='); W[k] = float(v)
    C = load_cands(a.merged_dir, a.samples, W)
    E = pd.read_parquet(os.path.join(a.merged_dir, 'cands_%s.parquet' % a.samples[0]), columns=KEY + ['m_edep_mev'])
    E = pd.concat([E] + [pd.read_parquet(os.path.join(a.merged_dir, 'cands_%s.parquet' % s), columns=KEY + ['m_edep_mev']) for s in a.samples[1:]], ignore_index=True)
    print('candidates', len(C), 'signal', int(C.sig.sum()), 'load %.0f s' % (time.time() - t0), flush=True)
    out = dict(weights=W, samples=a.samples, l1_dir=os.path.basename(os.path.normpath(a.l1_dir)), models={}); md = []
    rng = np.random.default_rng(a.seed)
    for m in a.models:
        t1 = time.time(); spec = L1_MODELS[m]; ra = spec['rule_a']
        O, cols, keep = rows_for(C, a.merged_dir, m, a.samples, 'B')
        O = O.merge(E, on=KEY, how='left')
        y = O.sig.to_numpy(bool); w = O.w_exp.to_numpy(float); fired = O.flag.to_numpy() == ra['fired']
        X = O[cols].to_numpy(np.float32)
        # the retrained model on every object
        lr = rec['models'][m]['kept_lr']; b, it = load_booster(os.path.join(a.l1_dir, 'l1_%s_lr%s.json' % (m, lr)), a.nthread)
        s = np.empty(len(O), np.float32)
        for i in range(0, len(O), 2_000_000):
            d = xgb.DMatrix(X[i:i + 2_000_000], feature_names=cols); s[i:i + 2_000_000] = b.predict(d, output_margin=True, iteration_range=it) if it else b.predict(d, output_margin=True)
        O['bdt'] = s
        # counts and the rule's effect per class
        r = {}
        for cls, msk in (('signal', y), ('background', ~y)):
            r[cls] = dict(objects=int(msk.sum()), weighted=float(w[msk].sum()), fired=int((msk & fired).sum()),
                          fired_share=float(w[msk & fired].sum() / w[msk].sum()) if w[msk].sum() > 0 else float('nan'),
                          fired_share_unweighted=float((msk & fired).sum() / max(1, msk.sum())),
                          per_sample={smp: dict(objects=int((msk & (O['sample'].to_numpy() == smp)).sum()), fired=int((msk & fired & (O['sample'].to_numpy() == smp)).sum())) for smp in a.samples})
        # same background rejection, two signal losses
        fb = r['background']['fired_share']; bs = s[~y]; bw = w[~y]
        o = np.argsort(bs); cw = np.cumsum(bw[o]) / bw.sum(); k = int(np.searchsorted(cw, fb)); cut = float(bs[o][min(k, len(o) - 1)]) if len(o) else float('nan')
        sig_below = share_below(s[y], w[y], cut); bkg_below = share_below(s[~y], w[~y], cut)
        r['matched'] = dict(rule_background_fired=fb, rule_signal_fired=r['signal']['fired_share'], model_cut=cut, model_background_below=bkg_below, model_signal_below=sig_below)
        # one-variable separation
        r['input_auc'] = {c: wauc(X[:, j].astype(np.float64), y.astype(int), w) for j, c in enumerate(cols)}
        r['model_auc'] = wauc(s.astype(np.float64), y.astype(int), w)
        r['model_auc_unweighted'] = wauc(s.astype(np.float64), y.astype(int), np.ones(len(s)))
        sep = sorted(cols, key=lambda c: -abs(r['input_auc'][c] - 0.5)); r['drawn'] = sep[:2]
        # failure cases
        def cases(mask, order, n):
            idx = np.flatnonzero(mask); idx = idx[np.argsort(order[idx])][:n]
            rows = []
            for i in idx:
                row = {k: (int(O[k].iat[i]) if k != 'file_id' else str(O[k].iat[i])) for k in KEY}
                row.update(sample=str(O['sample'].iat[i]), object_index=int(O['index'].iat[i]) if 'index' in O else None, flag=int(O['flag'].iat[i]), model_score=float(s[i]), m_edep_mev=float(O['m_edep_mev'].iat[i]) if pd.notna(O['m_edep_mev'].iat[i]) else None)
                row.update({c: float(X[i, j]) for j, c in enumerate(cols)}); rows.append(row)
            return rows
        edep = np.nan_to_num(O['m_edep_mev'].to_numpy(float), nan=0.0)
        sig_lo = np.quantile(s[y], 0.05) if y.any() else float('nan')
        r['cases'] = dict(rule_fired_on_signal=cases(y & fired, -edep, a.n_cases),
                          rule_missed_background_model_catches=cases((~y) & (~fired) & (s < sig_lo), s, a.n_cases),
                          model_lowest_on_signal=cases(y, s, a.n_cases),
                          rule_missed_background_model_catches_count=int(((~y) & (~fired) & (s < sig_lo)).sum()), model_signal_5pct_cut=float(sig_lo))
        r['rows'] = dict(objects=int(len(O)), candidates=int(O.groupby(KEY).ngroups) if len(O) < 3_000_000 else None, inputs=cols, kept_lr=lr)
        # figure
        n_in = len(cols); ncol = min(4, n_in); nrow_in = int(np.ceil(n_in / ncol))
        fig = plt.figure(figsize=(4.2 * ncol, 3.1 * (nrow_in + 1)))
        gs = fig.add_gridspec(nrow_in + 1, ncol, hspace=0.55, wspace=0.3)
        for j, c in enumerate(cols):
            ax = fig.add_subplot(gs[j // ncol, j % ncol]); v = X[:, j].astype(float); fin = np.isfinite(v)
            lo, hi = (np.quantile(v[fin], [0.005, 0.995]) if fin.any() else (0, 1));
            if lo == hi: lo, hi = lo - 1, hi + 1
            bins = np.linspace(lo, hi, 41)
            for cls, msk, col, ls in (('signal (true νe CC)', y & fin, 'C3', '-'), ('background', (~y) & fin, 'C0', '-'), ('background, rule fired', (~y) & fired & fin, 'C0', '--'), ('signal, rule fired', y & fired & fin, 'C3', '--')):
                if msk.sum() == 0: continue
                ax.hist(np.clip(v[msk], lo, hi), bins=bins, weights=w[msk] / w[msk].sum(), histtype='step', color=col, ls=ls, label=cls, lw=1.4)
            ax.set_title('%s  (AUC %.2f)' % (c.replace(m + '_v_', ''), r['input_auc'][c]), fontsize=9); ax.tick_params(labelsize=8)
            if j == 0: ax.legend(fontsize=7, frameon=False)
        ax = fig.add_subplot(gs[nrow_in, 0:2]); (c1, th1), (c2, th2) = RULE_DRAW.get(m, ((r['drawn'][0], []), (r['drawn'][1], []))); j1, j2 = cols.index(c1), cols.index(c2)
        r['drawn'] = [c1, c2]; lim = {}
        for cls, msk, col in (('background', ~y, 'C0'), ('signal', y, 'C3')):
            idx = np.flatnonzero(msk); idx = rng.choice(idx, min(a.scatter_n, len(idx)), replace=False) if len(idx) else idx
            ax.scatter(X[idx, j1], X[idx, j2], s=4, c=col, alpha=0.35, label=cls, marker='.', linewidths=0)
            f = np.flatnonzero(msk & fired); f = rng.choice(f, min(a.scatter_n // 4, len(f)), replace=False) if len(f) else f
            if len(f): ax.scatter(X[f, j1], X[f, j2], s=18, facecolors='none', edgecolors='k' if cls == 'background' else 'C1', linewidths=0.6, label=cls + ', rule fired')
        for t in th1: ax.axvline(t, color='k', lw=0.7, ls='--')
        for t in th2: ax.axhline(t, color='k', lw=0.7, ls='--')
        fin = np.isfinite(X[:, j1]) & np.isfinite(X[:, j2])
        if fin.any():
            q1 = np.quantile(X[fin, j1], [0.0, 0.995]); q2 = np.quantile(X[fin, j2], [0.0, 0.995])
            ax.set_xlim(q1[0] - 0.02 * (q1[1] - q1[0] or 1), max(q1[1], max(th1, default=q1[1]) * 1.05)); ax.set_ylim(q2[0] - 0.02 * (q2[1] - q2[0] or 1), max(q2[1], max(th2, default=q2[1]) * 1.05))
        ax.set_xlabel(c1.replace(m + '_v_', '')); ax.set_ylabel(c2.replace(m + '_v_', '')); ax.legend(fontsize=7, frameon=False, markerscale=2)
        ax.set_title('the two inputs the hand-written cut uses (dashed = its thresholds); circled = the cut fired', fontsize=9)
        if n_in > 1 or True:
            ax = fig.add_subplot(gs[nrow_in, 2:4] if ncol >= 4 else gs[nrow_in, ncol - 1])
            bins = np.linspace(np.quantile(s, 0.002), np.quantile(s, 0.998), 51)
            for cls, msk, col, ls in (('signal', y, 'C3', '-'), ('background', ~y, 'C0', '-'), ('background, rule fired', (~y) & fired, 'C0', '--'), ('signal, rule fired', y & fired, 'C3', '--')):
                if msk.sum(): ax.hist(np.clip(s[msk], bins[0], bins[-1]), bins=bins, weights=w[msk] / w[msk].sum(), histtype='step', color=col, ls=ls, label=cls, lw=1.4)
            ax.axvline(cut, color='k', lw=0.8, ls=':'); ax.set_xlabel('retrained %s score (raw margin)' % m); ax.legend(fontsize=7, frameon=False)
            ax.set_title('model score; dotted = cut with the rule\'s background rejection (%.1f %%): signal below %.1f %% vs rule %.1f %%' % (100 * fb, 100 * sig_below, 100 * r['signal']['fired_share']), fontsize=8)
        fig.suptitle('%s: %s objects (%s signal, %s background), exposure-weighted shapes; rule fired on %.1f %% of signal and %.1f %% of background (weighted)' % (
            m, f"{len(O):,}", f"{int(y.sum()):,}", f"{int((~y).sum()):,}", 100 * r['signal']['fired_share'], 100 * fb), fontsize=10)
        fig.savefig(os.path.join(a.out, 'l1_rule_%s.png' % m), dpi=110, bbox_inches='tight'); plt.close(fig)
        out['models'][m] = r
        md.append('| %s | %s | %s | %.1f %% | %.1f %% | %.1f %% | %.3f | %s |' % (m, f"{r['signal']['objects']:,}", f"{r['background']['objects']:,}", 100 * r['signal']['fired_share'], 100 * fb, 100 * sig_below, r['model_auc'], ', '.join(x.replace(m + '_v_', '') for x in r['drawn'])))
        print('%s: %s objects, rule fired on signal %.2f %% / background %.2f %% (weighted); model at same background rejection loses %.2f %% of signal; model AUC %.3f; %.0f s' % (
            m, f"{len(O):,}", 100 * r['signal']['fired_share'], 100 * fb, 100 * sig_below, r['model_auc'], time.time() - t1), flush=True)
        del O, X, s
    out['wall_s'] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(a.out, 'l1_rule_cases.json'), 'w'), indent=1)
    with open(os.path.join(a.out, 'l1_rule_cases.md'), 'w') as f:
        f.write('| model | signal objects | background objects | rule fired on signal (weighted) | rule fired on background (weighted) | retrained model: signal lost at the same background rejection | model AUC (weighted) | inputs drawn |\n|---|---:|---:|---:|---:|---:|---:|---|\n' + '\n'.join(md) + '\n')
    print('done in %.0f s' % (time.time() - t0))

if __name__ == '__main__':
    main()
