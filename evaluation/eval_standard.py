#!/usr/bin/env python3
"""The evaluation standard (issue #9): Matteo Vicenzi's nueCC efficiency / purity table, computed from spec_v1 tables.

Method (docs/normalization_and_evaluation_standard.md; his work/plots.py, effpur.py, cuts.py at commit 9f2803b):
  mixed MC   BNB MC without the spills that hold a nu_e/anti-nu_e CC vertex inside the filter volume, weight 1, plus the
             nueCC-filtered MC whole, weight POT_bnb / POT_nuecc. POT = sum of pot_hdr over the files read without error.
  candidates T_tagger rows that are not placeholders (kine_reco_Enu == 0, neutrino_type == 0, vertex at the origin).
  efficiency truth mode: one row per matchable signal neutrino; it passes when its best reco object is a candidate
             (nu_ct_cand_index >= 0, nu_ct_iou_n > 0) and that candidate passes the selection.
  purity     reco mode: one row per candidate passing the selection; signal when its best truth partner is a neutrino
             that passes the signal definition.
  fractions  weighted, sum(w k) / sum(w n), binomial error with sum(w^2); counts are at the BNB POT.
Signal, FV and the placeholder rule come from spec_v1.py. Files with label_unreadable = 1 are left out (candidates and
neutrinos) and counted; files with truth_pf_unreadable = 1 are counted (no true lepton, so no e500 signal from them).

  python eval_standard.py TABLE_DIR BNB_SAMPLE NUECC_SAMPLE [--score nue_score] [--cut 7.0] [--classes nueCC,nueCC_e500]
                          [--drop-repeated-keys] [--file-ids keep.txt] [--out result.json]
--score names the candidate column holding the model score (the stored nue_score by default; a retrained model's column
otherwise). --drop-repeated-keys drops events whose (run, subrun, event) is held by two files, as Matteo's loader does.
--file-ids restricts both samples to the file ids listed (one per line), e.g. the lists with the hold-out files removed.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'step3_spec'))
import spec_v1

EV = ['file_id', 'run', 'subrun', 'event']
EDGES = [0, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 6000]          # MeV; the last bin is the overflow
FILTER = dict(x=(-202.3, 202.3), y=(-203.73, 203.73), z=(0.0, 501.0))          # his plots.NUECC_FILTER
E500_MEV = 500.0


def load(d, sample, score, file_ids):
    F = pd.read_parquet(os.path.join(d, 'files_%s.parquet' % sample))
    T = pd.read_parquet(os.path.join(d, 'truth_%s.parquet' % sample))
    need = EV + ['cand', 'reco_x', 'reco_y', 'reco_z', 'kine_reco_Enu', 'neutrino_type', 'reco_e_mev', 'partner_kind', 'm_nu_idx', 'iou_n', 'label_unreadable', score]
    C = pd.read_parquet(os.path.join(d, 'cands_%s.parquet' % sample), columns=list(dict.fromkeys(need)))
    for c in ['run', 'subrun', 'event']: C[c] = C[c].astype('int64')
    if file_ids is not None:
        F = F[F.file_id.isin(file_ids)]; T = T[T.file_id.isin(file_ids)]; C = C[C.file_id.isin(file_ids)]
    ok = F[F.error == '']
    info = dict(files=int(len(F)), files_with_error=int((F.error != '').sum()), label_unreadable_files=int(F.label_unreadable.sum()),
                events=int(ok.events.sum()), wc_events=int(ok.wc_events.sum()), pot_hdr=float(ok.pot_hdr.sum()), pot_total_hist=float(ok.pot_total_hist.sum()),
                candidate_rows=int(len(C)), truth_rows=int(len(T)))
    C = C[C.label_unreadable == 0]; T = T[T.label_unreadable == 0]
    # files whose T_truth_pf could not be read have no true lepton: their neutrinos cannot enter the e500 signal (counted here)
    info['truth_pf_unreadable_files'] = int(F.truth_pf_unreadable.sum()) if 'truth_pf_unreadable' in F.columns else 0
    ph = spec_v1.placeholder_mask(C.kine_reco_Enu, C.neutrino_type, C.reco_x, C.reco_y, C.reco_z)
    info['placeholder_rows'] = int(ph.sum()); C = C[~ph].copy()
    return F, C, T.copy(), info


def drop_repeated(C, T, info):
    k = pd.concat([C[EV], T[EV]]).drop_duplicates()
    n = k.groupby(['run', 'subrun', 'event']).size(); rep = n[n > 1].index
    info['repeated_keys_dropped'] = int(len(rep))
    f = lambda D: D[~pd.MultiIndex.from_frame(D[['run', 'subrun', 'event']]).isin(rep)]
    return f(C), f(T)


def wfrac(parts):
    """(fraction, error, weighted n, weighted k, raw k, raw n) of [(boolean array, weight)]; as his plots._wfrac."""
    k = np.array([p.sum() for p, _ in parts], float); n = np.array([len(p) for p, _ in parts], float); w = np.array([w for _, w in parts], float)
    wn = (w * n).sum()
    if wn == 0: return np.nan, np.nan, 0.0, 0.0, 0, 0
    f = (w * k).sum() / wn
    err = np.sqrt((w ** 2 * (k * (1 - f) ** 2 + (n - k) * f ** 2)).sum()) / wn
    return float(f), float(err), float(wn), float(f * wn), int(k.sum()), int(n.sum())


def true_signal(T, cls):
    """Signal definition on truth-table rows (matchable is required by the loops, as in his table)."""
    base = spec_v1.matchable_mask(T.edep_mev, T.vx, T.vy, T.vz) & (np.abs(T.pdg) == 12) & (T.ccnc == 0) & spec_v1.in_fv(T.vx, T.vy, T.vz)
    if cls == 'nueCC': return base & (T.edep_mev > spec_v1.EDEP_MIN_MEV)
    if cls == 'nueCC_e500': return base & (np.abs(T.true_lep_pdg) == 11) & (T.true_lep_ke_mev > E500_MEV)
    raise ValueError(cls)


def steps_for(cls, score, cut):
    s = [('candidate exists', lambda c: np.ones(len(c), bool)),
         ('+ reco vertex in FV', lambda c: spec_v1.in_fv(c.reco_x, c.reco_y, c.reco_z)),
         ('+ %s > %g' % (score, cut), lambda c: (c[score] > cut).to_numpy())]
    if cls == 'nueCC_e500': s.append(('+ reco electron > %g MeV' % E500_MEV, lambda c: (c.reco_e_mev > E500_MEV).to_numpy()))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('table_dir'); ap.add_argument('bnb'); ap.add_argument('nuecc')
    ap.add_argument('--score', default='nue_score'); ap.add_argument('--cut', type=float, default=7.0); ap.add_argument('--classes', default='nueCC,nueCC_e500')
    ap.add_argument('--drop-repeated-keys', action='store_true'); ap.add_argument('--file-ids'); ap.add_argument('--out')
    a = ap.parse_args()
    ids = None if not a.file_ids else set(l.strip() for l in open(a.file_ids) if l.strip())
    res = dict(score=a.score, cut=a.cut, samples={}, classes={}); parts = []
    for name, sample in [('bnb', a.bnb), ('nuecc', a.nuecc)]:
        F, C, T, info = load(a.table_dir, sample, a.score, ids)
        if a.drop_repeated_keys: C, T = drop_repeated(C, T, info)
        if name == 'bnb':                       # drop the spills the filtered sample holds
            hit = (np.abs(T.pdg) == 12) & (T.ccnc == 0) & T.vx.between(*FILTER['x']) & T.vy.between(*FILTER['y']) & T.vz.between(*FILTER['z'])
            sp = pd.MultiIndex.from_frame(T.loc[hit, EV].drop_duplicates())
            info['filtered_spills_dropped'] = int(len(sp)); n0 = len(C)
            C = C[~pd.MultiIndex.from_frame(C[EV]).isin(sp)]; T = T[~pd.MultiIndex.from_frame(T[EV]).isin(sp)]
            info['candidates_after_spill_drop'] = int(len(C)); info['candidates_before_spill_drop'] = int(n0)
        info['candidates_used'] = int(len(C)); res['samples'][sample] = info; parts.append([name, C, T, info['pot_hdr']])
    w = {'bnb': 1.0, 'nuecc': parts[0][3] / parts[1][3]}; res['weights'] = w
    print('samples: ' + json.dumps(res['samples'])); print('weights: bnb 1, nuecc %.7g' % w['nuecc'])
    for cls in a.classes.split(','):
        steps = steps_for(cls, a.score, a.cut); rows = []; binned = []
        pre = []
        for name, C, T, _ in parts:
            sig = T[true_signal(T, cls)].copy()
            tm = sig.merge(C, left_on=EV + ['nu_ct_cand_index'], right_on=EV + ['cand'], how='left', suffixes=('', '_c'))
            tm['matched'] = (tm.nu_ct_cand_index >= 0) & (tm.nu_ct_iou_n > 0) & tm.cand.notna()
            Tk = T.assign(is_sig=true_signal(T, cls))[EV + ['nu_idx', 'is_sig']]
            rm = C.merge(Tk, left_on=EV + ['m_nu_idx'], right_on=EV + ['nu_idx'], how='left')
            rm['signal'] = (rm.partner_kind == 'neutrino') & (rm.iou_n > 0) & (rm.is_sig == True)
            pre.append((name, tm, rm))
        for k in range(len(steps)):
            eff, pur, effb, purb = [], [], [], []
            for name, tm, rm in pre:
                pt = tm.matched.to_numpy().copy(); pr = np.ones(len(rm), bool)
                for _, fn in steps[:k + 1]:
                    with np.errstate(invalid='ignore'): pt &= np.asarray(fn(tm), bool) & tm.cand.notna().to_numpy(); pr &= np.asarray(fn(rm), bool)
                sel = rm[pr]
                eff.append((pt, w[name])); pur.append((sel.signal.to_numpy(), w[name]))
                effb.append((1000.0 * tm.E.to_numpy(), pt, w[name])); purb.append((sel.kine_reco_Enu.to_numpy(), sel.signal.to_numpy(), w[name]))
            fe, ee, nsig, ksig, rke, rne = wfrac(eff); fp, ep, nsel, ksel, rkp, rnp = wfrac(pur)
            rows.append(dict(step=steps[k][0], efficiency=fe, efficiency_err=ee, signal_kept=ksig, signal=nsig, raw_eff=[rke, rne],
                             purity=fp, purity_err=ep, signal_selected=ksel, selected=nsel, raw_pur=[rkp, rnp]))
            b = dict(step=steps[k][0], edges_mev=EDGES, efficiency_vs_true_E=[], purity_vs_reco_E=[])
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                b['efficiency_vs_true_E'].append(wfrac([(p[(x > lo) & (x <= hi)], ww) for x, p, ww in effb])[:2])
                b['purity_vs_reco_E'].append(wfrac([(p[(x > lo) & (x <= hi)], ww) for x, p, ww in purb])[:2])
            binned.append(b)
        res['classes'][cls] = dict(steps=rows, binned=binned)
        print('\n=== %s   (BNB MC + nueCC-filtered MC; counts at the BNB POT)' % cls)
        print('%-28s %-16s %-20s %-16s %-24s %s' % ('step', 'efficiency', 'signal kept / signal', 'purity', 'signal sel. / selected', 'raw eff, raw pur'))
        for r in rows:
            print('%-28s %5.1f +- %-6.1f %9.1f / %-9.1f %5.1f +- %-6.1f %9.1f / %-13.1f %d/%d, %d/%d' % (r['step'], 100 * r['efficiency'], 100 * r['efficiency_err'], r['signal_kept'], r['signal'],
                  100 * r['purity'], 100 * r['purity_err'], r['signal_selected'], r['selected'], r['raw_eff'][0], r['raw_eff'][1], r['raw_pur'][0], r['raw_pur'][1]))
        last = binned[-1]
        print('  last step, per bin [MeV]: ' + '  '.join('%d-%d' % (lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])))
        print('  efficiency vs true E  %: ' + '  '.join('%.1f' % (100 * f) if f == f else '-' for f, _ in last['efficiency_vs_true_E']))
        print('  purity vs reco E      %: ' + '  '.join('%.1f' % (100 * f) if f == f else '-' for f, _ in last['purity_vs_reco_E']))
    if a.out: json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
