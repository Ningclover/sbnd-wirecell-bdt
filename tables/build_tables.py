#!/usr/bin/env python3
"""spec_v1 table builder: merged CAF(s) -> cands_<sample>.parquet, truth_<sample>.parquet, files_<sample>.parquet
(and objs_<sample>.parquet with --objs). The same code runs on Aurora (training samples) and on gpvm / FermiGrid
(inference samples). Python 3.9 syntax; needs numpy, pandas, pyarrow, uproot (5.6 or newer).

  python build_tables.py SAMPLE OUT_DIR --files a.root b.root ...          # these files (paths or root:// URLs)
  python build_tables.py SAMPLE OUT_DIR --list list.txt --slice 0:200      # lines [0, 200) of a list, one path per line
  python build_tables.py SAMPLE OUT_DIR --list list.txt --nproc 8          # local multiprocessing over the whole list

Columns are the ones spec_v1.md section 5 names. The label rules are applied later, when a table is read, through
spec_v1.py; this file only extracts. file_id = the CAF file name by default (--file-id-from basename); 'unit' gives the aurora-r2 unit folder two levels up.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
WC = 'recTreeWireCell'
KEY = ['sample', 'file_id', 'run', 'subrun', 'event']
KEY_CLASH = {'wc.T_tagger.run', 'wc.T_tagger.subrun', 'wc.T_tagger.event', 'wc.T_tagger.eventNo', 'wc.T_tagger.runNo', 'wc.T_tagger.subRunNo'}   # tagger scalars that would overwrite the key

MATCH_BRANCHES = ['level', 'side', 'obj', 'nu_index', 'truth_nu_idx', 'truth_trackid', 'nmatch', 'mutual',
                  'match_obj', 'match_iou_n', 'match_iou_q', 'match_shared_n', 'match_shared_q',
                  'match_obj..idx', 'match_obj..length', 'match_iou_n..idx', 'match_iou_q..idx', 'match_shared_n..idx', 'match_shared_q..idx']
DLV = ['pass', 'nu_index', 'accepted', 'status', 'n_points', 'n_space', 'two_end_veto', 'final_valid', 'final_x', 'final_y', 'final_z',
       'rerank_valid', 'rerank_x', 'rerank_y', 'rerank_z', 'trad_valid', 'trad_x', 'trad_y', 'trad_z', 'hint_valid', 'dl_x', 'dl_y', 'dl_z',
       'truth_all_nu_idx', 'truth_all_reco_x', 'truth_all_reco_y', 'truth_all_reco_z', 'truth_all_reco_x..idx', 'truth_all_reco_x..length',
       'truth_all_nu_idx..idx', 'truth_all_nu_idx..length']
HDR = ['rec.hdr.run', 'rec.hdr.subrun', 'rec.hdr.evt', 'rec.hdr.first_in_subrun', 'rec.hdr.pot', 'rec.hdr.nbnbinfo', 'rec.hdr.noffbeambnb', 'rec.hdr.ngenevt', 'rec.hdr.ismc']


def _first(a, i, name, default=np.nan):
    v = a[name][i]
    return v[0] if len(v) else default


def _slice(a, name, i, c):
    """The c-th sub-list of a doubly jagged branch (flattened per event with ..idx/..length)."""
    i0 = int(a[name + '..idx'][i][c]); n = int(a[name + '..length'][i][c])
    return a[name][i][i0:i0 + n]


def extract(path, sample, file_id, want_objs=False, root=None):
    rel = os.path.relpath(path, root) if root and not path.startswith('root://') else path
    out = dict(file_id=file_id, path=rel, events=0, wc_events=0, candidates=0, label_unreadable=0, partner_missing=0,
               wct_version='', pot_hdr=0.0, pot_total_hist=np.nan, n_bnb_spills=0.0, n_offbeam_gates=0.0, ngenevt=0.0, ismc=-1, truth_pf_unreadable=0, error='')
    cands, truth, objs = [], [], []
    try:
        f = uproot.open(path)
    except Exception as e:
        out['error'] = 'OPEN_FAIL ' + str(e).splitlines()[0][:120]; return out, cands, truth, objs
    # ---- header exposure (recTree) ----
    try:
        h = f['recTree'].arrays(HDR, library='np')
        fis = h['rec.hdr.first_in_subrun'] == 1
        out.update(pot_hdr=float(h['rec.hdr.pot'][fis].sum()), n_bnb_spills=float(h['rec.hdr.nbnbinfo'][fis].sum()),
                   n_offbeam_gates=float(h['rec.hdr.noffbeambnb'][fis].sum()), ngenevt=float(h['rec.hdr.ngenevt'][fis].sum()),
                   ismc=int(h['rec.hdr.ismc'][0]) if len(h['rec.hdr.ismc']) else -1, events=int(len(h['rec.hdr.run'])))
        if 'TotalPOT;1' in f.keys() or 'TotalPOT' in f:
            try: out['pot_total_hist'] = float(np.sum(f['TotalPOT'].values()))
            except Exception: pass
    except Exception as e:
        out['error'] = 'HDR_FAIL ' + str(e).splitlines()[0][:120]
    t = f[WC]; keys = set(t.keys())
    tag_scalar = sorted(k for k in keys if k.startswith('wc.T_tagger.') and '..' not in k and k != 'wc.T_tagger.' and k not in KEY_CLASH)   # per-candidate scalars; a branch shorter than the candidate list gives NaN
    tag_list = sorted({k.split('..')[0] for k in keys if k.startswith('wc.T_tagger.') and k.endswith('..idx')})
    has_match = any(k.startswith('wc.T_match.') for k in keys)
    has_truth = 'wc.T_truth_nu.nu_idx' in keys
    base = ['wc.run', 'wc.subrun', 'wc.evt', 'wc.valid', 'wc.Trun.wct_version',
            'wc.T_kine.nu_index', 'wc.T_kine.kine_reco_Enu', 'wc.T_kine.kine_particle_type', 'wc.T_kine.kine_energy_particle', 'wc.T_kine.kine_particle_pool',
            'wc.T_kine.kine_particle_type..idx', 'wc.T_kine.kine_particle_type..length']
    base += tag_scalar
    if has_truth:
        base += ['wc.T_truth_nu.' + n for n in ['nu_idx', 'pdg', 'ccnc', 'mode', 'E', 'edep', 'vtx_x', 'vtx_y', 'vtx_z']]
    base += ['wc.T_dlvtx_call.' + n for n in DLV if 'wc.T_dlvtx_call.' + n in keys]
    if want_objs:
        base += [b for b in tag_list] + [b + '..idx' for b in tag_list] + [b + '..length' for b in tag_list]
    try:
        a = t.arrays(base, library='np')
    except Exception as e:
        out['error'] = 'READ_FAIL ' + str(e).splitlines()[0][:120]; return out, cands, truth, objs
    pf = None   # T_truth_pf read on its own: in some files its E/KE baskets are unreadable (same uproot limit as T_match); then only the true-lepton columns are lost
    if has_truth:
        try:
            pf = t.arrays(['wc.T_truth_pf.' + n for n in ['nu_row', 'pdg', 'parent_trackid', 'KE']], library='np')
        except Exception:
            out['truth_pf_unreadable'] = 1; pf = None
    m = None
    if has_match:
        try:
            m = t.arrays(['wc.T_match.' + n for n in MATCH_BRANCHES], library='np')
        except Exception:
            out['label_unreadable'] = 1; m = None
    out['wc_events'] = int((a['wc.valid'] == 1).sum())
    wv = a['wc.Trun.wct_version']; out['wct_version'] = str(wv[0][0]) if len(wv) and len(wv[0]) else ''
    unreadable = int(out['label_unreadable'])
    for i in range(len(a['wc.run'])):
        if a['wc.valid'][i] != 1: continue
        key = dict(sample=sample, file_id=file_id, run=int(a['wc.run'][i]), subrun=int(a['wc.subrun'][i]), event=int(a['wc.evt'][i]))
        # SCE-shifted true vertices of the first DL call, by nu_idx
        sce = {}
        if 'wc.T_dlvtx_call.pass' in a and len(a['wc.T_dlvtx_call.pass'][i]) and 'wc.T_dlvtx_call.truth_all_reco_x..idx' in a:
            n0 = int(a['wc.T_dlvtx_call.truth_all_reco_x..length'][i][0]); i0 = int(a['wc.T_dlvtx_call.truth_all_reco_x..idx'][i][0])
            if n0 > 0:
                if 'wc.T_dlvtx_call.truth_all_nu_idx' in a and len(a['wc.T_dlvtx_call.truth_all_nu_idx'][i]) >= i0 + n0:
                    idxs = a['wc.T_dlvtx_call.truth_all_nu_idx'][i][i0:i0 + n0]
                else:
                    idxs = range(n0)
                for q, ix in enumerate(idxs):
                    sce[int(ix)] = (float(a['wc.T_dlvtx_call.truth_all_reco_x'][i][i0 + q]), float(a['wc.T_dlvtx_call.truth_all_reco_y'][i][i0 + q]), float(a['wc.T_dlvtx_call.truth_all_reco_z'][i][i0 + q]))
        # truth rows
        tn = {}
        if has_truth:
            tn = {k.split('.')[-1]: a[k][i] for k in a if k.startswith('wc.T_truth_nu.')}
            pf_nu, pf_pdg, pf_par, pf_ke = (pf['wc.T_truth_pf.' + n][i] for n in ('nu_row', 'pdg', 'parent_trackid', 'KE')) if pf is not None else ([], [], [], [])
        truth_by_idx = {int(ix): q for q, ix in enumerate(tn.get('nu_idx', []))}
        # T_match maps
        lv = sd = None; reco_rows = {}; truth_rows = {}; obj_of_reco_row = {}
        if m is not None:
            lv, sd = m['wc.T_match.level'][i], m['wc.T_match.side'][i]
            for k in range(len(lv)):
                if lv[k] != 2: continue
                if sd[k] == 0: reco_rows[int(m['wc.T_match.nu_index'][i][k])] = k; obj_of_reco_row[int(m['wc.T_match.obj'][i][k])] = k
                else: truth_rows[int(m['wc.T_match.obj'][i][k])] = k
            tni_of_truth_row = {k: int(m['wc.T_match.truth_nu_idx'][i][k]) for k in truth_rows.values()}
            truth_row_of_tni = {tni: k for k, tni in tni_of_truth_row.items() if tni >= 0}
        def partners(k):
            n = int(m['wc.T_match.nmatch'][i][k])
            if n <= 0: return []
            i0 = int(m['wc.T_match.match_obj..idx'][i][k]); nn = int(m['wc.T_match.match_obj..length'][i][k])
            return [(int(m['wc.T_match.match_obj'][i][i0 + j]), float(m['wc.T_match.match_iou_n'][i][int(m['wc.T_match.match_iou_n..idx'][i][k]) + j]),
                     float(m['wc.T_match.match_iou_q'][i][int(m['wc.T_match.match_iou_q..idx'][i][k]) + j]),
                     float(m['wc.T_match.match_shared_n'][i][int(m['wc.T_match.match_shared_n..idx'][i][k]) + j]),
                     float(m['wc.T_match.match_shared_q'][i][int(m['wc.T_match.match_shared_q..idx'][i][k]) + j])) for j in range(nn)]
        if has_truth:
            for q, ix in enumerate(tn['nu_idx']):
                ix = int(ix); r = dict(key, nu_idx=ix, pdg=int(tn['pdg'][q]), ccnc=int(tn['ccnc'][q]), mode=int(tn['mode'][q]), E=float(tn['E'][q]), edep_mev=1000.0 * float(tn['edep'][q]),
                                       vx=float(tn['vtx_x'][q]), vy=float(tn['vtx_y'][q]), vz=float(tn['vtx_z'][q]), sce_x=np.nan, sce_y=np.nan, sce_z=np.nan,
                                       true_lep_pdg=0, true_lep_ke_mev=np.nan, truth_pf_unreadable=int(out['truth_pf_unreadable']), nu_ct_cand_index=-1, nu_ct_iou_n=np.nan, nu_ct_iou_q=np.nan, nu_ct_mutual=-1, label_unreadable=unreadable)
                if ix in sce: r['sce_x'], r['sce_y'], r['sce_z'] = sce[ix]
                best = None
                for j in range(len(pf_nu)):
                    if int(pf_nu[j]) == q and int(pf_par[j]) == 0 and abs(int(pf_pdg[j])) in (11, 13):
                        if best is None or float(pf_ke[j]) > best[1]: best = (int(pf_pdg[j]), float(pf_ke[j]))
                if best is not None: r['true_lep_pdg'], r['true_lep_ke_mev'] = best[0], 1000.0 * best[1]
                if m is not None and ix in truth_row_of_tni:
                    k = truth_row_of_tni[ix]; r['nu_ct_mutual'] = int(m['wc.T_match.mutual'][i][k]); ps = partners(k)
                    if ps:
                        po, ioun, iouq, _, _ = ps[0]; r['nu_ct_iou_n'], r['nu_ct_iou_q'] = ioun, iouq
                        if po in obj_of_reco_row: r['nu_ct_cand_index'] = int(m['wc.T_match.nu_index'][i][obj_of_reco_row[po]])
                truth.append(r)
        # candidates
        kine_idx = {int(v): c for c, v in enumerate(a['wc.T_kine.nu_index'][i])}
        for c, ni in enumerate(a['wc.T_tagger.nu_index'][i]):
            ni = int(ni); rx, ry, rz = (float(a['wc.T_tagger.nu_' + k][i][c]) for k in 'xyz')
            r = dict(key, cand=ni, wct_version=out['wct_version'], reco_x=rx, reco_y=ry, reco_z=rz, kine_reco_Enu=np.nan, reco_e_mev=np.nan,
                     partner_kind='none', iou_n=np.nan, iou_q=np.nan, shared_n=np.nan, shared_q=np.nan, iou_n_2=np.nan, iou_q_2=np.nan, mutual=-1, nmatch=0,
                     m_nu_idx=-1, m_pdg=0, m_ccnc=-1, m_E=np.nan, m_edep_mev=np.nan, m_vx=np.nan, m_vy=np.nan, m_vz=np.nan, m_sce_x=np.nan, m_sce_y=np.nan, m_sce_z=np.nan,
                     dist_sce_cm=np.nan, dist_raw_cm=np.nan, sce_fallback=0, label_unreadable=unreadable)
            for k in tag_scalar:
                v = a[k][i]; r[k[len('wc.T_tagger.'):]] = float(v[c]) if c < len(v) else np.nan
            kc = kine_idx.get(ni)
            if kc is not None:
                r['kine_reco_Enu'] = float(a['wc.T_kine.kine_reco_Enu'][i][kc])
                i0 = int(a['wc.T_kine.kine_particle_type..idx'][i][kc]); n = int(a['wc.T_kine.kine_particle_type..length'][i][kc])
                ty = a['wc.T_kine.kine_particle_type'][i][i0:i0 + n]; en = a['wc.T_kine.kine_energy_particle'][i][i0:i0 + n]; po_ = a['wc.T_kine.kine_particle_pool'][i][i0:i0 + n]
                sel = [float(en[j]) for j in range(n) if abs(int(ty[j])) == 11 and int(po_[j]) == 0]
                if sel: r['reco_e_mev'] = max(sel)
            # Matteo's placeholder rule (wcreco.py): an empty T_tagger row, not a candidate: Enu 0, neutrino_type 0, vertex at the origin
            r['is_placeholder'] = int(r['kine_reco_Enu'] == 0.0 and r.get('neutrino_type', 0.0) == 0.0 and rx == 0.0 and ry == 0.0 and rz == 0.0)
            if m is not None and not unreadable:
                k = reco_rows.get(ni)
                if k is not None:
                    r['nmatch'] = int(m['wc.T_match.nmatch'][i][k]); r['mutual'] = int(m['wc.T_match.mutual'][i][k]); ps = partners(k)
                    if ps:
                        po, r['iou_n'], r['iou_q'], r['shared_n'], r['shared_q'] = ps[0]
                        if len(ps) > 1: r['iou_n_2'], r['iou_q_2'] = ps[1][1], ps[1][2]
                        if po in truth_rows:
                            tni = tni_of_truth_row[truth_rows[po]]
                            if tni >= 0 and tni in truth_by_idx:
                                q = truth_by_idx[tni]; vx, vy, vz = float(tn['vtx_x'][q]), float(tn['vtx_y'][q]), float(tn['vtx_z'][q])
                                r.update(partner_kind='neutrino', m_nu_idx=tni, m_pdg=int(tn['pdg'][q]), m_ccnc=int(tn['ccnc'][q]), m_E=float(tn['E'][q]), m_edep_mev=1000.0 * float(tn['edep'][q]), m_vx=vx, m_vy=vy, m_vz=vz)
                                r['dist_raw_cm'] = float(np.sqrt((vx - rx) ** 2 + (vy - ry) ** 2 + (vz - rz) ** 2))
                                if tni in sce:
                                    sx, sy, sz = sce[tni]; r['m_sce_x'], r['m_sce_y'], r['m_sce_z'] = sx, sy, sz; r['dist_sce_cm'] = float(np.sqrt((sx - rx) ** 2 + (sy - ry) ** 2 + (sz - rz) ** 2))
                                else:
                                    r['dist_sce_cm'] = r['dist_raw_cm']; r['sce_fallback'] = 1
                            else:
                                r['partner_kind'] = 'cosmic'
                        else:
                            r['partner_kind'] = 'missing'; out['partner_missing'] += 1
            cands.append(r)
            if want_objs:
                for b in tag_list:
                    name = b[len('wc.T_tagger.'):]
                    if c < len(a[b + '..idx'][i]):
                        i0 = int(a[b + '..idx'][i][c]); n = int(a[b + '..length'][i][c])
                        for j in range(n): objs.append(dict(key, cand=ni, list_name=name, index=j, value=float(a[b][i][i0 + j])))
    out['candidates'] = len(cands)
    return out, cands, truth, objs


def _worker(args):
    path, sample, fid, want_objs, root = args
    try:
        return extract(path, sample, fid, want_objs, root)
    except Exception as e:
        return dict(file_id=fid, path=path, error='EXC ' + str(e).splitlines()[0][:120], events=0, wc_events=0, candidates=0, label_unreadable=0, partner_missing=0, wct_version='',
                    pot_hdr=0.0, pot_total_hist=np.nan, n_bnb_spills=0.0, n_offbeam_gates=0.0, ngenevt=0.0, ismc=-1, truth_pf_unreadable=0), [], [], []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('sample'); ap.add_argument('out_dir')
    ap.add_argument('--files', nargs='*', default=[]); ap.add_argument('--list'); ap.add_argument('--slice', default=None, help='start:stop lines of --list')
    ap.add_argument('--nproc', type=int, default=1); ap.add_argument('--objs', action='store_true'); ap.add_argument('--file-id-from', choices=['basename', 'dirname', 'unit'], default='basename', help='basename: the file name (default, unique in every sample); dirname: the parent folder; unit: the folder two levels up (the aurora-r2 unit id)')
    ap.add_argument('--tag', default='', help='suffix for the output names, e.g. a slice id')
    ap.add_argument('--root', default=None, help='sample root; the files table stores paths relative to it (spec_v1 section 5)')
    args = ap.parse_args()
    files = list(args.files)
    if args.list:
        lines = [l.strip() for l in open(args.list) if l.strip() and not l.startswith('#')]
        if args.slice:
            s0, s1 = args.slice.split(':'); lines = lines[int(s0 or 0):int(s1) if s1 else None]
        files += lines
    if not files: sys.exit('no input files')
    fid = {'basename': lambda p: os.path.basename(p), 'dirname': lambda p: os.path.basename(os.path.dirname(p.rstrip('/'))),
           'unit': lambda p: os.path.basename(os.path.dirname(os.path.dirname(p.rstrip('/'))))}[args.file_id_from]
    jobs = [(p, args.sample, fid(p), args.objs, args.root) for p in files]
    t0 = time.time()
    if args.nproc > 1:
        import multiprocessing as mp
        with mp.Pool(args.nproc) as pool: res = pool.map(_worker, jobs, chunksize=2)
    else:
        res = [_worker(j) for j in jobs]
    os.makedirs(args.out_dir, exist_ok=True); sfx = ('_' + args.tag) if args.tag else ''
    F = pd.DataFrame([r[0] for r in res]); C = pd.DataFrame([x for r in res for x in r[1]]); T = pd.DataFrame([x for r in res for x in r[2]])
    F.to_parquet(os.path.join(args.out_dir, 'files_%s%s.parquet' % (args.sample, sfx)))
    C.to_parquet(os.path.join(args.out_dir, 'cands_%s%s.parquet' % (args.sample, sfx)))
    T.to_parquet(os.path.join(args.out_dir, 'truth_%s%s.parquet' % (args.sample, sfx)))
    if args.objs:
        pd.DataFrame([x for r in res for x in r[3]]).to_parquet(os.path.join(args.out_dir, 'objs_%s%s.parquet' % (args.sample, sfx)))
    summ = dict(sample=args.sample, files=len(files), errors=int((F.error != '').sum()), label_unreadable_files=int(F.label_unreadable.sum()), partner_missing=int(F.partner_missing.sum()),
                events=int(F.events.sum()), wc_events=int(F.wc_events.sum()), candidates=len(C), truth_rows=len(T), pot_hdr=float(F.pot_hdr.sum()), wall_s=round(time.time() - t0, 1))
    json.dump(summ, open(os.path.join(args.out_dir, 'build_%s%s.json' % (args.sample, sfx)), 'w'), indent=1); print(json.dumps(summ))


if __name__ == '__main__':
    main()
