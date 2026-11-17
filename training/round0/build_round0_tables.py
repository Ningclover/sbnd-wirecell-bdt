"""Round-0 tables: one row per candidate with the 258 shipped nue-model inputs, the stored scores, the reco vertex, and
the T_match level-2 label columns of spec_v1 (partner, matched truth, SCE vertex distance), plus one truth table.
Usage: python build_round0_tables.py <campaign> <list> <n_files> <nproc> <out_dir>"""
import sys, os, time, json, multiprocessing as mp
import numpy as np, pandas as pd, uproot
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = [l.strip() for l in open(os.path.join(HERE, 'nue_l2_inputs_258.txt')) if l.strip()]
TAG = [n for n in NAMES if n != 'kine_reco_Enu']
B = (['wc.run', 'wc.subrun', 'wc.evt', 'wc.valid', 'wc.T_tagger.nu_index', 'wc.T_tagger.br_filled', 'wc.T_tagger.nue_score', 'wc.T_tagger.numu_score',
      'wc.T_tagger.nu_x', 'wc.T_tagger.nu_y', 'wc.T_tagger.nu_z', 'wc.T_kine.kine_reco_Enu',
      'wc.T_truth_nu.nu_idx', 'wc.T_truth_nu.pdg', 'wc.T_truth_nu.ccnc', 'wc.T_truth_nu.E', 'wc.T_truth_nu.edep', 'wc.T_truth_nu.vtx_x', 'wc.T_truth_nu.vtx_y', 'wc.T_truth_nu.vtx_z',
      'wc.T_match.level', 'wc.T_match.side', 'wc.T_match.obj', 'wc.T_match.nu_index', 'wc.T_match.truth_nu_idx', 'wc.T_match.nmatch', 'wc.T_match.mutual',
      'wc.T_match.match_obj', 'wc.T_match.match_iou_q', 'wc.T_match.match_iou_n', 'wc.T_match.match_obj..idx', 'wc.T_match.match_iou_q..idx', 'wc.T_match.match_iou_n..idx',
      'wc.T_dlvtx_call.pass', 'wc.T_dlvtx_call.truth_all_reco_x', 'wc.T_dlvtx_call.truth_all_reco_y', 'wc.T_dlvtx_call.truth_all_reco_z',
      'wc.T_dlvtx_call.truth_all_reco_x..idx', 'wc.T_dlvtx_call.truth_all_reco_x..length', 'wc.Trun.wct_version']
     + ['wc.T_tagger.' + n for n in TAG])

def one(args):
    fi, caf, name = args
    cands, truth = [], []
    try:
        a = uproot.open(caf)['recTreeWireCell'].arrays(B, library='np')
    except Exception as e:
        return fi, caf, 'READ_FAIL ' + str(e).splitlines()[0][:100], [], []
    for i in range(len(a['wc.run'])):
        if a['wc.valid'][i] != 1: continue
        key = dict(sample=name, file=fi, run=int(a['wc.run'][i]), subrun=int(a['wc.subrun'][i]), event=int(a['wc.evt'][i]))
        tn = {k.split('.')[-1]: a[k][i] for k in B if k.startswith('wc.T_truth_nu.')}
        wv = a['wc.Trun.wct_version'][i]; wv = str(wv[0]) if len(wv) else ''
        sce = None
        if len(a['wc.T_dlvtx_call.pass'][i]):
            i0 = int(a['wc.T_dlvtx_call.truth_all_reco_x..idx'][i][0]); n0 = int(a['wc.T_dlvtx_call.truth_all_reco_x..length'][i][0])
            if n0 > 0: sce = np.stack([a['wc.T_dlvtx_call.truth_all_reco_' + c][i][i0:i0 + n0] for c in 'xyz'], axis=1)
        for q, ix in enumerate(tn['nu_idx']):
            truth.append(dict(key, nu_idx=int(ix), pdg=int(tn['pdg'][q]), ccnc=int(tn['ccnc'][q]), E=float(tn['E'][q]), edep_mev=1000 * float(tn['edep'][q]),
                              vx=float(tn['vtx_x'][q]), vy=float(tn['vtx_y'][q]), vz=float(tn['vtx_z'][q])))
        truth_by_idx = {int(ix): q for q, ix in enumerate(tn['nu_idx'])}
        lv, sd = a['wc.T_match.level'][i], a['wc.T_match.side'][i]
        tr = {int(o): j for j, o in enumerate(a['wc.T_match.obj'][i]) if lv[j] == 2 and sd[j] == 1}
        reco_rows = {int(o): k for k, o in enumerate(a['wc.T_match.nu_index'][i]) if lv[k] == 2 and sd[k] == 0}
        enu = a['wc.T_kine.kine_reco_Enu'][i]
        for c, ni in enumerate(a['wc.T_tagger.nu_index'][i]):
            rx, ry, rz = (float(a['wc.T_tagger.nu_' + k][i][c]) for k in 'xyz')
            r = dict(key, cand=int(ni), wct_version=wv, br_filled=int(a['wc.T_tagger.br_filled'][i][c]), nue_score=float(a['wc.T_tagger.nue_score'][i][c]),
                     numu_score=float(a['wc.T_tagger.numu_score'][i][c]), reco_x=rx, reco_y=ry, reco_z=rz,
                     partner='none', iou_q=np.nan, iou_n=np.nan, mutual=-1, nmatch=0, m_nu_idx=-1, m_pdg=0, m_ccnc=-1, m_edep_mev=np.nan, m_vx=np.nan, m_vy=np.nan, m_vz=np.nan, dist_sce=np.nan)
            r['kine_reco_Enu'] = float(enu[c]) if c < len(enu) else np.nan
            for n in TAG: r[n] = float(a['wc.T_tagger.' + n][i][c])
            j = reco_rows.get(int(ni))
            if j is not None:
                r['nmatch'] = int(a['wc.T_match.nmatch'][i][j]); r['mutual'] = int(a['wc.T_match.mutual'][i][j])
                if r['nmatch'] > 0:
                    po = int(a['wc.T_match.match_obj'][i][int(a['wc.T_match.match_obj..idx'][i][j])])
                    r['iou_q'] = float(a['wc.T_match.match_iou_q'][i][int(a['wc.T_match.match_iou_q..idx'][i][j])])
                    r['iou_n'] = float(a['wc.T_match.match_iou_n'][i][int(a['wc.T_match.match_iou_n..idx'][i][j])])
                    if po in tr:
                        tni = int(a['wc.T_match.truth_nu_idx'][i][tr[po]])
                        if tni >= 0 and tni in truth_by_idx:
                            q = truth_by_idx[tni]; r.update(partner='neutrino', m_nu_idx=tni, m_pdg=int(tn['pdg'][q]), m_ccnc=int(tn['ccnc'][q]), m_edep_mev=1000 * float(tn['edep'][q]),
                                                           m_vx=float(tn['vtx_x'][q]), m_vy=float(tn['vtx_y'][q]), m_vz=float(tn['vtx_z'][q]))
                            if sce is not None and q < len(sce): r['dist_sce'] = float(np.sqrt(((sce[q] - [rx, ry, rz]) ** 2).sum()))
                        else: r['partner'] = 'cosmic'
                    else: r['partner'] = 'partner_row_missing'
            else: r['partner'] = 'no_level2_row'
            cands.append(r)
    return fi, caf, 'OK', cands, truth

if __name__ == '__main__':
    name, lst, nf, nproc, outd = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    os.makedirs(outd, exist_ok=True); files = [l.strip() for l in open(lst) if l.strip()][:nf]; t0 = time.time()
    with mp.Pool(nproc) as p: res = p.map(one, [(i, f, name) for i, f in enumerate(files)], chunksize=4)
    C = pd.DataFrame([r for _, _, st, cs, _ in res for r in cs]); T = pd.DataFrame([r for _, _, st, _, ts in res for r in ts])
    C.to_parquet(os.path.join(outd, f'cands_{name}.parquet')); T.to_parquet(os.path.join(outd, f'truth_{name}.parquet'))
    fails = [(f, st) for _, f, st, _, _ in res if st != 'OK']
    json.dump(dict(campaign=name, files=len(files), read_fail=len(fails), fails=fails, events=int(T[['file','run','subrun','event']].drop_duplicates().shape[0]) if len(T) else 0,
                   candidates=len(C), wall_s=round(time.time() - t0, 1)), open(os.path.join(outd, f'build_{name}.json'), 'w'), indent=1)
    print(name, 'files', len(files), 'fails', len(fails), 'events', int(T[['file','run','subrun','event']].drop_duplicates().shape[0]) if len(T) else 0, 'cands', len(C), 'wall', round(time.time() - t0, 1))
