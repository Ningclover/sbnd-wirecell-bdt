"""spec_v1: the label, the fiducial volume and the split, as code. Both servers import this file.
Inputs are plain numpy arrays or pandas columns with the names of spec_v1.md section 4 (per-candidate table).
Nothing here reads a ROOT file; the table builders do that and call these functions.
"""
import numpy as np

SPEC_VERSION = "v1"
VTX_CUT_CM = 5.0          # reco vertex vs SCE-shifted true vertex of the matched neutrino
EDEP_MIN_MEV = 75.0       # deposited energy of the true neutrino, in MeV (CAF stores GeV: multiply by 1000)
SENTINEL = -15.0          # nue_score when br_filled == 0 (the model never ran)

# Fiducial volume: the collaboration's nu_e FV (Pandora/SPINE nueCC FV, SBND technote eq. 6; Matteo's cuts.FV),
# decided 2026-11-11: the box below minus the cathode band, minus the cut-out of the top downstream
# corner of the East TPC. Used unchanged for the training label, the pre-cut and the evaluation.
FV = dict(x=(-190.0, 190.0), y=(-190.0, 190.0), z=(10.0, 450.0), cathode_gap=5.0)
FV_CUTOUT = dict(x_max=0.0, y_min=100.0, z_min=250.0)   # a vertex is in the cut-out when x < 0 and y > 100 and z > 250

def in_fv(x, y, z):
    """True inside the FV. Same function for the true vertex (signal) and the reco vertex (pre-cut); same for data and MC.
    Closed box bounds as in Matteo's cuts.in_box; NaN is outside."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    box = ((x >= FV['x'][0]) & (x <= FV['x'][1]) & (y >= FV['y'][0]) & (y <= FV['y'][1]) &
           (z >= FV['z'][0]) & (z <= FV['z'][1]) & (np.abs(x) > FV['cathode_gap']))
    cut = (x < FV_CUTOUT['x_max']) & (y > FV_CUTOUT['y_min']) & (z > FV_CUTOUT['z_min'])
    return box & ~cut

def in_wcgpu1_box(x, y, z):
    """The small-sample study's box, reporting only (round 0 reproduces wcgpu with it)."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    return ((x > -191.05) & (x < 191.05) & (y > -189.31) & (y < 189.31) & (z > 10.85) & (z < 450.15) & (np.abs(x) > 5.0))

# Reporting-only second box: the toolkit tagger box (reference note 135), never used for the label.
def in_toolkit_box(x, y, z):
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    return (np.abs(x) < 198.55) & (np.abs(y) < 196.312) & (z > 3.85) & (z < 495.15)

MATCH_BRANCHES = ['level', 'side', 'obj', 'nu_index', 'truth_nu_idx', 'truth_trackid', 'nmatch', 'mutual', 'match_obj', 'match_iou_n', 'match_iou_q',
                  'match_shared_n', 'match_shared_q']   # the wc.T_match branches the label reads; any one unreadable -> label_unreadable for the file

def placeholder_mask(kine_reco_Enu, neutrino_type, reco_x, reco_y, reco_z):
    """Empty T_tagger rows that are not candidates (Matteo's wcreco.py): Enu 0, neutrino_type 0, vertex at the origin.
    They never pass br_filled or the FV; drop them before counting 'candidate exists'."""
    return ((np.asarray(kine_reco_Enu, float) == 0.0) & (np.asarray(neutrino_type, float) == 0.0)
            & (np.asarray(reco_x, float) == 0.0) & (np.asarray(reco_y, float) == 0.0) & (np.asarray(reco_z, float) == 0.0))

def partner_class(partner_kind, label_unreadable):
    """partner_kind per candidate: 'neutrino' (match_obj[0] at level 2 is a truth row with truth_nu_idx >= 0),
    'cosmic' (that row has truth_nu_idx < 0), 'none' (nmatch == 0 or no level-2 reco row), 'missing' (partner row not found).
    Returns the class used by the label: 'neutrino', 'cosmic', or 'unknown' (label unreadable)."""
    pk = np.asarray(partner_kind, dtype=object); lu = np.asarray(label_unreadable, bool)
    out = np.where(pk == 'neutrino', 'neutrino', 'cosmic').astype(object)
    out[lu] = 'unknown'
    return out

def pool_mask(br_filled, reco_x, reco_y, reco_z):
    """Rows that enter training and the purity denominator: the model ran, and the reco vertex is in the FV."""
    return (np.asarray(br_filled) == 1) & in_fv(reco_x, reco_y, reco_z)

def signal_mask(partner_kind, label_unreadable, dist_sce_cm, m_pdg, m_ccnc, m_vx, m_vy, m_vz, m_edep_mev):
    """Signal: partner is a neutrino, reco vertex within VTX_CUT_CM of its SCE-shifted true vertex (distance finite),
    |pdg| == 12 and CC, true vertex (raw, no SCE) in the FV, deposited energy > EDEP_MIN_MEV."""
    cls = partner_class(partner_kind, label_unreadable)
    d = np.asarray(dist_sce_cm, float)
    vtx_ok = np.isfinite(d) & (d < VTX_CUT_CM)
    nue_cc = (np.abs(np.asarray(m_pdg)) == 12) & (np.asarray(m_ccnc) == 0)
    return (cls == 'neutrino') & vtx_ok & nue_cc & in_fv(m_vx, m_vy, m_vz) & (np.asarray(m_edep_mev, float) > EDEP_MIN_MEV)

def background_class(partner_kind, label_unreadable, dist_sce_cm, m_pdg, m_ccnc):
    """For pool rows that are not signal. Same classes as the wcgpu study."""
    cls = partner_class(partner_kind, label_unreadable)
    d = np.asarray(dist_sce_cm, float); vtx_ok = np.isfinite(d) & (d < VTX_CUT_CM)
    pdg = np.abs(np.asarray(m_pdg)); ccnc = np.asarray(m_ccnc)
    out = np.full(len(cls), 'other', dtype=object)
    out[(cls == 'neutrino') & (pdg == 12) & (ccnc == 0) & ~vtx_ok] = 'nue_cc_bad_vertex'
    out[(cls == 'neutrino') & (pdg == 12) & (ccnc == 0) & vtx_ok] = 'nue_cc_outside_fv_or_low_edep'
    out[(cls == 'neutrino') & (pdg == 14) & (ccnc == 0)] = 'numu_cc'
    out[(cls == 'neutrino') & (ccnc == 1)] = 'nc'
    out[cls == 'cosmic'] = 'cosmic'
    out[cls == 'unknown'] = 'label_unreadable'
    return out

ACTIVE = dict(x=(-201.3, 201.3), y=(-203.73, 203.73), z=(0.0, 501.0))   # GDML volTPCActive, Matteo's cuts.ACTIVE

def in_active(x, y, z):
    """Vertex inside the active volume, bounds included (Matteo's cuts.in_active)."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    return (x >= ACTIVE['x'][0]) & (x <= ACTIVE['x'][1]) & (y >= ACTIVE['y'][0]) & (y <= ACTIVE['y'][1]) & (z >= ACTIVE['z'][0]) & (z <= ACTIVE['z'][1])

def matchable_mask(edep_mev, vx, vy, vz):
    """The neutrinos the evaluation loops start from (Matteo's cuts.MATCHABLE): visible and in the active volume."""
    return (np.asarray(edep_mev, float) > 0) & in_active(vx, vy, vz)

def true_signal_e500_mask(pdg, ccnc, vx, vy, vz, true_lep_pdg, true_lep_ke_mev):
    """The e500 signal of the evaluation standard: nu_e/anti-nu_e CC, true vertex in FV, true electron KE > 500 MeV, no edep cut."""
    return ((np.abs(np.asarray(pdg)) == 12) & (np.asarray(ccnc) == 0) & in_fv(vx, vy, vz)
            & (np.abs(np.asarray(true_lep_pdg, float)) == 11) & (np.asarray(true_lep_ke_mev, float) > 500.0))

def true_signal_mask(pdg, ccnc, vx, vy, vz, edep_mev):
    """Efficiency denominator, a pure truth count over T_truth_nu rows: nu_e/anti-nu_e CC, true vertex in FV, > 75 MeV."""
    return (np.abs(np.asarray(pdg)) == 12) & (np.asarray(ccnc) == 0) & in_fv(vx, vy, vz) & (np.asarray(edep_mev, float) > EDEP_MIN_MEV)

def split_v11(sample, run, subrun):
    """spec_v1.1 (2026-11-18, user decision): on Aurora every row trains except a fixed development slice used only to
    choose settings (early stopping, depth, weights). 'dev' = subrun % 10 == 0 for MC; beam-off training half (all subrun 1,
    any sample whose name contains 'beam-off-train'): run % 10 == 1. Everything else 'train'. Evaluation happens only on the Fermilab (fg1) production at gpvm."""
    sample = np.asarray(sample, dtype=object); run = np.asarray(run, int); subrun = np.asarray(subrun, int)
    out = np.where(subrun % 10 == 0, 'dev', 'train').astype(object)
    bo = np.char.find(sample.astype(str), 'beam-off-train') >= 0   # fg1-beam-off-train, wcl1116-beam-off-train (the same odd-run files)
    out[bo] = np.where(run[bo] % 10 == 1, 'dev', 'train')
    return out

def split(sample, run, subrun):
    """'train' or 'test'. MC: even subrun = train, odd = test (per sample). Beam-off data: all of the odd-run training half
    is training input only; its own train/test split is by run (run % 4 in {1} -> test, else train), since every file has subrun 1."""
    sample = np.asarray(sample, dtype=object); run = np.asarray(run, int); subrun = np.asarray(subrun, int)
    out = np.where(subrun % 2 == 0, 'train', 'test').astype(object)
    bo = np.char.find(sample.astype(str), 'beam-off-train') >= 0   # fg1-beam-off-train, wcl1116-beam-off-train (the same odd-run files)
    out[bo] = np.where(run[bo] % 4 == 1, 'test', 'train')
    return out

# The nu_e CC filter volume of the filtered simulation (M. Vicenzi's plots.NUECC_FILTER, inferred from the filtered sample's
# truth): the active volume with x widened. The same numbers as evaluation/eval_standard.py's FILTER.
NUECC_FILTER = dict(x=(-202.3, 202.3), y=(-203.73, 203.73), z=(0.0, 501.0))
SPILL_KEY = ['file_id', 'run', 'subrun', 'event']

def filtered_spills(truth):
    """The spills (file_id, run, subrun, event) of a BNB truth table that hold a nu_e / anti-nu_e CC vertex inside the
    filter volume. The nu_e-filtered sample stands for exactly these spills with its own, much larger exposure, so they are
    dropped from the BNB sample whole (cosmics and any other neutrino in them included) before the two are combined; else
    those events are counted twice (user, 2026-11-19, from round 4 on; the evaluation has always done this)."""
    hit = ((np.abs(truth['pdg']) == 12) & (truth['ccnc'] == 0) & truth['vx'].between(*NUECC_FILTER['x'])
           & truth['vy'].between(*NUECC_FILTER['y']) & truth['vz'].between(*NUECC_FILTER['z']))
    return truth.loc[hit, SPILL_KEY].drop_duplicates()

def drop_filtered_spills(C, truth):
    """C without the rows whose spill is in filtered_spills(truth). Returns (kept rows, number of spills dropped)."""
    sp = filtered_spills(truth)
    if not len(sp): return C, 0
    k = C[SPILL_KEY].astype({'file_id': str}).merge(sp.astype({'file_id': str}).assign(_hit=1), on=SPILL_KEY, how='left')['_hit'].isna().to_numpy()
    return C[k], int(len(sp))
