"""Independent check of the chain weight files: a minimal TMVA BDT (Grad) evaluator in numpy,
written from the TMVA node convention (cType=1: right if x >= cut), scored on the #29 bundle and
compared with Aurora's release scores (score_release.py at tag nue-round4-v2).
Usage: check_chain_xml.py BUNDLE CW_DIR"""
import sys, os, json, math
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd

def load(xml):
    r = ET.parse(xml).getroot()
    names = [v.get('Expression') for v in r.iter('Variable')]
    trees = []
    for t in r.iter('BinaryTree'):
        trees.append(t.find('Node'))
    return names, trees

def ev_node(n, X, idx, out):
    iv = int(n.get('IVar'))
    if iv < 0:
        out[idx] += float(n.get('res')); return
    cut = float(n.get('Cut')); ct = n.get('cType') == '1'
    kids = {c.get('pos'): c for c in n.findall('Node')}
    x = X[idx, iv]
    right = (x >= cut) if ct else ~(x >= cut)
    ev_node(kids['r'], X, idx[right], out); ev_node(kids['l'], X, idx[~right], out)

def score(xml, df):
    names, trees = load(xml)
    X = df[names].to_numpy(np.float32).astype(np.float64)
    F = np.zeros(len(df)); idx = np.arange(len(df))
    for t in trees: ev_node(t, X, idx, F)
    return F, names

B, CW = sys.argv[1], sys.argv[2]
arms = {'A_v2': 'nue_round4_ruleA_v2/main', 'A_nogap_v2': 'nue_round4_ruleA_v2/nogap', 'B_v2': 'nue_round4_ruleB_v2/main'}
M = ['br3_3','br3_5','br3_6','pio_2','stw_2','stw_3','stw_4','sig_1','sig_2','lol_1','lol_2','tro_1','tro_2','tro_4','tro_5']
K = ['file_id','run','subrun','event','cand']
res = {}
for s in ['wcl1116-nuecc', 'wcl1116-bnb']:
    C = pd.read_parquet(f'{B}/cands_{s}.parquet'); A = pd.read_parquet(f'{B}/aurora_scores_{s}.parquet')
    for arm, sub in arms.items():
        W = f'{CW}/{sub}/uboone/weights'; a = C[K].merge(A[A.model == arm], on=K, how='left')
        l1 = {}
        for m in M:
            O = pd.read_parquet(f'{B}/objs_{m}_{s}.parquet')
            F, _ = score(f'{W}/{m}_BDT.weights.xml', O)
            O = O.assign(v=np.tanh(F).astype(np.float32))
            mn = O.groupby(K)['v'].min().rename('mine').reset_index()
            j = a[K + ['l1_' + m]].merge(mn, on=K, how='left')
            have = j["mine"].notna()
            d = (j['mine'] - j['l1_' + m]).abs()[have]
            l1[m] = dict(n=int(have.sum()), max_abs=float(d.max()) if len(d) else 0.0, n_over_1e6=int((d > 1e-6).sum()))
        # final model on Aurora's level-1 values, under the chain's names
        X = C.copy()
        for m in M: X[m + '_score'] = a['l1_' + m].to_numpy()
        F, names = score(f'{W}/XGB_nue_seed2_0923.xml', X)
        v = np.tanh(F).astype(np.float64)
        nue = np.log10((1 + v) / (1 - v))
        d = np.abs(nue - a['score'].to_numpy())
        res[f'{s} {arm}'] = dict(level1=l1, final_declared=len(names), final_max_abs=float(d.max()),
                                 final_n_over_1e5=int((d > 1e-5).sum()), n=len(C))
        print(f'{s:15s} {arm:11s} l1 worst {max(x["max_abs"] for x in l1.values()):.2e} '
              f'(over 1e-6: {sum(x["n_over_1e6"] for x in l1.values())}, objects-present {sum(x["n"] for x in l1.values())})  '
              f'final worst {d.max():.2e} over 1e-5: {(d>1e-5).sum()} of {len(C)}  declared {len(names)}')
json.dump(res, open(os.path.join(CW, 'check_chain_xml.json'), 'w'), indent=1)
