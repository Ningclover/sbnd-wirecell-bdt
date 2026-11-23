#!/usr/bin/env python3
"""Check a νμ release's Wire-Cell weight files against the release's own xgboost models, on table rows.

The XML side is read with a minimal TMVA gradient-BDT evaluator written here from TMVA's node
convention (cType=1: right if x >= cut), and follows the chain (UbooneNumuBDTScorer.cxx):
small model v = tanh(sum) per object, the chain's protections (numu_1 dQ_dx_cut inf -> 10,
cosmict_10 angle_beam NaN -> 0), max (numu_1, numu_2) or min (cosmict_10) over the candidate's
objects, the empty constant when it has none; final model with NaN -> 0 on the four protected
inputs, numu_score = float32(log10((1+v)/(1-v))). The xgboost side is score_numu_release.py.

  python check_numu_chain_xml.py RELEASE_DIR TABLE_DIR SAMPLE [--n 20000] [--weights RELEASE_DIR/wirecell]
Prints, per small model and for numu_score, the largest difference and the pass/fail agreement at the cut.
"""
import argparse, json, math, os, sys
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd, xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from make_numu_chain_weights import CHAIN_NAMES, FINAL_XML  # noqa: E402
KEY = ['file_id', 'run', 'subrun', 'event', 'cand']


class TmvaGrad:
    def __init__(self, path):
        r = ET.parse(path).getroot()
        self.names = [v.get('Expression') for v in r.iter('Variable')]
        self.trees = []
        for t in r.iter('BinaryTree'):
            iv, cut, res, kid = [], [], [], []
            def walk(n):
                i = len(iv); iv.append(int(n.get('IVar'))); cut.append(float(n.get('Cut'))); res.append(float(n.get('res'))); kid.append([-1, -1])
                if iv[i] >= 0:
                    ch = {c.get('pos'): c for c in n.findall('Node')}
                    kid[i][0] = walk(ch['l']); kid[i][1] = walk(ch['r'])
                return i
            walk(t.find('Node'))
            self.trees.append((np.array(iv), np.array(cut, np.float32), np.array(res), np.array(kid)))

    def sum(self, X):
        X = X.astype(np.float32); F = np.zeros(len(X)); rows = np.arange(len(X))
        for iv, cut, res, kid in self.trees:
            node = np.zeros(len(X), int)
            while True:
                inner = iv[node] >= 0
                if not inner.any(): break
                r = rows[inner]; n = node[inner]
                right = X[r, iv[n]] >= cut[n]
                node[inner] = np.where(right, kid[n, 1], kid[n, 0])
            F += res[node]
        return F


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('release'); ap.add_argument('tables'); ap.add_argument('sample')
    ap.add_argument('--n', type=int, default=20000); ap.add_argument('--weights')
    a = ap.parse_args(); man = json.load(open(os.path.join(a.release, 'manifest.json')))
    W = os.path.join(a.weights or os.path.join(a.release, 'wirecell'), 'uboone', 'weights')
    names = man['inputs']
    C = pd.read_parquet(os.path.join(a.tables, 'cands_%s.parquet' % a.sample), columns=list(dict.fromkeys(KEY + names)))
    C = C.sample(n=min(a.n, len(C)), random_state=1).reset_index(drop=True)
    for n in man['nan_to_0']: C[n] = C[n].fillna(0.0)
    report = {}
    for m, t in man['level1']['models'].items():
        O = pd.read_parquet(os.path.join(a.tables, 'objs_%s_%s.parquet' % (m, a.sample)), columns=KEY + t['inputs'])
        O = O.merge(C[KEY], on=KEY)
        for c, (kind, v) in t['protect'].items():
            x = O[c].to_numpy(float); O[c] = np.where(np.isinf(x) if kind == 'inf' else np.isnan(x), v, x)
        X = O[t['inputs']].to_numpy(np.float32)
        b = xgb.Booster(); b.load_model(os.path.join(a.release, t['file']))
        O['ref'] = b.predict(xgb.DMatrix(X, feature_names=t['inputs']), output_margin=True).astype(np.float64)
        T = TmvaGrad(os.path.join(W, CHAIN_NAMES[m])); assert T.names == t['inputs'], m
        O['xml'] = np.tanh(T.sum(X))
        g = O.groupby(KEY, sort=False).agg(ref=('ref', t['pool']), xml=('xml', t['pool'])).reset_index()
        C = C.merge(g, on=KEY, how='left')
        ref = np.where(C.ref.isna(), t['empty_value'], np.tanh(t['scale'] * C.ref + t['offset'])).astype(np.float32)
        new = np.where(C.xml.isna(), t['empty_value'], C.xml).astype(np.float32)
        report[m] = dict(objects=len(O), candidates_with_objects=int(C.ref.notna().sum()), max_abs_diff=float(np.abs(ref - new).max()),
                         below_empty=int((C.xml.notna() & (new <= t['empty_value'])).sum()))
        C['ref_' + m] = ref; C['xml_' + m] = new; C = C.drop(columns=['ref', 'xml'])
    k = 2 / math.log(10); cut = man['operating_point']['cut_margin']
    Xr = C[names].copy(); Xx = C[names].copy()
    for m in man['level1']['models']:
        Xr[m + '_score'] = C['ref_' + m]; Xx[m + '_score'] = C['xml_' + m]
    b = xgb.Booster(); b.load_model(os.path.join(a.release, man['models']['d6_f0']['file']))
    margin = b.predict(xgb.DMatrix(Xr.to_numpy(np.float32), feature_names=names), output_margin=True).astype(np.float64)
    T = TmvaGrad(os.path.join(W, FINAL_XML)); assert T.names == names
    v = np.tanh(T.sum(Xx.to_numpy(np.float32)))
    score = np.log10((1 + v) / (1 - v)).astype(np.float32)
    ok = np.abs(v) < 1
    report['numu_score'] = dict(candidates=len(C), saturated=int((~ok).sum()),
                                max_abs_diff=float(np.abs(score[ok] - k * margin[ok]).max()),
                                pass_ref=int((margin > cut).sum()), pass_xml=int((score > k * cut).sum()),
                                pass_disagree=int(((margin > cut) != (score > k * cut)).sum()))
    print(json.dumps(dict(sample=a.sample, release=os.path.basename(a.release.rstrip('/')), checks=report), indent=1))


if __name__ == '__main__':
    main()
