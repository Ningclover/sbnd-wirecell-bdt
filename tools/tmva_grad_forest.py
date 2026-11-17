#!/usr/bin/env python3
"""Pure-numpy reader of a TMVA "BDT::BDT" gradient-boosted weight file (no ROOT).

Re-scores the shipped SBND/MicroBooNE nu_e final model (XGB_nue_seed2_0923.xml, 600 trees,
258 inputs) with the same arithmetic as the Wire-Cell toolkit (root/src/TmvaGradForest.cxx and
UbooneNueBDTScorer::cal_bdts_xgboost, branch apply-pointcloud):

  * inputs are float32; a node cut is read as double then narrowed to float32
  * at a node: right = (x[IVar] >= Cut), inverted when cType == 0; descend until nType != 0
  * sum = sum over trees (double) of the leaf `res` (float32)
  * v = 2 / (1 + exp(-2 sum)) - 1;  v = -999 if any input is NaN (TMVA::Reader)
  * nue_score = log10((1+v)/(1-v)), with |v| >= 1 or non-finite v pulled to +-nextafter(1, 0)
    (so a NaN input gives -16.2556, not -999), stored as float32
  * nue_score = -15 when br_filled == 0 (the model is never run)

The file is refused, as in the C++, if it has input transformations, BoostType != Grad,
preselection, Fisher-cut nodes, AnalysisType != 1, or variable names that differ from the
caller's list in order.

Python use:
    f = TmvaGradForest('XGB_nue_seed2_0923.xml')        # f.names = the 258 inputs in order
    score = f.nue_score(X, br_filled)                    # X: (n, 258), columns in f.names order

Command line, one merged CAF (inputs at wc.T_tagger.<name>, kine_reco_Enu at wc.T_kine.):
    python3 tmva_grad_forest.py XGB_nue_seed2_0923.xml caf.root
  prints how many candidates reproduce the stored wc.T_tagger.nue_score bit for bit.
  --layout tracking-pr reads a per-event tracking-pr.root (T_tagger + T_kine trees) instead.

Needs numpy; the command line also needs uproot.  Parsing the 199 MB file takes ~10-20 s.
"""
import argparse
import math
import re
import sys

import numpy as np

SENTINEL = -15.0
_TAG = re.compile(rb'<(/?)(BinaryTree|Node)\b([^>]*?)(/?)>')
_ATTR = re.compile(rb'(\w+)="([^"]*)"')


def _bool(s):
    """TMVA reads cType with std::stringstream >> bool: '1' -> true, '0' -> false."""
    return s.strip() not in (b'0', b'false')


class TmvaGradForest:
    def __init__(self, path, names=None):
        doc = open(path, 'rb').read()
        ms = doc.find(b'<MethodSetup')
        if ms < 0 or b'Method="BDT::BDT"' not in doc[ms:ms + 200]:
            raise ValueError('not a BDT::BDT weight file')
        head_end = doc.find(b'<Weights', ms)
        if head_end < 0:
            raise ValueError('no <Weights> element')
        head = doc[ms:head_end]
        nvar = int(re.search(rb'<Variables NVar="(\d+)"', head).group(1))
        self.names = [m.decode() for m in re.findall(rb'<Variable Expression="([^"]*)"', head)][:nvar]
        if len(self.names) != nvar:
            raise ValueError('fewer <Variable> elements than NVar')
        if names is not None and list(names) != self.names:
            raise ValueError('variable list differs from the file (names or order)')
        m = re.search(rb'<Transformations NTransformations="(\d+)"', head)
        if m and int(m.group(1)) != 0:
            raise ValueError('input-variable transformations present')
        opts = {k.decode(): v.decode().strip().lower()
                for k, v in re.findall(rb'<Option [^>]*?name="(\w+)"[^>]*>([^<]*)</Option>', head)}
        if opts.get('BoostType') != 'grad':
            raise ValueError(f"BoostType {opts.get('BoostType')!r} is not Grad")
        if opts.get('DoPreselection', 'false') not in ('false', '0'):
            raise ValueError('DoPreselection is on')
        wtag = doc[head_end:doc.find(b'>', head_end)]
        if b'PreselectionLowBkgVar0' in wtag:
            raise ValueError('preselection cuts present')
        wa = dict(_ATTR.findall(wtag))
        if int(wa.get(b'AnalysisType', wa.get(b'TreeType', b'-1'))) != 1:
            raise ValueError('Weights AnalysisType is not 1 (regression leaves)')
        ntrees = int(wa[b'NTrees'])
        wend = doc.find(b'</Weights>', head_end)

        cut, res, ivar, ntype, ctype, left, right, roots = [], [], [], [], [], [], [], []
        stack = []
        for t in _TAG.finditer(doc, head_end, wend):
            close, kind, attrs, selfclose = t.groups()
            if kind == b'BinaryTree':
                if close:
                    if stack:
                        raise ValueError('unclosed <Node> at </BinaryTree>')
                else:
                    roots.append(-1)
                continue
            if close:
                stack.pop()
                continue
            a = dict(_ATTR.findall(attrs))
            if int(a.get(b'NCoef', b'0')) != 0:
                raise ValueError('Fisher-cut node (NCoef != 0)')
            idx = len(cut)
            cut.append(float(a[b'Cut']))
            res.append(float(a[b'res']) if b'res' in a else -99.0)
            ivar.append(int(a[b'IVar']))
            ntype.append(int(a[b'nType']))
            ctype.append(_bool(a[b'cType']))
            left.append(-1)
            right.append(-1)
            if not stack:
                if roots[-1] >= 0:
                    raise ValueError('two root nodes in one tree')
                roots[-1] = idx
            elif a[b'pos'] == b'l':
                left[stack[-1]] = idx
            elif a[b'pos'] == b'r':
                right[stack[-1]] = idx
            else:
                raise ValueError('child node with pos neither l nor r')
            if not selfclose:
                stack.append(idx)
        if len(roots) != ntrees:
            raise ValueError(f'NTrees={ntrees} but {len(roots)} trees found')
        self.cut = np.array(cut, np.float64).astype(np.float32)   # strtod, then narrowed to float
        self.res = np.array(res, np.float64).astype(np.float32)
        self.ivar = np.array(ivar, np.int32)
        self.leaf = np.array(ntype, np.int32) != 0
        self.ctype = np.array(ctype, bool)
        self.left = np.array(left, np.int32)
        self.right = np.array(right, np.int32)
        self.roots = np.array(roots, np.int32)
        inner = ~self.leaf
        if (self.left[inner] < 0).any() or (self.right[inner] < 0).any():
            raise ValueError('intermediate node missing a child')
        if (self.ivar[inner] >= nvar).any():
            raise ValueError('node selects a variable beyond NVar')
        self.ivar[self.leaf] = 0                                  # unused on leaves; safe index

    def evaluate(self, X):
        """TMVA::Reader::EvaluateMVA for each row of X (n, nvars): v in (-1, 1), -999 on NaN."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != len(self.names):
            raise ValueError(f'X must be (n, {len(self.names)})')
        n = len(X)
        rows = np.arange(n)
        total = np.zeros(n, np.float64)
        for r in self.roots:
            cur = np.full(n, r, np.int32)
            live = ~self.leaf[cur]
            while live.any():
                c = cur[live]
                go_right = X[rows[live], self.ivar[c]] >= self.cut[c]
                go_right = np.where(self.ctype[c], go_right, ~go_right)
                cur[live] = np.where(go_right, self.right[c], self.left[c])
                live = ~self.leaf[cur]
            total += self.res[cur].astype(np.float64)
        v = 2.0 / (1.0 + np.exp(-2.0 * total)) - 1.0
        v[np.isnan(X).any(axis=1)] = -999.0
        return v

    @staticmethod
    def log_odds(v):
        """Toolkit bdt_log_odds_score: log10((1+v)/(1-v)) after pulling |v| >= 1 to +-nextafter(1,0)."""
        v = np.array(v, np.float64)
        sat = math.nextafter(1.0, 0.0)
        bad = ~np.isfinite(v) | (np.abs(v) >= 1.0)
        v[bad] = np.where(np.isfinite(v[bad]) & (v[bad] > 0), sat, -sat)
        return np.log10((1.0 + v) / (1.0 - v))

    def nue_score(self, X, br_filled):
        """The stored T_tagger.nue_score: float32, -15 where br_filled == 0."""
        s = self.log_odds(self.evaluate(X)).astype(np.float32)
        return np.where(np.asarray(br_filled) == 1, s, np.float32(SENTINEL)).astype(np.float32)


def _flat(a):
    a = np.asarray(a)
    if a.dtype == object:                       # jagged (one entry per event): concatenate
        return np.concatenate([np.asarray(x) for x in a]) if len(a) else np.array([])
    return a


def _read(path, names, layout):
    import uproot
    f = uproot.open(path)
    if layout == 'caf':
        t = f['recTreeWireCell']
        want = {n: ('wc.T_kine.' if n == 'kine_reco_Enu' else 'wc.T_tagger.') + n for n in names}
        want['br_filled'] = 'wc.T_tagger.br_filled'
        want['nue_score'] = 'wc.T_tagger.nue_score'
        arr = t.arrays(list(want.values()), library='np')
        cols = {k: _flat(arr[v]) for k, v in want.items()}
    else:                                        # per-event tracking-pr.root
        if 'T_tagger' not in f:                  # event with no neutrino candidate
            return (np.zeros((0, len(names)), np.float32), np.zeros(0, int), np.zeros(0, np.float32))
        tg = f['T_tagger']
        cols = {k: v for k, v in tg.arrays([n for n in names if n != 'kine_reco_Enu'] +
                                           ['br_filled', 'nue_score', 'nu_index'], library='np').items()}
        kin = f['T_kine'].arrays(['kine_reco_Enu', 'nu_index'], library='np')
        enu = dict(zip(kin['nu_index'].tolist(), kin['kine_reco_Enu'].tolist()))
        cols['kine_reco_Enu'] = np.array([enu.get(int(i), np.nan) for i in cols['nu_index']])
    X = np.column_stack([np.asarray(cols[n], np.float32) for n in names])
    return X, cols['br_filled'].astype(int), np.asarray(cols['nue_score'], np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('xml', help='shipped weight file, e.g. XGB_nue_seed2_0923.xml')
    ap.add_argument('files', nargs='+', help='merged CAF(s) or tracking-pr.root file(s)')
    ap.add_argument('--layout', choices=('caf', 'tracking-pr'), default='caf')
    a = ap.parse_args()
    f = TmvaGradForest(a.xml)
    print(f'{a.xml}: {len(f.roots)} trees, {len(f.cut)} nodes, {len(f.names)} inputs')
    Xs, bs, ss = zip(*[_read(p, f.names, a.layout) for p in a.files])
    X, br, stored = np.concatenate(Xs), np.concatenate(bs), np.concatenate(ss)
    ours = f.nue_score(X, br)
    same = ours == stored
    d = np.abs(ours.astype(np.float64) - stored)
    print(f'candidates {len(X)}  br_filled==1 {int((br == 1).sum())}  bit-identical {int(same.sum())}  '
          f'max |diff| {d.max() if len(d) else 0:.3g}  rows with a NaN input {int(np.isnan(X).any(1).sum())}')
    return 0 if same.all() else 1


if __name__ == '__main__':
    sys.exit(main())
