#!/usr/bin/env python3
"""Write an xgboost JSON model as the TMVA weight file the Wire-Cell chain already reads.

The chain books every model with `TMVA::Reader` and, for a gradient-boosted file, evaluates
`v = tanh(sum of leaf responses)`. Nothing in that path has to change: this writes the same
kind of file the shipped final model is (`Creator="xgboost2TMVA"`, `BoostType=Grad`,
`AnalysisType="1"`), tree for tree and cut for cut.

The one freedom used here is that **every leaf may be scaled and the forest shifted**:

    TMVA sum = scale * (xgboost raw margin) + offset,   scale > 0

`tanh` is strictly increasing, so the chain's "minimum over the candidate's objects" is still
the minimum raw margin; and a float keeps the whole range when the output is placed where it
has resolution (a margin of +-10 written out unscaled saturates the float and the model is
lost). For a level-1 model the range is placed above the value the chain substitutes when a
candidate has no object, so that value becomes a marker no real object can produce. For the
final model `scale = ln(10)/2` makes the chain's stored `nue_score` the raw margin itself.

Usage:
  xgb_to_tmva.py model.json out.xml --scale 1.1512925                 # final model
  xgb_to_tmva.py model.json out.xml --range 0.35 0.99                 # level-1 model
Writes the XML and prints scale, offset and the resulting output range.
"""
import argparse, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from xgb_json_forest import XgbJsonForest  # noqa: E402

LN10_2 = math.log(10.0) / 2.0   # scale that makes log10((1+v)/(1-v)) the raw margin


def affine_for_range(forest, vlo, vhi):
    """scale and offset placing the forest's whole output between vlo and vhi."""
    mlo, mhi = forest.margin_bounds()
    slo, shi = math.atanh(vlo), math.atanh(vhi)
    scale = (shi - slo) / (mhi - mlo)
    return scale, slo - scale * mlo


def _node(out, t, i, depth, pos, scale, extra):
    """One <Node>, with its children nested inside, in TMVA's own spelling."""
    leaf = t['left'][i] < 0
    head = ('<Node Cut="%s" IVar="%d" NCoef="0" cType="1" depth="%d" nType="%d" pos="%s" '
            'purity="0.0e+00" res="%s" rms="0.0e+00"')
    if leaf:
        res = repr(float(np.float32(scale * float(t['cut'][i]) + extra)))
        out.append(head % ('0.0e+00', -1, depth, -99, pos, res) + ' />')
        return
    cut = repr(float(np.float32(t['cut'][i])))
    out.append(head % (cut, int(t['ivar'][i]), depth, 0, pos, '0.0e+00') + '>')
    _node(out, t, int(t['left'][i]), depth + 1, 'l', scale, extra)
    _node(out, t, int(t['right'][i]), depth + 1, 'r', scale, extra)
    out.append('</Node>')


def convert(forest, out_path, scale, offset, names=None):
    """Write the forest as a TMVA BDT::BDT gradient-boosted weight file."""
    names = list(names or forest.names)
    if len(names) != forest.nvars:
        raise ValueError('%d names for %d inputs' % (len(names), forest.nvars))
    # the whole shift, plus the model's own starting margin, goes on the first tree's leaves
    extra = offset + scale * forest.base_margin
    parts = ['<MethodSetup Method="BDT::BDT"><Variables NVar="%d">' % len(names)]
    for i, n in enumerate(names):
        parts.append('<Variable Expression="%s" Internal="%s" Label="%s" Max="0.0e+00" '
                     'Min="0.0e+00" Title="%s" Type="F" Unit="" VarIndex="%d" />' % (n, n, n, n, i))
    parts.append('</Variables><GeneralInfo><Info name="Creator" value="xgboost2TMVA" />'
                 '<Info name="AnalysisType" value="Classification" /></GeneralInfo>'
                 '<Options><Option modified="No" name="NodePurityLimit">5.00e-01</Option>'
                 '<Option modified="Yes" name="BoostType">Grad</Option></Options>'
                 '<Weights AnalysisType="1" NTrees="%d">' % forest.ntrees)
    for it, t in enumerate(forest.trees):
        parts.append('<BinaryTree boostWeight="1.0e+00" itree="%d" type="DecisionTree">' % it)
        _node(parts, t, 0, 0, 's', scale, extra if it == 0 else 0.0)
        parts.append('</BinaryTree>')
    parts.append('</Weights></MethodSetup>')
    with open(out_path, 'w') as f:
        f.write(''.join(parts))
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model')
    ap.add_argument('out')
    ap.add_argument('--scale', type=float, help='leaf scale; ln(10)/2 makes nue_score the raw margin')
    ap.add_argument('--offset', type=float, default=0.0)
    ap.add_argument('--range', nargs=2, type=float, metavar=('VLO', 'VHI'),
                    help='instead of --scale: place the whole output between these two values')
    a = ap.parse_args()
    forest = XgbJsonForest(a.model)
    if a.range:
        scale, offset = affine_for_range(forest, *a.range)
    else:
        scale, offset = (a.scale if a.scale is not None else LN10_2), a.offset
    convert(forest, a.out, scale, offset)
    mlo, mhi = forest.margin_bounds()
    print(f'{os.path.basename(a.model)}: {forest.ntrees} trees, {forest.nnodes} nodes, '
          f'margin [{mlo:.2f}, {mhi:.2f}] -> scale {scale:.8g} offset {offset:.8g} -> '
          f'output [{math.tanh(scale * mlo + offset):.6f}, {math.tanh(scale * mhi + offset):.6f}]  '
          f'{os.path.getsize(a.out) / 1e6:.1f} MB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
