#!/usr/bin/env python3
"""Write a frozen release as the weight files the Wire-Cell chain already reads.

The chain names its weight files relative to `uboone/weights` and finds them on WIRECELL_PATH
(`cfg/pgrapher/experiment/sbnd/clus.jsonnet`, `bdt_weights_dir`).  This writes a directory with
that same layout holding only the 16 files of the release, so putting it in front of
`wire-cell-data` on WIRECELL_PATH swaps the models with no change to the toolkit and no change to
the configuration.

The level-1 scale and offset are taken from the release manifest (`level1.chain_transform`), which
is the table the final model was retrained against, not recomputed here.  The final model is
written with scale = ln(10)/2, so the chain's `nue_score` holds the raw margin.

Usage: make_chain_weights.py RELEASE_DIR OUTDIR [--variant main]
"""
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from xgb_json_forest import XgbJsonForest          # noqa: E402
from xgb_to_tmva import convert, LN10_2            # noqa: E402

FINAL_XML = 'XGB_nue_seed2_0923.xml'               # the shipped name of the final model


def pad_to(forest, target):
    """Re-index a forest onto `target`, a longer input list that contains all of its names.

    The chain registers one fixed list of inputs per model (`add_xgb_var`), and TmvaGradForest
    refuses a file that declares a different one.  A variant trained without an input (the νe
    `nogap` model drops `gap_n_bad`) therefore has to declare it anyway; no node tests it, so the
    forest is unchanged.
    """
    pos = {n: i for i, n in enumerate(target)}
    missing = [n for n in forest.names if n not in pos]
    if missing:
        raise ValueError('inputs not in the chain list: %s' % ', '.join(missing))
    remap = [pos[n] for n in forest.names]
    added = [n for n in target if n not in set(forest.names)]
    for t in forest.trees:
        t['ivar'] = [remap[v] if v >= 0 else v for v in t['ivar']]
    forest.names = list(target)
    return added


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('release')
    ap.add_argument('outdir')
    ap.add_argument('--variant', default='main')
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.release, 'manifest.json')))
    T = man['level1'].get('chain_transform')
    if not T:
        sys.exit('%s has no level1.chain_transform: not a chain release' % a.release)
    wdir = os.path.join(a.outdir, 'uboone', 'weights')
    os.makedirs(wdir, exist_ok=True)
    written = {}
    for m, md in man['level1']['models'].items():
        t = T['models'][m]
        forest = XgbJsonForest(os.path.join(a.release, md['file']))
        xml = os.path.join(wdir, '%s_BDT.weights.xml' % m)
        convert(forest, xml, t['scale'], t['offset'])
        written[m] = dict(xml=os.path.basename(xml), trees=forest.ntrees, inputs=forest.nvars,
                          scale=t['scale'], offset=t['offset'], empty_value=t['empty_value'],
                          mb=round(os.path.getsize(xml) / 1e6, 2))
        print('%-10s -> %-28s %4d trees %6.2f MB  stored range [%.4f, %.4f], empty %.2f'
              % (m, os.path.basename(xml), forest.ntrees, written[m]['mb'],
                 t['value_min'], t['value_max'], t['empty_value']))
    l2 = man['level2'][a.variant]
    forest = XgbJsonForest(os.path.join(a.release, l2['file']))
    xml = os.path.join(wdir, FINAL_XML)
    # the chain registers the pooled level-1 values under their shipped branch names
    # (`br3_3_score`, ...), and TmvaGradForest refuses a file whose variable names differ from
    # the caller's (`TmvaGradForest.cxx:191`, "variable N mismatch"), so rename them here.
    ren = {md['level2_input_name']: md['shipped_score_name'] for md in man['level1']['models'].values()}
    # the chain registers one fixed input list; a variant trained without one of them has to
    # declare it anyway, so pad onto the main variant's list (which is that registration order)
    target = [ren.get(n, n) for n in man['level2']['main']['inputs']]
    forest.names = [ren.get(n, n) for n in forest.names]
    added = pad_to(forest, target)
    convert(forest, xml, LN10_2, 0.0, names=target)
    written['level2'] = dict(xml=FINAL_XML, variant=a.variant, trees=forest.ntrees,
                             inputs_declared=len(target), declared_but_unused=added,
                             inputs=forest.nvars, scale=LN10_2, offset=0.0,
                             mb=round(os.path.getsize(xml) / 1e6, 2))
    print('%-10s -> %-28s %4d trees %6.2f MB  nue_score = raw margin (scale ln(10)/2)'
          % ('level2/' + a.variant, FINAL_XML, forest.ntrees, written['level2']['mb']))
    meta = dict(release=os.path.basename(a.release.rstrip('/')), variant=a.variant,
                frozen_utc=man.get('frozen_utc'), files=written)
    json.dump(meta, open(os.path.join(a.outdir, 'chain_weights.json'), 'w'), indent=1)
    print('\nwrote %s  (put %s in front of wire-cell-data on WIRECELL_PATH)' % (wdir, a.outdir))
    return 0


if __name__ == '__main__':
    sys.exit(main())
