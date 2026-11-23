#!/usr/bin/env python3
"""Write a frozen νμ release as the weight files the Wire-Cell chain already reads.

The SBND configuration (`cfg/pgrapher/experiment/sbnd/clus.jsonnet`, UbooneNumuBDTScorer) books
four νμ files from `uboone/weights` on WIRECELL_PATH. This writes the ones the release retrains,
under their shipped names, so putting OUTDIR in front of `wire-cell-data` swaps the models with no
change to the toolkit or the configuration:

  numu_1      -> numu_tagger1.weights.xml       TMVA::Reader, max over the vertex segments, -0.4 if none
  numu_2      -> numu_tagger2.weights.xml       TMVA::Reader, max over the objects, -0.1 if none
  cosmict_10  -> cos_tagger_10.weights.xml      TMVA::Reader, min over the objects, 0.7 if none
  final       -> numu_scalars_scores_0923.xml   numu_score = log10((1+v)/(1-v))

numu_3 is not retrained (its file stays the shipped one).

Small models: the chain stores v = tanh(TMVA sum). Each file is written with the release's own scale
and offset (manifest `level1.models.<m>`), so the chain stores float32(tanh(scale * margin + offset)),
the value the final model was trained on; the empty constant lies below every value a real object
can give. Final model: written unscaled, so numu_score = (2 / ln 10) * margin = 0.868589 * margin,
the shipped scale (as round 1).

Usage: make_numu_chain_weights.py RELEASE_DIR OUTDIR [--model d6_f0]
"""
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'release', 'wirecell'))
from xgb_json_forest import XgbJsonForest          # noqa: E402
from xgb_to_tmva import convert                    # noqa: E402

CHAIN_NAMES = {'numu_1': 'numu_tagger1.weights.xml', 'numu_2': 'numu_tagger2.weights.xml',
               'cosmict_10': 'cos_tagger_10.weights.xml'}
FINAL_XML = 'numu_scalars_scores_0923.xml'


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('release'); ap.add_argument('outdir'); ap.add_argument('--model', default='d6_f0')
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.release, 'manifest.json')))
    wdir = os.path.join(a.outdir, 'uboone', 'weights'); os.makedirs(wdir, exist_ok=True)
    written = {}
    for m, t in (man.get('level1') or {}).get('models', {}).items():
        forest = XgbJsonForest(os.path.join(a.release, t['file']))
        if forest.names != t['inputs']:
            sys.exit('%s: model inputs %s differ from the manifest %s' % (m, forest.names, t['inputs']))
        xml = os.path.join(wdir, CHAIN_NAMES[m])
        convert(forest, xml, t['scale'], t['offset'])
        written[m] = dict(xml=CHAIN_NAMES[m], trees=forest.ntrees, inputs=t['inputs'], scale=t['scale'],
                          offset=t['offset'], empty_value=t['empty_value'], pool=t['pool'], sha256=sha(xml))
        print('%-10s -> %-28s %4d trees, scale %.6g offset %.6g' % (m, CHAIN_NAMES[m], forest.ntrees, t['scale'], t['offset']))
    l2 = man['models'][a.model]
    forest = XgbJsonForest(os.path.join(a.release, l2['file']))
    if forest.names != man['inputs']:
        sys.exit('final model inputs differ from the manifest list')
    xml = os.path.join(wdir, FINAL_XML)
    convert(forest, xml, 1.0, 0.0)
    written['final'] = dict(xml=FINAL_XML, model=a.model, trees=forest.ntrees, inputs=len(man['inputs']),
                            scale=1.0, offset=0.0, sha256=sha(xml))
    print('%-10s -> %-28s %4d trees, numu_score = 0.868589 * margin' % ('final', FINAL_XML, forest.ntrees))
    json.dump(dict(release=os.path.basename(a.release.rstrip('/')), files=written),
              open(os.path.join(a.outdir, 'chain_weights.json'), 'w'), indent=1)
    print('wrote %s (put %s in front of wire-cell-data on WIRECELL_PATH)' % (wdir, a.outdir))


if __name__ == '__main__':
    main()
