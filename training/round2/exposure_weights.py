"""Training weights at the BNB POT from the merged tables' files_<sample>.parquet, by the round 2 / round 3 recipe (PLAN.md,
"Round 2 training weights", user 2026-11-19):
  simulation:  POT_bnb / POT_sample, POT from the TotalPOT histogram (pot_total_hist), the header undercounting Aurora's
               filtered samples; BNB = 1.
  beam-off:    (1 - f) * (POT_bnb / POT_per_spill) / G_off, G_off = the header off-beam gate count of the files read.
Only files read without error count. Prints a JSON record and the --weights argument for train_l1.py / train_l2.py.
  python exposure_weights.py MERGED_DIR --bnb S --mc S [S ...] --offbeam S [--f 0.0725] [--pot-per-spill 4.81e12] [--out w.json]
"""
import argparse, json, os
import pandas as pd

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('merged_dir'); ap.add_argument('--bnb', required=True); ap.add_argument('--mc', nargs='*', default=[])
    ap.add_argument('--offbeam'); ap.add_argument('--f', type=float, default=0.0725); ap.add_argument('--pot-per-spill', type=float, default=4.81e12); ap.add_argument('--out')
    a = ap.parse_args()
    def files(s):
        F = pd.read_parquet(os.path.join(a.merged_dir, 'files_%s.parquet' % s)); return F[F.error.fillna('') == '']
    pot = {s: float(files(s).pot_total_hist.sum()) for s in [a.bnb] + a.mc}
    W = {a.bnb: 1.0}; W.update({s: pot[a.bnb] / pot[s] for s in a.mc})
    rec = dict(pot_total_hist=pot, f=a.f, pot_per_spill=a.pot_per_spill)
    if a.offbeam:
        g = float(files(a.offbeam).n_offbeam_gates.sum()); W[a.offbeam] = (1 - a.f) * (pot[a.bnb] / a.pot_per_spill) / g
        rec.update(offbeam_gates=g, offbeam_files=int(len(files(a.offbeam))))
    rec['weights'] = W; rec['weights_arg'] = ' '.join('%s=%.6g' % kv for kv in W.items())
    print(json.dumps(rec, indent=1))
    if a.out: json.dump(rec, open(a.out, 'w'), indent=1)

if __name__ == '__main__':
    main()
