"""Draw the fixed-purity comparison: the nueCC efficiency after all cuts of every model at one target purity (the other way
round from the fixed-efficiency table), from the scan file of the evaluation at Fermilab (gpvm's efficiency_at_purity<P>.json:
per model the interpolated efficiency at the target purity, the efficiencies where the purity curve is shifted by its own
error, and the scanned (cut, efficiency, purity) points).

  python evaluation/plot_efficiency_at_purity.py SCAN.json --out fig.png [--curves curves.png]
"""
import argparse, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('scan'); ap.add_argument('--out', default='efficiency_at_purity.png'); ap.add_argument('--curves'); ap.add_argument('--title', default=''); ap.add_argument('--drop', nargs='*', default=[], help='score columns (keys of the scan file) to leave out')
    a = ap.parse_args(); d = json.load(open(a.scan)); P = 100 * d['target_purity']; M = {k: v for k, v in d['models'].items() if k not in a.drop}
    labels = [m['label'] for m in M.values()]; eff = np.array([100 * m['efficiency'] for m in M.values()])
    lo = np.array([100 * m['efficiency_if_purity_minus_err'] for m in M.values()]); hi = np.array([100 * m['efficiency_if_purity_plus_err'] for m in M.values()])
    fig, ax = plt.subplots(figsize=(7.5, 3.8)); y = np.arange(len(labels))[::-1]
    cols = ['C%d' % i for i in range(len(labels))]
    ax.barh(y, eff, color=cols, alpha=0.85, height=0.6)
    ax.errorbar(eff, y, xerr=[eff - lo, hi - eff], fmt='none', ecolor='k', capsize=4, lw=1.2)
    for yi, e, l, h in zip(y, eff, lo, hi): ax.text(e + 0.3, yi, '%.1f %%  (%.1f to %.1f)' % (e, l, h), va='center', fontsize=9)
    ref = eff[0]; ax.axvline(ref, color='C0', ls='--', lw=1, alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlabel('nueCC efficiency after all cuts [%%], at %.0f %% purity' % P)
    ax.set_xlim(0, max(hi) + 12); ax.grid(axis='x', ls='--', alpha=0.5); ax.set_axisbelow(True)
    ax.set_title(a.title or 'Same purity (%.0f %%), compare the efficiency; fg1, BNB MC + nueCC-filtered MC at the BNB POT' % P, fontsize=10)
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print('wrote', a.out)
    if a.curves:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for (k, m), c in zip(M.items(), cols):
            s = sorted(m['scan'], key=lambda p: p['efficiency']); x = 100 * np.array([p['efficiency'] for p in s]); yv = 100 * np.array([p['purity'] for p in s]); e = 100 * np.array([p['purity_err'] for p in s])
            ax.plot(x, yv, '-o', ms=3, color=c, label='%s: %.1f %% at %.0f %% purity' % (m['label'], 100 * m['efficiency'], P)); ax.fill_between(x, yv - e, yv + e, color=c, alpha=0.15)
        ax.axhline(P, color='k', ls='--', lw=1); ax.set_xlabel('nueCC efficiency after all cuts [%]'); ax.set_ylabel('purity [%]'); ax.grid(ls='--', alpha=0.5); ax.legend(fontsize=8)
        ax.set_title('Purity against efficiency as the score cut moves; fg1 at the BNB POT', fontsize=10); fig.tight_layout(); fig.savefig(a.curves, dpi=130); print('wrote', a.curves)

if __name__ == '__main__': main()
