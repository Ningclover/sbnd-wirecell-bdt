"""Draw the evaluation-standard figure (the nueCC efficiency / purity page of the SBND Wire-Cell nu_e CC report, 2026-11-10):
efficiency against true neutrino energy and purity against reconstructed neutrino energy, one curve per progressive cut,
from one or more result files of eval_standard.py. Several files overlay (e.g. shipped score and a retrained model).

  python evaluation/plot_eval_standard.py RESULT.json [RESULT2.json ...] [--labels A B] [--cls nueCC] [--out fig.png]
                                          [--last-only]
"""
import argparse, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('results', nargs='+'); ap.add_argument('--labels', nargs='*'); ap.add_argument('--cls', default='nueCC')
    ap.add_argument('--out', default='eval_standard.png'); ap.add_argument('--last-only', action='store_true', help='draw only the full selection of each file'); ap.add_argument('--title', default='')
    a = ap.parse_args(); labels = a.labels or [r for r in a.results]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = ['o', 's', '^', 'D']
    for i, (path, lab) in enumerate(zip(a.results, labels)):
        res = json.load(open(path)); cls = res['classes'][a.cls]; binned = cls['binned']; steps = cls['steps']
        if a.last_only: binned, steps = binned[-1:], steps[-1:]
        for k, (b, st) in enumerate(zip(binned, steps)):
            e = np.array(b['edges_mev'], float); x = 0.5 * (e[:-1] + e[1:]); xe = 0.5 * (e[1:] - e[:-1])
            col = 'C%d' % (k if len(a.results) == 1 else i)
            for ax, key, tot in ((axes[0], 'efficiency_vs_true_E', st['efficiency']), (axes[1], 'purity_vs_reco_E', st['purity'])):
                v = np.array([p[0] for p in b[key]], float); err = np.array([p[1] for p in b[key]], float)
                name = '%s%s (%.1f%%)' % (lab + ': ' if len(a.results) > 1 else '', st['step'], 100 * tot)
                ax.errorbar(x, v, xerr=xe, yerr=err, fmt=styles[i % 4], ms=3.5, color=col, lw=1, capsize=0, label=name)
    axes[0].set_xlabel('true E$_\\nu$ [MeV]'); axes[0].set_ylabel('efficiency'); axes[0].set_title('%s: efficiency' % a.cls)
    axes[1].set_xlabel('reco E$_\\nu$ [MeV]'); axes[1].set_ylabel('purity'); axes[1].set_title('%s: purity' % a.cls)
    for ax in axes: ax.set_ylim(0, 1.05); ax.set_xlim(0, 6000); ax.grid(ls='--', alpha=0.5); ax.legend(fontsize=7, loc='lower right' if ax is axes[1] else 'best')
    fig.suptitle(a.title or '%s, BNB MC + nueCC-filtered MC (counts at the BNB POT)' % a.cls, fontsize=10); fig.tight_layout()
    fig.savefig(a.out, dpi=130); print('wrote', a.out)

if __name__ == '__main__': main()
