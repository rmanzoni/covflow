#!/usr/bin/env python3
"""
compare_killmaps.py -- do two kill-map builds see the same detector?

Compares the DATA hit efficiency of two builds of build_kill_maps.py, cell by
cell, e.g. the prompt / non-prompt split of HITEMU.md 3.7:

  A: non-prompt probes (the decay-length cut of the covflow selection)
  B: prompt probes     (--drop-cut lxy --data-extra '<decay length> < cut')

If the hit efficiency of a cell does not depend on which J/psi the probe came
from, the pulls (eps_A - eps_B) / sigma are N(0, 1), and all the data, prompt
and non-prompt, can be used for the kill maps.

Per surface:
  - epoch-integrated eps per cell (sum over the run ranges), both builds;
  - pulls, their mean, RMS, chi2/ndf and the fraction above 3;
  - per run range as well, if the two builds share their ranges
    (build B with --ranges-from A).
And the selection-loss test of HITEMU 10.4 for both builds: selected probes
per unit luminosity crossing a cell, relative to the cell's average, against
the cell's hit efficiency in that range. If part of the loss comes from the
decay-length cut, it is smaller for the prompt build.

Usage (tcsh, covflow env):
  python compare_killmaps.py km_np/killmaps_2026 km_p/killmaps_2026 \\
         --labels non-prompt prompt --out cmp_2026
Outputs cmp_2026/compare_<epoch>.pdf and .json; the summary is printed.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import hitemu as H
from pixel_eff_maps import die


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('a', help='kill-map stem A (…/killmaps_<epoch>)')
    p.add_argument('b', help='kill-map stem B')
    p.add_argument('--labels', nargs=2, default=['A', 'B'])
    p.add_argument('--min-probes', type=float, default=30,
                   help='cells with fewer data probes in either build are skipped')
    p.add_argument('--surfaces', nargs='+', default=None, help='default: all common')
    p.add_argument('--out', default='compare_killmaps')
    return p.parse_args()


def pulls(ka, na, kb, nb, mask):
    """Two-proportion z-test: under 'same efficiency' the error of eps_A - eps_B
    uses the pooled efficiency, sqrt(eps (1 - eps) (1/n_A + 1/n_B)). Better
    behaved than adding the two binomial errors when eps is close to 1 and one
    sample is small (that version gives a biased mean pull)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ea, eb = ka / na, kb / nb
        ep = (ka + kb) / (na + nb)
        ep = np.clip(ep, 0.5 / (na + nb), 1 - 0.5 / (na + nb))
        p = (ea - eb) / np.sqrt(ep * (1 - ep) * (1 / na + 1 / nb))
    p = np.where(mask, p, np.nan)
    return ea, eb, p


def summary(p):
    v = p[np.isfinite(p)]
    if len(v) == 0:
        return dict(n=0)
    return dict(n=int(len(v)), mean=float(v.mean()), rms=float(v.std()),
                chi2ndf=float((v ** 2).mean()), frac3=float((np.abs(v) > 3).mean()))


def yield_profile(km, s, bins):
    """HITEMU 10.4: probes per unit luminosity in a cell, relative to the cell's
    own average, against its efficiency in the range."""
    dd, dn = km.d_den[s], km.d_num[s]
    share = dd.sum((1, 2)) / dd.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = dd / (share[:, None, None] * dd.sum(0)[None])
    ok = (dd >= 30) & (dd.sum(0)[None] >= 30 * dd.shape[0])
    eps = np.where(ok, dn / np.maximum(dd, 1), np.nan)
    x, y, w = eps[ok], rate[ok], dd[ok]
    idx = np.digitize(x, bins) - 1
    prof = np.array([np.average(y[idx == i], weights=w[idx == i]) if (idx == i).sum() > 20
                     else np.nan for i in range(len(bins) - 1)])
    # cells that are dead in some ranges and alive in others: their yield in the
    # dead ranges over their yield in the alive ones (1 = no selection loss)
    dead, alive = ok & (eps < 0.4), ok & (eps > 0.8)
    changing = dead.any(0) & alive.any(0)
    dead, alive = dead & changing[None], alive & changing[None]
    ratio = (np.average(rate[dead], weights=dd[dead]) / np.average(rate[alive], weights=dd[alive])
             if dead.any() and alive.any() else np.nan)
    return prof, float(ratio)


def main():
    a = parse_args()
    A, B = H.KillMaps.load(a.a), H.KillMaps.load(a.b)
    if A.meta.get('epoch') != B.meta.get('epoch'):
        die('different epochs: %s and %s' % (A.meta.get('epoch'), B.meta.get('epoch')))
    for k in ('helix_origin', 'nbins_phi', 'nbins_r'):
        if A.meta.get(k) != B.meta.get(k):
            die('the builds differ in %s (%s vs %s): their cells are not comparable'
                % (k, A.meta.get(k), B.meta.get(k)))
    surfaces = a.surfaces or [s for s in A.surfaces if s in B.surfaces]
    same_ranges = A.ranges == B.ranges
    la, lb = a.labels
    os.makedirs(a.out, exist_ok=True)
    epoch = A.meta.get('epoch')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams['mathtext.default'] = 'regular'

    print('compare kill maps, epoch %s: A = %s (%s), B = %s (%s)' % (epoch, la, a.a, lb, a.b))
    for lab, km in ((la, A), (lb, B)):
        print('  %-12s selection: %s | data: %s' % (lab, km.meta.get('selection') or '-',
                                                   km.meta.get('data_selection') or '-'))
    print('  run ranges: %s' % ('identical, compared range by range too' if same_ranges
                                else 'different (build B with --ranges-from A to compare per range)'))
    res = dict(epoch=epoch, labels=[la, lb], a=a.a, b=a.b, same_ranges=same_ranges, surfaces={})
    bins = np.linspace(0, 1, 11)
    pdf = PdfPages(os.path.join(a.out, 'compare_%s.pdf' % epoch))
    print('\n  %-5s %8s %8s %8s %7s %7s %8s %8s   (epoch-integrated, %s - %s)'
          % ('surf', 'eps A', 'eps B', 'cells', 'mean', 'RMS', 'chi2/ndf', '|p|>3', la, lb))
    for s in surfaces:
        if not (np.array_equal(A.edges[s][0], B.edges[s][0])
                and np.array_equal(A.edges[s][1], B.edges[s][1])):
            die('%s: different cell edges' % s)
        xe, ye = A.edges[s]
        xc = 0.5 * (xe[1:] + xe[:-1])
        acc = H.in_acceptance(s, xc)[:, None] & np.ones((1, len(ye) - 1), bool)
        ka, na = A.d_num[s].sum(0), A.d_den[s].sum(0)
        kb, nb = B.d_num[s].sum(0), B.d_den[s].sum(0)
        m = acc & (na >= a.min_probes) & (nb >= a.min_probes)
        ea, eb, p = pulls(ka, na, kb, nb, m)
        sm = summary(p)
        sm['eps_a'] = float(ka[acc].sum() / max(na[acc].sum(), 1))
        sm['eps_b'] = float(kb[acc].sum() / max(nb[acc].sum(), 1))
        sm['probes_a'], sm['probes_b'] = float(na[acc].sum()), float(nb[acc].sum())
        print('  %-5s %8.4f %8.4f %8d %7.2f %7.2f %8.2f %8.3f'
              % (s, sm['eps_a'], sm['eps_b'], sm.get('n', 0), sm.get('mean', np.nan),
                 sm.get('rms', np.nan), sm.get('chi2ndf', np.nan), sm.get('frac3', np.nan)))
        per_range = []
        if same_ranges:
            for r in range(len(A.ranges)):
                mr = acc & (A.d_den[s][r] >= a.min_probes) & (B.d_den[s][r] >= a.min_probes)
                _, _, pr = pulls(A.d_num[s][r], A.d_den[s][r], B.d_num[s][r], B.d_den[s][r], mr)
                per_range.append(summary(pr))
        sm['per_range'] = per_range
        prof_a, ratio_a = yield_profile(A, s, bins)
        prof_b, ratio_b = yield_profile(B, s, bins)
        sm['dead_alive_yield'] = {la: ratio_a, lb: ratio_b}
        res['surfaces'][s] = sm

        fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
        ax[0].scatter(eb[m], ea[m], s=4, alpha=0.5)
        ax[0].plot([0, 1], [0, 1], 'r:', lw=1)
        ax[0].set_xlabel('ε, %s' % lb); ax[0].set_ylabel('ε, %s' % la)
        ax[0].set_title('%s: hit efficiency per cell, epoch-integrated' % s)
        pm = np.where(m, p, np.nan)
        im = ax[1].pcolormesh(xe, ye, pm.T, cmap='RdBu_r', vmin=-4, vmax=4)
        fig.colorbar(im, ax=ax[1], label='pull')
        ax[1].set_xlabel(('z' if s[0] == 'L' else 'r') + ' [cm]'); ax[1].set_ylabel('φ [rad]')
        ax[1].set_title('pull (ε$_{%s}$ − ε$_{%s}$) / σ' % (la, lb))
        v = p[np.isfinite(p)]
        hb = np.linspace(-6, 6, 61)
        ax[2].hist(np.clip(v, -6, 6), hb, histtype='stepfilled', color='#aecde5')
        g = np.exp(-0.5 * (0.5 * (hb[1:] + hb[:-1])) ** 2) / np.sqrt(2 * np.pi) * len(v) * (hb[1] - hb[0])
        ax[2].plot(0.5 * (hb[1:] + hb[:-1]), g, 'k--', label='N(0, 1)')
        ax[2].set_xlabel('pull'); ax[2].legend(frameon=False)
        ax[2].set_title('mean %.2f, RMS %.2f, χ²/ndf %.2f' % (sm.get('mean', np.nan),
                                                             sm.get('rms', np.nan),
                                                             sm.get('chi2ndf', np.nan)))
        xc10 = 0.5 * (bins[1:] + bins[:-1])
        ax[3].plot(xc10, prof_a, 'o-', label='%s (dead/alive %.3f)' % (la, ratio_a))
        ax[3].plot(xc10, prof_b, 's-', label='%s (dead/alive %.3f)' % (lb, ratio_b))
        ax[3].axhline(1, color='0.5', lw=0.8); ax[3].set_ylim(0.8, 1.15)
        ax[3].set_xlabel('ε of the cell in that run range')
        ax[3].set_ylabel('probes per unit luminosity,\nrelative to the cell average')
        ax[3].set_title('selection loss in dead cells (HITEMU 10.4)'); ax[3].legend(fontsize=8)
        for x_ in ax:
            x_.grid(alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print('\n  selection loss (HITEMU 10.4): for cells dead (ε < 0.4) in some run ranges and '
          'alive (ε > 0.8) in others,\n  probes per unit luminosity when dead over when alive '
          '(1 = no loss; nan = no such cells)')
    for s, sm in res['surfaces'].items():
        r = sm['dead_alive_yield']
        print('  %-5s %s %.3f   %s %.3f' % (s, la, r[la], lb, r[lb]))
    if same_ranges:
        print('\n  per run range, chi2/ndf of the pulls (%s):' % ', '.join(surfaces))
        for k in range(len(A.ranges)):
            print('  range %d %s' % (k, '  '.join('%s %.2f (%d)' % (
                s, res['surfaces'][s]['per_range'][k].get('chi2ndf', np.nan),
                res['surfaces'][s]['per_range'][k].get('n', 0)) for s in surfaces)))
    pdf.close()
    with open(os.path.join(a.out, 'compare_%s.json' % epoch), 'w') as f:
        json.dump(res, f, indent=1)
    print('\n[out] %s/compare_%s.pdf, .json' % (a.out, epoch))
    print('Reading: chi2/ndf ~ 1 and |pull| > 3 at the 0.3% level: same detector, '
          'use both samples. A pattern in the pull map (module edges, one side of a disk): '
          'the efficiency depends on the probe sample there.')


if __name__ == '__main__':
    main()
