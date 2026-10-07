#!/usr/bin/env python3
"""
inspect_killmaps.py -- where do the data probes of a kill map sit, and what
does the emulation leave missing?

For every surface (L1, D1+, D1-) and run range it splits the DATA probes in
acceptance by
  - how the data efficiency of their cell was obtained (fallback level 0-3),
  - how the MC efficiency of their cell was obtained (0: own cell, 1: surface
    average, 3: no MC),
  - whether the cell is "uncorrectable" (eps_data > max_weight * eps_MC),
and prints the efficiency that the emulation cannot provide there,
    missing = sum over uncorrectable cells of probes * (eps_data_measured - eps_MC) / all probes,
which is what the per-cell closure of emulate_hit_loss.py should show as
"MC emulated - data" (negative) for that surface.

    python inspect_killmaps.py test_km/killmaps_2026
    python inspect_killmaps.py test_km/killmaps_2026 --max-weight 1.5 2 3 5
"""
import argparse
import numpy as np
import hitemu as H


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('killmaps', help='stem of killmaps_<epoch>.npz/.json')
    p.add_argument('--max-weight', type=float, nargs='+', default=[None],
                   help='recompute with these caps (default: the one stored)')
    a = p.parse_args()
    for cap in a.max_weight:
        km = H.KillMaps.load(a.killmaps, max_weight=cap)
        print('\n=== %s   max_weight %.2f   min_cell %g' % (a.killmaps, km.max_weight, km.min_cell))
        for s in km.surfaces:
            xe, ye = km.edges[s]
            xc = 0.5 * (xe[1:] + xe[:-1])
            acc = H.in_acceptance(s, xc)[:, None] & np.ones((1, len(ye) - 1), bool)
            print('\n%s' % s)
            print('  range  data probes | by data level 0 / 1 / 2 / 3     | by MC level 0 / 1 / 3   '
                  '| uncorrectable: probes  [lev0 data]  missing eff')
            for r in range(km.d_den[s].shape[0]):
                dd, dn = km.d_den[s][r], km.d_num[s][r]
                tot = dd[acc].sum()
                if tot <= 0:
                    print('  %-5d  no probes' % r)
                    continue
                ld, lm, unc = km.level_d[s][r], km.level_m[s], km.uncorrectable[s][r]
                fd = [dd[acc & (ld == k)].sum() / tot for k in (0, 1, 2, 3)]
                fm = [dd[acc & (lm == k)].sum() / tot for k in (0, 1, 3)]
                u = acc & unc
                with np.errstate(divide='ignore', invalid='ignore'):
                    emeas = np.where(dd > 0, dn / dd, 0.0)
                miss = (dd * (emeas - np.nan_to_num(km.eps_m[s])))[u].sum() / tot
                print('  %-5d  %11d | %.3f %.3f %.3f %.3f          | %.3f %.3f %.3f        '
                      '|  %.4f  [%.4f]  %+.4f'
                      % (r, tot, *fd, *fm, dd[u].sum() / tot, dd[u & (ld == 0)].sum() / tot, -miss))
    print('\nlevel: 0 own cell (>= min_cell probes), 1 nearest run ranges, 2 surface average of the range, 3 none'
          '\n[lev0 data]: part of the uncorrectable probes whose data efficiency is measured in the cell itself;'
          '\nthe rest compares a surface AVERAGE with a real MC cell and is a fallback artefact.')


if __name__ == '__main__':
    main()
