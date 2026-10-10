#!/usr/bin/env python3
"""
build_kill_maps.py -- step 1 of the hit-loss emulation.

For one epoch of configs/run3_epochs.py:
  pass 1 (data)   per-run L1 and D1 efficiency (probes inside acceptance)
                  -> the runs are split into blocks ("run ranges") at the
                     largest efficiency changes (binary segmentation on the
                     pooled binomial likelihood, L1 and D1 together)
  pass 2 (data)   efficiency maps per run range
  pass 3 (MC)     efficiency maps, pileup-weighted (one: MC has no time axis)
  -> killmaps_<epoch>.npz / .json   input of emulate_hit_loss.py
     killmaps_<epoch>.pdf           ranges, maps per range, P_kill per range

Definitions are those of pixel_eff_maps.py: a probe is mu1 or mu2 crossing the
surface, with >= --min-other-hits valid pixel hits besides the one on it;
eps(cell) = probes with a valid hit / probes.

Each run range gets a share of the MC events in the emulation equal to its
share of the luminosity: by default the share of SELECTED DATA EVENTS (a proxy
that also contains trigger and selection efficiency), or the recorded
luminosity per run from a brilcalc csv with --lumi-csv.

Usage (tcsh, UI, covflow env, from the repository):
  python build_kill_maps.py --epoch 2026 --max-events 300000 --out test_km
  python build_kill_maps.py --epoch 2026 |& tee km_2026.log
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import hitemu as H
import pixel_eff_maps as P
from pixel_eff_maps import die, warn, GEOM

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CFG = os.path.join(HERE, 'configs', 'run3_epochs.py')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    H.add_common_args(p, DEFAULT_CFG)
    p.add_argument('--nbins-phi', type=int, default=48,
                   help='phi bins of the kill maps (default %(default)s: about two '
                        'per L1 ROC row)')
    p.add_argument('--nbins-r', type=int, default=20)
    p.add_argument('--max-ranges', type=int, default=10)
    p.add_argument('--min-cell-probes', type=float, default=40.,
                   help='a run range must hold on average this many L1 probes per '
                        'cell in acceptance (sets the minimum range size)')
    p.add_argument('--min-gain', type=float, default=25.,
                   help='a split must lower -2lnL by more than this')
    p.add_argument('--min-cell', type=float, default=30.,
                   help='cells with fewer (effective) probes fall back, see hitemu.KillMaps')
    p.add_argument('--prior', type=float, default=0.0,
                   help='pseudo-probes pulling sparse data cells toward eps_MC '
                        '(default %(default)s = off)')
    p.add_argument('--min-weight', type=float, default=0.0,
                   help='floor of the weight (1-eps_data)/(1-eps_MC) of MC tracks '
                        'without the hit in cells where data is more efficient '
                        '(default %(default)s = no floor; it was 0.2 until Oct 2026: '
                        'a floor biases the emulated efficiency low wherever data and '
                        'MC are both ~0.99, by ~0.1-0.2%% per surface)')
    p.add_argument('--max-weight', type=float, default=3.0,
                   help='cells where eps_data/eps_MC exceeds this cannot be '
                        'emulated (MC has (almost) no hits there) and are left '
                        'unchanged and listed (default %(default)s; was 1.5 until Oct 2026: '
                        'on the full 2026 maps 3 halves the D1+ efficiency left missing)')
    p.add_argument('--fallback', choices=('mcshape', 'average'), default='mcshape',
                   help='data efficiency of cells with too few probes in the whole epoch: '
                        'mcshape = eps_MC of the cell x one data/MC factor per range '
                        '(default), average = data surface average (old behaviour, '
                        'biased low where MC has dead cells)')
    p.add_argument('--surfaces', nargs='+', default=['auto'],
                   help="surfaces to map: 'auto' (default: all ten with the per-layer "
                        "hit masks in the ntuple, else L1 D1+ D1-), 'all', or a list "
                        "such as L1 L2 D1+ D1-")
    p.add_argument('--lumi-csv', default=None,
                   help='brilcalc csv with recorded luminosity per run')
    p.add_argument('--out', default=None, help='default: killmaps_<epoch>')
    p.add_argument('--png', action='store_true')
    return p.parse_args()


def probes(v, s, a):
    """Crossing, cell inputs and hit flag for the probes of surface s. "Other
    hits" excludes all the hits on the surface's layer (2 on module overlaps,
    from the per-layer counts when the ntuple has them)."""
    x, y = H.cross_view(s, v)
    hit = H.has_hit(s, v).astype(np.float64)
    ok = H.good_track(v) & np.isfinite(x) & np.isfinite(y) \
        & (H.other_pixel_hits(s, v) >= a.min_other_hits)
    return x, y, hit, ok


def choose_surfaces(req, tmpl):
    if req == ['auto']:
        return H.default_surfaces(tmpl)
    if req == ['all']:
        req = list(H.ALL_SURFACES)
    bad = [s for s in req if s not in H.ALL_SURFACES]
    if bad:
        die('unknown surface(s) %s: use %s' % (bad, ' '.join(H.ALL_SURFACES)))
    if any(s not in H.KILL_SURFACES for s in req) and not H.has_masks(tmpl):
        die('surfaces beyond L1 / D1 need the per-layer hit masks '
            '({mu}_pix_valid_mask, {mu}_pix_hit_count), not in these ntuples')
    if 'L1' not in req or 'D1+' not in req or 'D1-' not in req:
        die('L1, D1+ and D1- are always needed (they define the run ranges)')
    return [s for s in H.ALL_SURFACES if s in req]


def main():
    a = parse_args()
    t0 = time.time()
    info = P.load_epoch(a)
    tmpl, run_branch = H.resolve_branches(a)
    tmpl = H.drop_missing_optional(tmpl, [info['data'], info['mc']], info['tree'], a.muons)
    surfaces = choose_surfaces(a.surfaces, tmpl)
    out = a.out or 'killmaps_%s' % a.epoch
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, 'killmaps_%s' % a.epoch)

    sel_d, sel_m = H.selection_expr(info, False), H.selection_expr(info, True)
    need_d = H.sample_branches(tmpl, a.muons, sel_d, [run_branch])
    need_m = H.sample_branches(tmpl, a.muons, sel_m,
                               [info['mc_weight']] if info['mc_weight'] else [])
    print('kill maps, epoch %s: %d data files, MC %s (weight %s)'
          % (a.epoch, len(info['data']), ', '.join(os.path.basename(f) for f in info['mc']),
             info['mc_weight'] or 'none'))
    entries = P.preflight({'data': info['data'], 'mc': info['mc']}, info['tree'],
                          {'data': need_d, 'mc': need_m})
    print('surfaces : %s (%s)' % (' '.join(surfaces), 'per-layer hit masks' if H.has_masks(tmpl)
                                    else 'no per-layer hit mask: L1 / D1 from the first-hit branches'))
    edges = {s: H.surface_edges(s, a.nbins_phi, a.nbins_r) for s in surfaces}

    # ------------------------------------------------------------- pass 1
    print('\n[1/3] per-run efficiency (data)')
    per_run = {}           # run -> [events, L1 probes, L1 hits, D1 probes, D1 hits]
    for chunk, n in H.iterate(info['data'], info['tree'], need_d, a, entries, 'data runs'):
        sel = P.eval_selection(sel_d, chunk, n)
        run_ev = np.asarray(chunk[run_branch])[:n][sel].astype(np.int64)
        ur, inv = np.unique(run_ev, return_inverse=True)
        acc = np.zeros((len(ur), 5))
        acc[:, 0] = np.bincount(inv, minlength=len(ur))
        for mu in a.muons:
            v = H.muon_view(chunk, mu, tmpl, a, sel)
            for s in ('L1', 'D1+', 'D1-'):
                x, y, hit, ok = probes(v, s, a)
                ok &= H.in_acceptance(s, x)
                col = 1 if s == 'L1' else 3
                acc[:, col] += np.bincount(inv[ok], minlength=len(ur))
                acc[:, col + 1] += np.bincount(inv[ok], weights=hit[ok], minlength=len(ur))
        for r, row in zip(ur.tolist(), acc):
            per_run.setdefault(r, np.zeros(5))
            per_run[r] += row
    if not per_run:
        die('no selected data events')
    runs = np.array(sorted(per_run))
    tab = np.array([per_run[r] for r in runs])

    # ------------------------------------------------------- segmentation
    xe, ye = edges['L1']
    xc = 0.5 * (xe[1:] + xe[:-1])
    n_cells = int(H.in_acceptance('L1', xc).sum()) * (len(ye) - 1)
    min_den = a.min_cell_probes * n_cells
    num = {'L1': tab[:, 2], 'D1': tab[:, 4]}
    den = {'L1': tab[:, 1], 'D1': tab[:, 3]}
    segs, gains = H.segment_runs(runs, num, den, min_den, a.max_ranges, a.min_gain)
    ranges = [(int(runs[i]), int(runs[j - 1])) for i, j in segs]
    if a.lumi_csv:
        lumi_run = H.read_lumi_csv(a.lumi_csv)
        missing = [r for r in runs if r not in lumi_run]
        if missing:
            warn('%d runs with selected events have no luminosity in %s, e.g. %s'
                 % (len(missing), a.lumi_csv, missing[:5]))
        lumi = np.array([sum(lumi_run.get(int(r), 0.0) for r in runs[i:j]) for i, j in segs])
        lumi_kind = 'recorded luminosity (%s)' % a.lumi_csv
    else:
        lumi = np.array([tab[i:j, 0].sum() for i, j in segs])
        lumi_kind = 'selected data events (proxy)'
    print('\n    %d runs, %d L1 probes in acceptance; minimum per range %d probes '
          '(%g per cell x %d cells)' % (len(runs), tab[:, 1].sum(), min_den,
                                         a.min_cell_probes, n_cells))
    print('    %-3s %-15s %6s %12s %8s %8s' % ('#', 'runs', 'share', 'L1 probes', 'eps L1', 'eps D1'))
    for k, (i, j) in enumerate(segs):
        print('    %-3d %6d-%-8d %6.3f %12d %8.4f %8.4f'
              % (k, runs[i], runs[j - 1], lumi[k] / lumi.sum(), tab[i:j, 1].sum(),
                 tab[i:j, 2].sum() / max(tab[i:j, 1].sum(), 1),
                 tab[i:j, 4].sum() / max(tab[i:j, 3].sum(), 1)))
    R = len(ranges)
    first = np.array([r[0] for r in ranges])

    # ------------------------------------------------------------- pass 2
    print('\n[2/3] maps per run range (data)')
    d_num = {s: np.zeros((R, len(edges[s][0]) - 1, len(edges[s][1]) - 1)) for s in edges}
    d_den = {s: np.zeros_like(d_num[s]) for s in edges}
    for chunk, n in H.iterate(info['data'], info['tree'], need_d, a, entries, 'data maps'):
        sel = P.eval_selection(sel_d, chunk, n)
        run_ev = np.asarray(chunk[run_branch])[:n][sel].astype(np.int64)
        rng_idx = np.clip(np.searchsorted(first, run_ev, side='right') - 1, 0, R - 1)
        for mu in a.muons:
            v = H.muon_view(chunk, mu, tmpl, a, sel)
            for s in surfaces:
                x, y, hit, ok = probes(v, s, a)
                ix, iy, inmap = H.cell_index(*edges[s], x, y)
                ok &= inmap
                shape = d_num[s].shape
                flat = np.ravel_multi_index((rng_idx[ok], ix[ok], iy[ok]), shape)
                d_den[s] += np.bincount(flat, minlength=np.prod(shape)).reshape(shape)
                d_num[s] += np.bincount(flat, weights=hit[ok],
                                        minlength=np.prod(shape)).reshape(shape)

    # ------------------------------------------------------------- pass 3
    print('\n[3/3] maps (MC, weighted)')
    m = {k: {s: np.zeros((len(edges[s][0]) - 1, len(edges[s][1]) - 1)) for s in edges}
         for k in ('num', 'den', 'num2', 'den2')}
    n_drop = 0
    for chunk, n in H.iterate(info['mc'], info['tree'], need_m, a, entries, 'MC maps'):
        sel = P.eval_selection(sel_m, chunk, n)
        if info['mc_weight']:
            w = np.asarray(chunk[info['mc_weight']], np.float64)[:n]
            fin = np.isfinite(w)
            n_drop += int((sel & ~fin).sum())
            sel &= fin
            w = w[sel]
        else:
            w = np.ones(int(sel.sum()))
        for mu in a.muons:
            v = H.muon_view(chunk, mu, tmpl, a, sel)
            for s in surfaces:
                x, y, hit, ok = probes(v, s, a)
                ix, iy, inmap = H.cell_index(*edges[s], x, y)
                ok &= inmap
                shape = m['num'][s].shape
                flat = np.ravel_multi_index((ix[ok], iy[ok]), shape)
                for k, ww in (('den', w[ok]), ('den2', w[ok] ** 2),
                              ('num', w[ok] * hit[ok]), ('num2', w[ok] ** 2 * hit[ok])):
                    m[k][s] += np.bincount(flat, weights=ww,
                                           minlength=np.prod(shape)).reshape(shape)
    if n_drop:
        print('    %d selected MC events dropped for a non-finite %s' % (n_drop, info['mc_weight']))

    meta = dict(epoch=a.epoch, config=os.path.abspath(a.config), lumi_kind=lumi_kind,
                min_den_per_range=min_den, min_gain=a.min_gain, splits=gains,
                nbins_phi=a.nbins_phi, nbins_r=a.nbins_r, geometry=GEOM,
                branches=tmpl, z0_from=a.z0_from, helix_origin=a.helix_origin,
                min_other_hits=a.min_other_hits,
                selection=info['selection'], mc_selection=info['mc_selection'],
                mc_weight=info['mc_weight'], max_events=a.max_events,
                per_run=dict(runs=runs.tolist(), events=tab[:, 0].tolist(),
                             L1_probes=tab[:, 1].tolist(), L1_hits=tab[:, 2].tolist(),
                             D1_probes=tab[:, 3].tolist(), D1_hits=tab[:, 4].tolist()))
    meta['hit_masks'] = H.has_masks(tmpl)
    km = H.KillMaps(list(surfaces), edges, ranges, lumi, d_num, d_den,
                    m['num'], m['den'], m['num2'], m['den2'], meta).finalise(a.min_cell,
                                                                            a.max_weight,
                                                                            a.prior,
                                                                            a.min_weight,
                                                                            a.fallback)
    km.save(stem)

    # --------------------------------------------------------------- report
    lines = report_lines(km, runs, tab, segs, a)
    print('\n' + '\n'.join(lines[2:]))
    plot(km, runs, tab, segs, lines, stem, a)
    print('\n[out] %s.npz, %s.json, %s.pdf   (%.1f min)' % (stem, stem, stem,
                                                          (time.time() - t0) / 60))


def report_lines(km, runs, tab, segs, a):
    L = ['Kill maps, epoch %s' % a.epoch, '',
         'run ranges: binary segmentation of the per-run L1 + D1 efficiency',
         '            (at most %d ranges, each >= %d L1 probes, split gain > %g in -2lnL)'
         % (a.max_ranges, km.meta['min_den_per_range'], a.min_gain),
         'MC share  : %s' % km.meta['lumi_kind'],
         'eps       = HIT EFFICIENCY of a cell = probes with a valid hit / probes (eps_data: data, eps_MC: MC)',
         'P_kill    = 1 - eps_data/eps_MC where eps_data < eps_MC (else 0)',
         'weights   = eps_data/eps_MC (hit) and (1-eps_data)/(1-eps_MC) (no hit)',
         '            where eps_data > eps_MC: bounded by 1/eps_MC',
         '',
         '%-3s %-17s %6s %11s %7s %7s %7s | %s' % ('#', 'runs', 'share', 'L1 probes',
                                                  'eps L1', 'eps D1', 'kill L1',
                                                  'cells by fallback level 0/1/2/3 (L1)')]
    for k, (i, j) in enumerate(segs):
        s = 'L1'
        xe, ye = km.edges[s]
        acc = H.in_acceptance(s, 0.5 * (xe[1:] + xe[:-1]))
        lev = km.level_d[s][k][acc]
        pk = km.p_kill[s][k]
        dd = km.d_den[s][k]
        mean_pk = (pk * dd)[acc].sum() / max(dd[acc].sum(), 1)
        cnt = [int((lev == l).sum()) for l in range(4)]
        L.append('%-3d %7d-%-9d %6.3f %11d %7.4f %7.4f %7.4f | %s'
                 % (k, runs[i], runs[j - 1], km.lumi[k], tab[i:j, 1].sum(),
                    tab[i:j, 2].sum() / max(tab[i:j, 1].sum(), 1),
                    tab[i:j, 4].sum() / max(tab[i:j, 3].sum(), 1), mean_pk,
                    '/'.join(map(str, cnt))))
    L += ['', 'kill L1 = average P_kill over the data probes of the range',
          'fallback: 0 range+cell, 1 cell over the nearest ranges (scaled), 2 %s, '
          '3 none (min %g probes)'
          % ('eps_MC of the cell x data/MC factor of the range' if km.fallback == 'mcshape'
             else 'range surface average', km.min_cell), '']
    for s in km.surfaces:
        xe, ye = km.edges[s]
        acc = H.in_acceptance(s, 0.5 * (xe[1:] + xe[:-1]))
        w = km.w_hit[s][:, acc]
        frac = float(sum(km.lumi[k] * (w[k] > 1).mean() for k in range(len(km.lumi))))
        unc = km.uncorrectable[s]
        dd = km.d_den[s]
        lost = float(sum(km.lumi[k] * dd[k][unc[k]].sum() / max(dd[k].sum(), 1)
                         for k in range(len(km.lumi))))
        dacc = dd[:, acc]
        lv = km.level_d[s][:, acc]
        bylev = [float(dacc[lv == k].sum() / max(dacc.sum(), 1)) for k in (0, 1, 2, 3)]
        with np.errstate(divide='ignore', invalid='ignore'):
            emeas = np.where(dd > 0, km.d_num[s] / dd, 0.0)
        miss = float((dd * (emeas - np.nan_to_num(km.eps_m[s])[None]))[unc].sum() / max(dd.sum(), 1))
        unc_in = unc & acc[None, :, None]
        miss_in = float((dd * (emeas - np.nan_to_num(km.eps_m[s])[None]))[unc_in].sum()
                        / max(dd[:, acc].sum(), 1))
        L.append('%-4s data probes by fallback level 0/1/2/3: %s'
                 % (s, ' '.join('%.3f' % x for x in bylev)))
        L.append('     efficiency left missing in uncorrectable cells (emulated - data): %+.4f '
                 '(all mapped cells), %+.4f (nominal acceptance only)' % (-miss, -miss_in))
        L.append('%-4s cells in acceptance with eps_data > eps_MC (reweighted): %.3f, '
                 'largest weight %.3f' % (s, frac, float(w.max())))
        L.append('     data probes in UNCORRECTABLE cells, all mapped cells incl. disk edges (eps_data/eps_MC > %.2f, MC '
                 'nearly dead): %.4f' % (km.max_weight, lost))
    L += ['', 'PER-SURFACE SUMMARY: hit efficiency eps in nominal acceptance (data per run range, '
          'MC on the data illumination of the range) and average P_kill',
          '  surf   ' + ' '.join('%-21s' % ('range %d' % k) for k in range(len(km.ranges))),
          '         ' + ' '.join('%-21s' % 'data / MC / kill' for _ in km.ranges)]
    for s in km.surfaces:
        xe, ye = km.edges[s]
        acc = H.in_acceptance(s, 0.5 * (xe[1:] + xe[:-1]))
        cols = []
        for k in range(len(km.ranges)):
            dd, dn = km.d_den[s][k][acc], km.d_num[s][k][acc]
            em = np.nan_to_num(km.eps_m[s][acc])
            pk = km.p_kill[s][k][acc]
            tot = max(dd.sum(), 1e-300)
            cols.append('%.3f / %.3f / %.3f' % (dn.sum() / tot, (em * dd).sum() / tot,
                                                (pk * dd).sum() / tot))
        L.append('  %-5s  %s' % (s, ' '.join('%-21s' % c for c in cols)))
    return L


def plot(km, runs, tab, segs, lines, stem, a):
    plt = P.setup_mpl()
    book = P.Book(stem + '.pdf', png_prefix=stem if a.png else None)
    P.page_text(book, lines, 'summary')

    # per-run efficiency with the ranges
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    idx = np.arange(len(runs))
    for ax, (col, lab) in zip(axes, ((1, 'L1'), (3, 'D1'))):
        d, n = tab[:, col], tab[:, col + 1]
        m = d >= 200
        eps = np.where(m, n / np.maximum(d, 1), np.nan)
        err = np.where(m, np.sqrt(np.clip(eps * (1 - eps), 0, None) / np.maximum(d, 1)), np.nan)
        ax.errorbar(idx[m], eps[m], yerr=err[m], fmt='o', ms=2, color='k', elinewidth=0.6)
        for k, (i, j) in enumerate(segs):
            e = n[i:j].sum() / max(d[i:j].sum(), 1)
            ax.hlines(e, i - 0.5, j - 0.5, color='#d95f02', lw=2)
            if i > 0:
                ax.axvline(i - 0.5, color='#1b9e77', lw=1, ls='--')
        ax.set_ylabel('hit efficiency eps_data(%s), per run' % lab)
    step = max(1, len(runs) // 14)
    axes[-1].set_xticks(idx[::step])
    axes[-1].set_xticklabels([str(r) for r in runs[::step]], rotation=45, fontsize=8)
    axes[-1].set_xlabel('run (runs with >= 200 probes shown; orange = range average, '
                        'green = range boundary)')
    fig.suptitle('Run ranges, epoch %s' % a.epoch, fontsize=13)
    fig.tight_layout()
    book.add(fig, 'ranges')

    R = len(km.ranges)
    ncol = min(R, 4)
    nrow = int(np.ceil(R / ncol))
    for s in km.surfaces:
        surf = P.Surface(s, type('A', (), dict(nbins_phi=a.nbins_phi, nbins_r=a.nbins_r))())
        for what, cmap, vmin, vmax in (('eps_data', 'viridis', 0, 1),
                                       ('P_kill', 'magma_r', 0, 1)):
            fig = plt.figure(figsize=(4.6 * ncol, 4.0 * nrow))
            for k in range(R):
                M = km.eps_d[s][k] if what == 'eps_data' else \
                    np.where(np.isfinite(km.eps_d[s][k]), km.p_kill[s][k], np.nan)
                ax = P.draw_map(fig, (nrow, ncol, k + 1), surf, M,
                                'range %d: %d-%d' % (k, *km.ranges[k]), cmap, vmin, vmax,
                                {'eps_data': 'data hit efficiency (eps_data)',
                                 'P_kill': 'P_kill = 1 - eps_data/eps_MC'}[what])
                ax.title.set_fontsize(9)
            fig.suptitle('%s %s per run range, epoch %s' % (
                s, {'eps_data': 'data hit efficiency eps_data', 'P_kill': 'kill probability P_kill'}[what],
                a.epoch), fontsize=12)
            fig.tight_layout()
            book.add(fig, '%s_%s' % (s.replace('+', 'p').replace('-', 'm'), what))
        fig = plt.figure(figsize=(18, 5.6) if s[0] == 'L' else (18, 6.2))
        ed = km.lumi_avg_eps_data(s)
        P.draw_map(fig, (1, 3, 1), surf, ed, 'data hit efficiency eps_data, luminosity-weighted over ranges',
                   'viridis', 0, 1, 'hit efficiency, data (eps_data)')
        P.draw_map(fig, (1, 3, 2), surf, km.eps_m[s], 'MC hit efficiency eps_MC (PU-weighted, after fallback)',
                   'viridis', 0, 1, 'hit efficiency, MC (eps_MC)')
        wmax = np.where(km.uncorrectable[s].any(0), np.nan, km.w_hit[s].max(0))
        P.draw_map(fig, (1, 3, 3), surf, np.where(np.isfinite(ed), wmax, np.nan),
                   'largest hit weight over ranges\n(grey inside acceptance: uncorrectable)',
                   'Blues', 1, km.max_weight, 'ratio of hit efficiencies eps_data/eps_MC')
        fig.suptitle('%s summary, epoch %s' % (s, a.epoch), fontsize=13)
        fig.tight_layout()
        book.add(fig, '%s_summary' % s.replace('+', 'p').replace('-', 'm'))
    book.close()


if __name__ == '__main__':
    main()
