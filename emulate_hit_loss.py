#!/usr/bin/env python3
"""
emulate_hit_loss.py -- steps 2, 3 and 5 of the hit-loss emulation.

STEP 2 (always). Every selected MC event is assigned to a data run range of the
epoch, with probability = the range's luminosity share. For each muon, the L1
and D1 hits are removed with probability P_kill of the cell crossed, in that
range (kill maps of build_kill_maps.py); in cells where data is more efficient
than MC the event gets the bounded weight instead. The context is rebuilt
(pixel hits, first BPix layer, first FPix disk). Checks:
  - per cell: efficiency of the emulated MC against the data (luminosity-
    weighted over the ranges): equal by construction, so this checks the code;
  - context: joint distribution of (first BPix layer, first FPix disk) per
    |eta| bin and of the pixel-hit count, data vs MC before / after;
  - how many MC muons are left with no pixel hit (tracks the reconstruction
    would probably have lost: counted, not removed).

STEP 3, --route A. The covariance of each muon that lost a hit is moved with
the epoch's trained covflow flows from its context c to the emulated c':
        C'_A = f_data^-1( f_MC(C; c) ; c' )

STEP 5, --route B. The information of the removed hit is taken out of the MC
covariance, with the hit errors fitted by noL1_data_study.py:
        C'_B,raw   = (C^-1 - H^T V^-1 H)^-1          (MC-like at c')
        C'_B,final = f_data^-1( f_MC(C'_B,raw ; c') ; c')   if flows are given
The B,raw output is what a covflow retrained on emulated MC would start from
(--write-tree).

Route checks, on the muons that lost L1 (and separately D1):
  - C' - C must be a valid covariance (losing a hit cannot make a track more
    precise): fraction where it is not, and the size of the negative part;
  - sigma'(dxy) >= sigma(dxy), sigma'(dsz) >= sigma(dsz): fraction that holds;
  - log10 sigma of each parameter, emulated MC against data tracks without the
    hit (all of them, and only those in dead cells = random losses), MC
    reweighted to data in (pt, |eta|, pixel hits, z0*sign(eta));
  - B,raw against MC's own tracks without L1 (closure inside MC);
  - track by track: route A against route B,final;
  - beamspot IP significance of the no-L1 muons, with the parameters smeared by
    delta ~ N(0, C' - C) (needs {mu}_bs_dxy and {mu}_bs_dxy_e).

Usage (tcsh, UI, covflow env):
  python emulate_hit_loss.py --epoch 2026 --killmaps killmaps_2026/killmaps_2026
  python emulate_hit_loss.py --epoch 2026 --killmaps killmaps_2026/killmaps_2026 \\
         --route A B --flow-dir "$RUNS/@epoch@/@mu@/task_0" \\
         --hit-errors noL1_data_2026/hiterrors_data_2026.json
  add --write-tree to also write the emulated MC (route B,raw covariances)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import hitemu as H
import pixel_eff_maps as P
from pixel_eff_maps import die

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CFG = os.path.join(HERE, 'configs', 'run3_epochs.py')
PARAMS = list(H.F.PARAM_NAMES)
FB_EDGES = np.arange(-0.5, 5.5)       # first BPix layer 0..4
FE_EDGES = np.arange(-0.5, 4.5)       # first FPix disk 0..3
NP_EDGES = np.arange(-0.5, 12.5)      # pixel hits
CTX_ETA = np.array([0.0, 0.8, 1.2, 1.6, 2.0, 2.6])
SIG_EDGES = np.linspace(-10, 40, 101)
VARIANTS = ('orig', 'A', 'Braw', 'Bfin')
VLABEL = {'orig': 'MC before (C)', 'A': "route A", 'Braw': 'route B, raw (MC-like)',
          'Bfin': 'route B, final'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    H.add_common_args(p, DEFAULT_CFG)
    p.add_argument('--killmaps', required=True)
    p.add_argument('--route', nargs='*', default=[], choices=('A', 'B'))
    p.add_argument('--flow-dir', default=None,
                   help='trained covflow run of the epoch, with {epoch} and {mu} '
                        '(or @epoch@ and @mu@) placeholders, e.g. '
                        '".../run3_epochs_TAG/@epoch@/@mu@/task_0"')
    p.add_argument('--hit-errors', nargs='+', default=None,
                   help='hiterrors_*.json for route B; for each surface the first '
                        'file with a fit is used: give the MC fit first, the data '
                        'fit as fallback')
    p.add_argument('--device', default='cpu')
    p.add_argument('--eps-dead', type=float, default=0.4,
                   help='dead cell for the "random losses only" data target')
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--write-tree', action='store_true',
                   help='write the emulated MC (needs route B): context and '
                        'covariance branches replaced, see the output json')
    p.add_argument('--closure-only', action='store_true',
                   help='only the hit killing and its closure (per cell, context): '
                        'the covariance and beamspot branches are not read and no '
                        'route / target histograms are made. Much faster; not '
                        'compatible with --route')
    p.add_argument('--fallback', choices=('mcshape', 'average'), default=None,
                   help='re-finalise the kill maps with this level-2 fallback '
                        '(default: the one stored in the maps; maps written before '
                        'Oct 2026 get mcshape)')
    p.add_argument('--min-weight', type=float, default=None,
                   help='re-finalise the kill maps with this floor on w_nohit '
                        '(default: the one stored in the maps; 0 = no floor)')
    p.add_argument('--max-weight', type=float, default=None,
                   help='re-finalise the kill maps with this cap on eps_data/eps_MC '
                        '(default: the one stored in the maps)')
    p.add_argument('--out', default=None, help='default: emu_<epoch>')
    p.add_argument('--png', action='store_true')
    a = p.parse_args()
    if 'A' in a.route and not a.flow_dir:
        die('--route A needs --flow-dir')
    if 'B' in a.route and not a.hit_errors:
        die('--route B needs --hit-errors')
    if a.closure_only and (a.route or a.write_tree):
        die('--closure-only cannot be combined with --route / --write-tree')
    if a.write_tree and 'B' not in a.route:
        die('--write-tree writes route B,raw covariances: add --route B')
    return a


class RwVar:
    """One variable histogrammed in reweighting cells (see hitemu.rw_cell)."""

    def __init__(self, edges):
        _, shape = H.rw_cell(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
        self.edges = edges
        self.h = np.zeros((int(np.prod(shape)), len(edges) - 1))

    def fill(self, cell, x, w):
        ik = np.searchsorted(self.edges, x, 'right') - 1
        ok = (cell >= 0) & np.isfinite(x) & np.isfinite(w) & (ik >= 0) & (ik < len(self.edges) - 1)
        H.hist_add(self.h, (cell[ok], ik[ok]), w[ok])

    def projected(self, target=None):
        h = self.h
        if target is not None:
            mine, theirs = h.sum(1), target.h.sum(1)
            with np.errstate(divide='ignore', invalid='ignore'):
                h = h * np.where(mine > 0, theirs / mine, 0.0)[:, None]
        out = h.sum(0)
        return out / out.sum() if out.sum() > 0 else out


class Ctx:
    """Context histograms: (|eta| bin, first_b, first_e) and n_pix."""

    def __init__(self):
        self.joint = np.zeros((len(CTX_ETA) - 1, len(FB_EDGES) - 1, len(FE_EDGES) - 1))
        self.npix = np.zeros((len(CTX_ETA) - 1, len(NP_EDGES) - 1))

    def fill(self, eta, fb, fe, npix, w):
        ie = np.searchsorted(CTX_ETA, np.abs(eta), 'right') - 1
        ib = np.searchsorted(FB_EDGES, fb, 'right') - 1
        ic = np.searchsorted(FE_EDGES, fe, 'right') - 1
        ip = np.searchsorted(NP_EDGES, npix, 'right') - 1
        ok = (ie >= 0) & (ie < len(CTX_ETA) - 1) & np.isfinite(w)
        okj = ok & (ib >= 0) & (ib < len(FB_EDGES) - 1) & (ic >= 0) & (ic < len(FE_EDGES) - 1)
        H.hist_add(self.joint, (ie[okj], ib[okj], ic[okj]), w[okj])
        okn = ok & (ip >= 0) & (ip < len(NP_EDGES) - 1)
        H.hist_add(self.npix, (ie[okn], ip[okn]), w[okn])


def cell_closure(km, maps, s):
    """Per-cell closure numbers and maps of surface s against the RAW data
    counts (summed over run ranges), all samples on the data illumination."""
    num, den = maps['emu'][s]
    xe, ye = km.edges[s]
    acc = H.in_acceptance(s, 0.5 * (xe[1:] + xe[:-1]))[:, None] & np.ones((1, len(ye) - 1), bool)
    dn, dd = km.d_num[s].sum(0), km.d_den[s].sum(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        e_emu = np.where(den > 0, num / den, np.nan)
        e_dat = np.where(dd > 0, dn / dd, np.nan)
        e_mc = np.where(km.m_den[s] > 0, km.m_num[s] / km.m_den[s], np.nan)
    both = acc & np.isfinite(e_emu) & (dd > 0)
    meas = both & (dd >= km.min_cell)
    avg = lambda e, m: float((e[m] * dd[m]).sum() / max(dd[m].sum(), 1e-300))
    return dict(data=float(dn[both].sum() / max(dd[both].sum(), 1e-300)),
                mc=avg(np.nan_to_num(e_mc), both), emu=avg(e_emu, both),
                data_meas=float(dn[meas].sum() / max(dd[meas].sum(), 1e-300)),
                emu_meas=avg(e_emu, meas),
                frac_meas=float(dd[meas].sum() / max(dd[both].sum(), 1e-300)),
                maps=(np.where(meas, e_dat, np.nan), e_emu, np.where(meas, e_emu - e_dat, np.nan)))


def tv(h1, h2):
    """Total variation between two histograms of the same shape, normalised."""
    a, b = h1 / max(h1.sum(), 1e-300), h2 / max(h2.sum(), 1e-300)
    return 0.5 * float(np.abs(a - b).sum())


def tv_eta(h1, h2):
    """TV of the conditional distributions per |eta| bin, averaged with the
    data |eta| distribution (so that a pt/eta spectrum difference does not
    count as a hit-pattern difference)."""
    w = h1.reshape(h1.shape[0], -1).sum(1)
    w = w / max(w.sum(), 1e-300)
    return float(sum(w[i] * tv(h1[i], h2[i]) for i in range(h1.shape[0])))


def main():
    a = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    info = P.load_epoch(a)
    km = H.KillMaps.load(a.killmaps, fallback=a.fallback, max_weight=a.max_weight,
                         min_weight=a.min_weight)
    print('kill maps %s, level-2 fallback: %s, max weight %.2f, min weight %.2f'
          % (a.killmaps, km.fallback, km.max_weight, km.min_weight))
    timing = {}
    if km.meta.get('epoch') != a.epoch:
        die('kill maps are for epoch %s, not %s' % (km.meta.get('epoch'), a.epoch))
    he = H.HitErrors(a.hit_errors) if 'B' in a.route else None
    flows = {}
    if a.flow_dir:
        for mu in a.muons:
            # {epoch}/{mu} or @epoch@/@mu@ (the latter survives tcsh brace expansion)
            d = a.flow_dir.replace('@epoch@', a.epoch).replace('@mu@', mu)
            d = d.format(epoch=a.epoch, mu=mu)
            print('loading covflow run %s' % d)
            flows[mu] = H.CovFlow(d, mu, device=a.device)
    tmpl, run_branch = H.resolve_branches(a, need_cov=not a.closure_only,
                                          need_bs=not a.closure_only)
    tmpl = H.drop_missing_optional(tmpl, [info['data'], info['mc']], info['tree'], a.muons)
    has_bs = 'bs_dxy' in tmpl and 'bs_dxy_e' in tmpl
    extra_surf = [s for s in km.surfaces if s not in H.KILL_SURFACES]
    if extra_surf and not H.has_masks(tmpl):
        die('the kill maps have %s, but these ntuples have no per-layer hit masks'
            % ' '.join(extra_surf))
    print('hit pattern: %s' % ('per-layer masks (exact context after killing, surfaces %s)'
                               % ' '.join(km.surfaces) if H.has_masks(tmpl) else
                               'no per-layer mask: L1 / D1 only, next crossed layer assumed valid'))

    sel_d, sel_m = H.selection_expr(info, False), H.selection_expr(info, True)
    flow_br = sorted({b for f in flows.values() for b in f.needed_branches()})
    need_d = H.sample_branches(tmpl, a.muons, sel_d, [run_branch])
    tree_br = []
    if a.write_tree and H.has_masks(tmpl):
        # with the masks the emulated tree can also carry the exact n_pix_layer /
        # pix_first_layer, if the ntuple has them (TreeWriter replaces them)
        import uproot
        with uproot.open(info['mc'][0]) as fh:
            have = set(fh[info['tree']].keys())
        tree_br = [b % mu for mu in a.muons for b in ('%s_n_pix_layer', '%s_pix_first_layer')
                   if b % mu in have]
    need_m = H.sample_branches(tmpl, a.muons, sel_m,
                               ([info['mc_weight']] if info['mc_weight'] else []) + flow_br
                               + tree_br)
    entries = P.preflight({'data': info['data'], 'mc': info['mc']}, info['tree'],
                          {'data': need_d, 'mc': need_m})
    out = a.out or 'emu_%s' % a.epoch
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, 'emu_%s' % a.epoch)

    surfaces = list(km.surfaces)
    ctx = {k: Ctx() for k in ('data', 'mc', 'emu')}
    maps = {k: {s: [np.zeros(km.m_num[s].shape) for _ in range(2)] for s in surfaces}
            for k in ('emu',)}
    slices = ('L1', 'D1')
    tgt = {sl: {k: H.RwHist() for k in ('all', 'dead')} for sl in slices}
    mcnat = {sl: H.RwHist() for sl in slices}
    var = {sl: {v: H.RwHist() for v in VARIANTS} for sl in slices}
    sig_d = {sl: RwVar(SIG_EDGES) for sl in slices}
    sig_m = {sl: {v: RwVar(SIG_EDGES) for v in VARIANTS} for sl in slices}
    checks = {sl: {v: dict(n=0.0, clipped=0.0, neg=[], up_dxy=0.0, up_dsz=0.0, notok=0.0,
                           inflated=0.0) for v in VARIANTS[1:]} for sl in slices}
    ab = {sl: dict(h=np.zeros((2, 60, 60)), d=np.zeros((2, 80))) for sl in slices}
    AB_EDGES = np.linspace(-3.6, -0.8, 61)
    D_EDGES = np.linspace(-0.4, 0.4, 81)
    tot = dict(mu=0.0, hitL1=0.0, killL1=0.0, hitD1=0.0, killD1=0.0, nopix=0.0,
               lowpix=0.0, w_ne1=0.0, ev=0.0, w_sum=0.0, w_sum2=0.0, w_max=1.0, w_min=1.0)

    # --------------------------------------------------------------- data
    print('\n[1/2] data: context, no-hit targets')
    t_pass = time.time()
    for chunk, n in H.iterate(info['data'], info['tree'], need_d, a, entries, 'data', timing):
        sel = P.eval_selection(sel_d, chunk, n)
        rix = km.range_of_run(np.asarray(chunk[run_branch])[:n][sel])
        for mu in a.muons:
            v = H.muon_view(chunk, mu, tmpl, a, sel)
            good = H.good_track(v)
            w1 = good.astype(float)
            ctx['data'].fill(v['eta'], v['first_b'], v['first_e'], v['n_pix'], w1)
            if a.closure_only:
                continue
            M = None
            for sl, surfs, nxt in (('L1', ('L1',), v['first_b'] == 2),
                                   ('D1', ('D1+', 'D1-'), v['first_e'] == 2)):
                inmap = np.zeros(len(good), bool)
                eps = np.full(len(good), np.nan)
                for s in surfs:
                    x, y = H.cross_view(s, v)
                    ix, iy, ok = km.lookup(s, x, y)
                    inmap |= ok
                    eps = np.where(ok, km.eps_d[s][rix, ix, iy], eps)
                m = good & inmap & nxt & np.all(np.isfinite(v['cov']), 1)
                if not m.any():
                    continue
                if M is None:
                    M = H.packed_to_matrix(v['cov'])
                zs = H.zsigned(v['z0'], v['eta'])
                with np.errstate(invalid='ignore'):
                    dead = m & (eps <= a.eps_dead)
                for key, mm in (('all', m), ('dead', dead)):
                    tgt[sl][key].fill(v['pt'][mm], v['eta'][mm], v['n_pix'][mm], zs[mm],
                                      M[mm], np.ones(mm.sum()))
                if has_bs:
                    cell, _ = H.rw_cell(v['pt'][m], v['eta'][m], v['n_pix'][m], zs[m])
                    with np.errstate(divide='ignore', invalid='ignore'):
                        sig_d[sl].fill(cell, v['bs_dxy'][m] / v['bs_dxy_e'][m], np.ones(m.sum()))

    # ----------------------------------------------------------------- MC
    print('\n[2/2] MC: emulation%s' % (', routes ' + ' '.join(a.route) if a.route else ''))
    writer = None
    if a.write_tree:
        writer = TreeWriter(stem + '_routeB.root', info, a)
    timing['data_total'] = time.time() - t_pass
    timing['data_read'] = timing.pop('read', 0.0)
    t_pass = time.time()
    for chunk, n in H.iterate(info['mc'], info['tree'], need_m, a, entries, 'MC', timing):
        sel = P.eval_selection(sel_m, chunk, n)
        if info['mc_weight']:
            wpu = np.asarray(chunk[info['mc_weight']], np.float64)[:n]
            sel &= np.isfinite(wpu)
            wpu = wpu[sel]
        else:
            wpu = np.ones(int(sel.sum()))
        ns = int(sel.sum())
        rix = rng.choice(len(km.ranges), size=ns, p=km.lumi)
        w_ev = np.ones(ns)
        per_mu = {}
        for mu in a.muons:
            v = H.muon_view(chunk, mu, tmpl, a, sel)
            good = H.good_track(v)
            e = H.emulate(v, rix, km, rng)
            e['w'] = np.where(good, e['w'], 1.0)
            w_ev *= e['w']
            per_mu[mu] = (v, good, e)
        w_tot = wpu * w_ev
        tot['ev'] += ns
        tot['w_sum'] += float(w_ev.sum())
        tot['w_sum2'] += float((w_ev ** 2).sum())
        tot['w_ne1'] += float((np.abs(w_ev - 1) > 1e-9).sum())
        tot['w_max'] = max(tot['w_max'], float(w_ev.max()) if ns else 1.0)
        tot['w_min'] = min(tot['w_min'], float(w_ev.min()) if ns else 1.0)
        tree_cols = {}
        for mu, (v, good, e) in per_mu.items():
            wg = np.where(good, wpu, 0.0)
            wge = np.where(good, w_tot, 0.0)
            ctx['mc'].fill(v['eta'], v['first_b'], v['first_e'], v['n_pix'], wg)
            ctx['emu'].fill(v['eta'], e['first_b'], e['first_e'], e['n_pix'], wge)
            hL1, hD1 = H.has_hit('L1', v), H.has_hit('D1+', v)
            for s in surfaces:
                hs = H.has_hit(s, v) & e['cell_ok'][s]
                tot['hit_' + s] = tot.get('hit_' + s, 0.0) + float(wg[hs].sum())
                tot['kill_' + s] = tot.get('kill_' + s, 0.0) + float(wg[e['kill'][s]].sum())
            tot['kill_any'] = tot.get('kill_any', 0.0) + float(wg[e['kill_any']].sum())
            tot['kill_other'] = tot.get('kill_other', 0.0) + float(wg[e['kill_other']].sum())
            tot['mu'] += wg.sum()
            tot['hitL1'] += wg[hL1 & e['cell_ok_L1']].sum()
            tot['killL1'] += wg[e['kill_L1']].sum()
            tot['hitD1'] += wg[hD1 & e['cell_ok_D1']].sum()
            tot['killD1'] += wg[e['kill_D1']].sum()
            tot['nopix'] += wge[e['n_pix'] <= 0].sum()
            tot['lowpix'] += wge[e['n_pix'] <= 2].sum()
            # per-cell closure (probe definition as in the maps)
            for s in surfaces:
                x, y = H.cross_view(s, v)
                ix, iy, ok = km.lookup(s, x, y)
                hit0 = H.has_hit(s, v)
                kill = e['kill'][s]
                ok &= good & (H.other_pixel_hits(s, v) >= a.min_other_hits)
                hit1 = (hit0 & ~kill).astype(float)
                H.hist_add(maps['emu'][s][0], (ix[ok], iy[ok]), wge[ok] * hit1[ok])
                H.hist_add(maps['emu'][s][1], (ix[ok], iy[ok]), wge[ok])

            if a.closure_only:
                continue
            routes_out = run_routes(a, v, good, e, chunk, sel, flows.get(mu), he, rng, mu)
            zs = H.zsigned(v['z0'], v['eta'])
            for sl, killed, nat in (('L1', e['kill_L1'], v['first_b'] == 2),
                                    ('D1', e['kill_D1'], v['first_e'] == 2)):
                m0 = good & nat & e['cell_ok_' + sl] & np.all(np.isfinite(v['cov']), 1)
                if m0.any():
                    mcnat[sl].fill(v['pt'][m0], v['eta'][m0], v['n_pix'][m0], zs[m0],
                                   H.packed_to_matrix(v['cov'][m0]), wg[m0])
                m = good & killed & np.all(np.isfinite(v['cov']), 1)
                if not m.any():
                    continue
                fill_route_checks(sl, m, v, e, zs, wge, routes_out, var, checks, sig_m,
                                  ab, AB_EDGES, D_EDGES, has_bs)
            if writer is not None:
                tree_cols[mu] = (v, e, routes_out)
        if writer is not None:
            writer.add(chunk, n, sel, w_ev, rix, tree_cols, a.muons)
    if writer is not None:
        writer.close()

    timing['mc_total'] = time.time() - t_pass
    timing['mc_read'] = timing.pop('read', 0.0)

    # ------------------------------------------------------------- report
    lines, summary = report(a, km, tot, ctx, maps, tgt, mcnat, var, checks, has_bs, he)
    lines += ['', 'TIMING (s): data pass %.0f (reading %.0f), MC pass %.0f (reading %.0f)'
              % (timing['data_total'], timing['data_read'], timing['mc_total'], timing['mc_read'])]
    summary['timing'] = timing
    print('\n' + '\n'.join(lines[2:]))
    for s_ in summary['cells'].values():
        s_.pop('maps', None)
    with open(stem + '.json', 'w') as fh:
        json.dump(summary, fh, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    plots(a, km, lines, ctx, maps, tgt, mcnat, var, sig_d, sig_m, ab, AB_EDGES, D_EDGES,
          has_bs, stem)
    print('\n[out] %s.pdf, %s.json%s   (%.1f min)'
          % (stem, stem, (', %s_routeB.root' % stem) if writer else '', (time.time() - t0) / 60))


def run_routes(a, v, good, e, chunk, sel, flow, he, rng, mu):
    """Route outputs for the muons that lost a hit: packed C' and the smearing."""
    out = {}
    # every muon that lost a hit gets a new covariance from route A (the flows
    # see the full emulated context); route B removes the L1 / D1 hits only
    killed = good & e['kill_any'] & np.all(np.isfinite(v['cov']), 1)
    idx = np.nonzero(killed)[0]
    out['idx'] = idx
    if len(idx) == 0 or not a.route:
        return out
    C0p = v['cov'][idx]
    C0 = H.packed_to_matrix(C0p)
    sub = {k: (x[idx] if isinstance(x, np.ndarray) and x.shape[:1] == good.shape else x)
           for k, x in v.items()}
    if 'B' in a.route:
        pB, okB, infB = H.route_b(C0p, sub, e['kill_L1'][idx], e['kill_D1'][idx], he)
        out['Braw'] = (pB, okB)
        out['inflated'] = infB
        if flow is not None:
            cp = flow.context(chunk, sel, e, sub=idx)
            pBf = np.full_like(pB, np.nan)
            if okB.any():
                pBf[okB] = flow.morph(pB[okB], cp[okB], cp[okB])
            out['Bfin'] = (pBf, okB.copy())
    if 'A' in a.route:
        c = flow.context(chunk, sel, None, sub=idx)
        cp = flow.context(chunk, sel, e, sub=idx)
        pA = flow.morph(C0p, c, cp)
        out['A'] = (pA, np.all(np.isfinite(pA), 1))
    for k in ('A', 'Braw', 'Bfin'):
        if k in out:
            M = H.packed_to_matrix(np.where(np.isfinite(out[k][0]), out[k][0], C0p))
            delta, clipped, neg = H.smear(M, C0, rng)
            out[k] = out[k] + (M, delta, clipped, neg)
    out['C0'] = C0
    return out


def fill_route_checks(sl, m, v, e, zs, w, ro, var, checks, sig_m, ab, AB_EDGES, D_EDGES, has_bs):
    idx = ro['idx']
    pos = np.searchsorted(idx, np.nonzero(m)[0])
    cell, _ = H.rw_cell(v['pt'][m], v['eta'][m], e['n_pix'][m], zs[m])
    wm = w[m]
    if 'C0' not in ro:
        if len(idx):
            C0 = H.packed_to_matrix(v['cov'][m])
            var[sl]['orig'].fill(v['pt'][m], v['eta'][m], e['n_pix'][m], zs[m], C0, wm)
        return
    C0 = ro['C0'][pos]
    var[sl]['orig'].fill(v['pt'][m], v['eta'][m], e['n_pix'][m], zs[m], C0, wm)
    s0 = H.sigmas(C0)
    if has_bs:
        with np.errstate(divide='ignore', invalid='ignore'):
            sig_m[sl]['orig'].fill(cell, v['bs_dxy'][m] / v['bs_dxy_e'][m], wm)
    for k in ('A', 'Braw', 'Bfin'):
        if k not in ro:
            continue
        packed, ok, M, delta, clipped, neg = ro[k]
        ok, M, delta, clipped, neg = ok[pos], M[pos], delta[pos], clipped[pos], neg[pos]
        c = checks[sl][k]
        c['n'] += wm.sum()
        c['notok'] += wm[~ok].sum()
        if k in ('Braw', 'Bfin') and 'inflated' in ro:
            c['inflated'] += wm[ok & ro['inflated'][pos]].sum()
        c['clipped'] += wm[ok & clipped].sum()
        c['neg'].append(neg[ok & clipped])
        s1 = H.sigmas(M)
        c['up_dxy'] += wm[ok & (s1[:, 3] >= s0[:, 3] * (1 - 1e-9))].sum()
        c['up_dsz'] += wm[ok & (s1[:, 4] >= s0[:, 4] * (1 - 1e-9))].sum()
        var[sl][k].fill(v['pt'][m][ok], v['eta'][m][ok], e['n_pix'][m][ok], zs[m][ok],
                        M[ok], wm[ok])
        if has_bs:
            bse = v['bs_dxy_e'][m]
            den = np.sqrt(np.clip(bse ** 2 - s0[:, 3] ** 2, 0, None) + s1[:, 3] ** 2)
            with np.errstate(divide='ignore', invalid='ignore'):
                sig = (v['bs_dxy'][m] + delta[:, 3]) / den
            sig_m[sl][k].fill(cell[ok], sig[ok], wm[ok])
    if 'A' in ro and 'Bfin' in ro:
        sA = H.sigmas(ro['A'][2][pos])
        sB = H.sigmas(ro['Bfin'][2][pos])
        good = ro['A'][1][pos] & ro['Bfin'][1][pos]
        with np.errstate(divide='ignore', invalid='ignore'):
            for j, col in ((0, 3), (1, 4)):
                la, lb = np.log10(sA[good, col]), np.log10(sB[good, col])
                h, _, _ = np.histogram2d(la, lb, bins=(AB_EDGES, AB_EDGES), weights=wm[good])
                ab[sl]['h'][j] += h
                ab[sl]['d'][j] += np.histogram(la - lb, bins=D_EDGES, weights=wm[good])[0]


class TreeWriter:
    """Selected MC events with the emulated context and route B,raw covariances,
    under the ORIGINAL branch names, so that covflow can be trained on it
    unchanged. Everything replaced is listed in the companion json."""

    CTX = {'n_pix_hit': 'n_pix', 'n_pix_b_hit': 'n_pix_b', 'n_pix_e_hit': 'n_pix_e',
           'pix_first_b_layer': 'first_b', 'pix_first_e_disk': 'first_e',
           # replaced only when the ntuple has the per-layer masks
           'n_pix_layer': 'n_pix_layer', 'pix_first_layer': 'first_layer',
           'pix_valid_mask': 'mask', 'pix_hit_count': 'count',
           'pix_miss_mask': 'miss_mask', 'pix_inact_mask': 'inact_mask'}

    def __init__(self, path, info, a):
        import uproot
        self.path, self.info, self.a = path, info, a
        self.f = uproot.recreate(path)
        self.made = False
        self.n = 0

    def add(self, chunk, n, sel, w_ev, rix, cols, muons):
        out = {k: np.asarray(v)[:n][sel] for k, v in chunk.items()}
        wname = self.info['mc_weight']
        if wname:
            out[wname + '_noemu'] = out[wname].copy()
            out[wname] = out[wname] * w_ev
        out['emu_weight'] = w_ev
        out['emu_range'] = rix
        for mu in muons:
            v, e, ro = cols[mu]
            for b, k in self.CTX.items():
                name = '%s_%s' % (mu, b)
                if name in out and k in e:
                    out['%s_orig_%s' % (mu, b)] = out[name].copy()
                    out[name] = e[k]
            out['%s_emu_kill_L1' % mu] = e['kill_L1'].astype(np.float32)
            out['%s_emu_kill_D1' % mu] = e['kill_D1'].astype(np.float32)
            kmask = np.zeros(len(e['w']), np.int64)
            for s_, ks in e['kill'].items():
                kmask |= np.where(ks, 1 << H.MASK_BIT[s_[:2]], 0)
            out['%s_emu_kill_mask' % mu] = kmask.astype(np.float32)
            idx = ro.get('idx', np.zeros(0, int))
            okB = np.ones(len(e['w']), np.float32)
            dl = np.zeros((len(e['w']), 5), np.float32)
            if 'Braw' in ro and len(idx):
                packed, ok, M, delta, clipped, neg = ro['Braw']
                for j, nm in enumerate(H.PACK_NAMES):
                    name = '%s_cov_%s' % (mu, nm)
                    col = np.asarray(out[name], np.float64).copy()
                    col[idx] = np.where(ok, packed[:, j], col[idx])
                    out[name] = col
                okB[idx] = ok
                dl[idx] = np.nan_to_num(delta)
            out['%s_emu_routeB_ok' % mu] = okB
            for j, p in enumerate(PARAMS):
                out['%s_emu_delta_%s' % (mu, p)] = dl[:, j]
        out = {k: (np.asarray(v).astype(np.int64) if k in ('run', 'lumi', 'event', 'emu_range')
                   else np.asarray(v).astype(np.float32)) for k, v in out.items()}
        if not self.made:
            self.f.mktree('tree', {k: v.dtype for k, v in out.items()})
            self.keys = list(out)
            self.made = True
        self.f['tree'].extend({k: out[k] for k in self.keys})
        self.n += len(next(iter(out.values())))

    def close(self):
        self.f.close()
        meta = dict(
            epoch=self.a.epoch, killmaps=os.path.abspath(self.a.killmaps),
            hit_errors=[os.path.abspath(x) for x in self.a.hit_errors], seed=self.a.seed,
            rows=self.n, selection_applied=True,
            replaced=dict(
                context='{mu}_' + ', {mu}_'.join(self.CTX) + ' -> emulated '
                        '(originals in {mu}_orig_*)',
                covariance='{mu}_cov_* -> route B,raw for muons that lost a hit',
                weight='%s -> %s * emu_weight (original in %s_noemu)'
                       % (self.info['mc_weight'], self.info['mc_weight'], self.info['mc_weight'])),
            added=['emu_weight', 'emu_range', '{mu}_emu_kill_L1', '{mu}_emu_kill_D1',
                   '{mu}_emu_kill_mask (layers whose hits were removed, same bits as '
                   '{mu}_pix_valid_mask)',
                   '{mu}_emu_routeB_ok', '{mu}_emu_delta_<param> (route B smearing, '
                   'NOT applied to any branch)'])
        with open(self.path.replace('.root', '.json'), 'w') as fh:
            json.dump(meta, fh, indent=1)


def report(a, km, tot, ctx, maps, tgt, mcnat, var, checks, has_bs, he=None):
    L = ['Hit-loss emulation, epoch %s' % a.epoch, '',
         'MC events assigned to run ranges with the luminosity shares %s'
         % ' '.join('%.3f' % x for x in km.lumi),
         'selected MC events %d; event weight from data > MC cells: mean %.4f, min %.3f, '
         'max %.3f, %.1f%% of events != 1, effective sample size %.3f of the events'
         % (tot['ev'], tot['w_sum'] / max(tot['ev'], 1), tot['w_min'], tot['w_max'],
            100 * tot['w_ne1'] / max(tot['ev'], 1),
            tot['w_sum'] ** 2 / max(tot['ev'] * tot['w_sum2'], 1e-300)),
         'hits removed, as a fraction of the MC hits on mapped cells: %s'
         % ', '.join('%s %.4f' % (s, tot.get('kill_' + s, 0) / max(tot.get('hit_' + s, 0), 1))
                     for s in km.surfaces),
         'muons that lost at least one hit: %.4f; of these, with a loss on L2-L4 / D2-D3: %.4f%s'
         % (tot.get('kill_any', 0) / max(tot['mu'], 1),
            tot.get('kill_other', 0) / max(tot.get('kill_any', 0), 1e-300),
            ' (route B removes only the L1 / D1 hits of these)' if a.route and 'B' in a.route else ''),
         'muons left with no pixel hit: %.4f, with <= 2: %.4f (kept; the reconstruction '
         'would probably have lost some)' % (tot['nopix'] / max(tot['mu'], 1),
                                             tot['lowpix'] / max(tot['mu'], 1)),
         '', 'PER-CELL CLOSURE: hit efficiency in acceptance, every sample weighted with the '
         'DATA probes of each cell']
    summary = dict(epoch=a.epoch, totals=tot, cells={}, context={}, routes={})
    L += ['  data = measured data hits / probes (raw counts, not the kill-map values);',
          '  "measured cells" = cells with >= %g data probes over the epoch, where the '
          'per-cell comparison is meaningful' % km.min_cell,
          '                    all cells                          |  measured cells only',
          '        data    MC before  MC emulated  emu-data       |  data    MC emulated  emu-data']
    for s in km.surfaces:
        r = cell_closure(km, maps, s)
        L.append('  %-4s  %.4f  %.4f     %.4f       %+.4f        |  %.4f  %.4f       %+.4f'
                 % (s, r['data'], r['mc'], r['emu'], r['emu'] - r['data'],
                    r['data_meas'], r['emu_meas'], r['emu_meas'] - r['data_meas']))
        summary['cells'][s] = r
    L += ['', 'CONTEXT CLOSURE: total variation from data (0 = identical), per |eta| bin, '
          'averaged with the data |eta| spectrum',
          '  (first BPix layer x first FPix disk):   MC before %.4f   MC emulated %.4f'
          % (tv_eta(ctx['data'].joint, ctx['mc'].joint), tv_eta(ctx['data'].joint, ctx['emu'].joint)),
          '  pixel hits:                             MC before %.4f   MC emulated %.4f'
          % (tv_eta(ctx['data'].npix, ctx['mc'].npix), tv_eta(ctx['data'].npix, ctx['emu'].npix))]
    for k in ('data', 'mc', 'emu'):
        j = ctx[k].joint.sum(0)
        summary['context'][k] = dict(first_b=(j.sum(1) / max(j.sum(), 1)).tolist(),
                                     first_e=(j.sum(0) / max(j.sum(), 1)).tolist())
    L.append('  fraction with first BPix layer 0/1/2/3/4:')
    for k, lab in (('data', 'data'), ('mc', 'MC before'), ('emu', 'MC emulated')):
        L.append('     %-12s %s' % (lab, ' '.join('%.4f' % x for x in summary['context'][k]['first_b'])))
    L.append('  fraction with first FPix disk 0/1/2/3:')
    for k, lab in (('data', 'data'), ('mc', 'MC before'), ('emu', 'MC emulated')):
        L.append('     %-12s %s' % (lab, ' '.join('%.4f' % x for x in summary['context'][k]['first_e'])))
    L.append('  fraction with pixel hits 0/1/2/3/4/5/>=6:')
    for k, lab in (('data', 'data'), ('mc', 'MC before'), ('emu', 'MC emulated')):
        h = ctx[k].npix.sum(0)
        h = h / max(h.sum(), 1e-300)
        f = list(h[:6]) + [h[6:].sum()]
        summary['context'][k]['n_pix'] = h.tolist()
        L.append('     %-12s %s' % (lab, ' '.join('%.4f' % x for x in f)))
    if he is not None:
        L += [''] + he.describe()
        for fam, (path, sample, _e) in he.choice.items():
            if sample != 'mc':
                L.append('  NOTE: %s uses hit errors fitted on %s; B,raw is meant to be '
                         'MC-like (hit errors fitted on MC)' % (fam, sample))
    if a.route:
        L += ['', 'ROUTES, muons that lost the hit. shift = median log10 sigma(MC) - median(data '
              'target), MC reweighted to the target in (pt, |eta|, pixel hits, z0*sign(eta))',
              '  failed   : no C\' (route B: V - H C H^T not PD even with V x 30)',
              '  inflated : route B needed a larger V than its |eta| bin for this track',
              '  not PD   : C\' - C has a negative eigenvalue > 0.1% of the largest '
              '(clipped for the smearing)',
              '  grows    : fraction with sigma\'(dxy) >= sigma(dxy) / sigma\'(dsz) >= sigma(dsz)']
        for sl in ('L1', 'D1'):
            for k in ('A', 'Braw', 'Bfin'):
                c = checks[sl][k]
                if c['n'] <= 0:
                    continue
                negs = np.concatenate(c['neg']) if c['neg'] else np.zeros(0)
                row = ('  %s %-21s failed %.4f  inflated %.4f  not PD %.4f (median %.2g)  '
                       'grows %.3f / %.3f'
                       % (sl, VLABEL[k].replace('route ', ''), c['notok'] / c['n'],
                          c['inflated'] / c['n'], c['clipped'] / c['n'],
                          float(np.median(negs)) if len(negs) else 0.0,
                          c['up_dxy'] / c['n'], c['up_dsz'] / c['n']))
                L.append(row)
            for tkey, tlab in (('dead', 'data, dead cells (random losses)'),
                               ('all', 'data, all no-hit tracks')):
                T = tgt[sl][tkey]
                if T.n <= 0:
                    continue
                L.append('  %s vs %s: shift dxy / dsz / qoverp  (KS dxy)' % (sl, tlab))
                for k in VARIANTS:
                    V = var[sl][k]
                    if V.n <= 0:
                        continue
                    sh = []
                    for p in ('dxy', 'dsz', 'qoverp'):
                        hv, ht = V.projected(p, target=T), T.projected(p)
                        sh.append(H.quantiles(hv, H.LOGSIG_EDGES[p])[1]
                                  - H.quantiles(ht, H.LOGSIG_EDGES[p])[1])
                    ks = H.ks_distance(V.projected('dxy', target=T), T.projected('dxy'))
                    L.append('     %-24s %+.4f / %+.4f / %+.4f   (%.3f)' % (VLABEL[k], *sh, ks))
                    summary['routes'].setdefault(sl, {}).setdefault(tkey, {})[k] = dict(
                        shift_dxy=sh[0], shift_dsz=sh[1], shift_qoverp=sh[2], ks_dxy=ks)
            if mcnat[sl].n > 0 and var[sl]['Braw'].n > 0:
                sh = []
                for p in ('dxy', 'dsz'):
                    hv = var[sl]['Braw'].projected(p, target=mcnat[sl])
                    sh.append(H.quantiles(hv, H.LOGSIG_EDGES[p])[1]
                              - H.quantiles(mcnat[sl].projected(p), H.LOGSIG_EDGES[p])[1])
                L.append('  %s closure in MC: route B,raw vs MC tracks without the hit: '
                         'shift dxy %+.4f, dsz %+.4f' % (sl, *sh))
    return L, summary


def plots(a, km, lines, ctx, maps, tgt, mcnat, var, sig_d, sig_m, ab, AB_EDGES, D_EDGES,
          has_bs, stem):
    plt = P.setup_mpl()
    book = P.Book(stem + '.pdf', png_prefix=stem if a.png else None)
    P.page_text(book, lines, 'summary')
    nb_phi = len(km.edges['L1'][1]) - 1
    for s in km.surfaces:
        surf = P.Surface(s, type('A', (), dict(nbins_phi=nb_phi,
                                               nbins_r=len(km.edges[s][0]) - 1
                                               if s[0] == 'D' else 20))())
        surf.xe, surf.ye = km.edges[s]
        r = cell_closure(km, maps, s)
        e_dat, e_emu, diff = r['maps']
        fig = plt.figure(figsize=(18, 5.6) if s == 'L1' else (18, 6.2))
        P.draw_map(fig, (1, 3, 1), surf, e_dat, 'data, measured (cells with >= %g probes)'
                   % km.min_cell, 'viridis', 0, 1, 'hit efficiency (eps)')
        P.draw_map(fig, (1, 3, 2), surf, e_emu, 'MC after emulation', 'viridis', 0, 1,
                   'hit efficiency (eps)')
        P.draw_map(fig, (1, 3, 3), surf, diff, 'hit efficiency, emulated MC - data (measured cells): %+.4f'
                   % (r['emu_meas'] - r['data_meas']), 'RdBu_r', -0.1, 0.1, 'hit efficiency difference')
        fig.suptitle('%s per-cell closure, epoch %s' % (s, a.epoch), fontsize=13)
        fig.tight_layout()
        book.add(fig, '%s_closure' % s.replace('+', 'p').replace('-', 'm'))

    # context
    fig, axes = plt.subplots(3, len(CTX_ETA) - 1, figsize=(20, 12))
    style = {'data': ('k', 'o', 'data'), 'mc': ('0.6', 's', 'MC before'),
             'emu': ('#d95f02', 'D', 'MC emulated')}
    for i in range(len(CTX_ETA) - 1):
        for row, (dim, edges, lab) in enumerate(((1, FB_EDGES, 'first BPix layer'),
                                                 (0, FE_EDGES, 'first FPix disk'))):
            ax = axes[row, i]
            for k, (c, mk, lb) in style.items():
                j = ctx[k].joint[i]
                hh = j.sum(axis=1) if row == 0 else j.sum(axis=0)
                hh = hh / max(hh.sum(), 1e-300)
                ax.plot(np.arange(len(hh)), hh, mk + '-', color=c, label=lb, ms=5)
            ax.set_yscale('log')
            ax.set_ylim(1e-4, 1.5)
            ax.set_xlabel(lab)
            if row == 0:
                ax.set_title('%.1f < |eta| < %.1f' % (CTX_ETA[i], CTX_ETA[i + 1]))
        ax = axes[2, i]
        for k, (c, mk, lb) in style.items():
            hh = ctx[k].npix[i]
            hh = hh / max(hh.sum(), 1e-300)
            ax.plot(np.arange(len(hh)), hh, mk + '-', color=c, label=lb, ms=5)
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 1.5)
        ax.set_xlabel('pixel hits')
        axes[0, 0].legend(fontsize=9)
    fig.suptitle('Context: hit pattern, data vs MC before / after emulation, epoch %s' % a.epoch,
                 fontsize=13)
    fig.tight_layout()
    book.add(fig, 'context')

    if a.route or any(var[sl]['orig'].n > 0 for sl in var):
        for sl in ('L1', 'D1'):
            T = tgt[sl]['dead'] if tgt[sl]['dead'].n > 0 else tgt[sl]['all']
            if T.n <= 0 or var[sl]['orig'].n <= 0:
                continue
            fig, axes = plt.subplots(2, 3, figsize=(17, 9))
            for ax, p in zip(axes.flat, PARAMS):
                e = H.LOGSIG_EDGES[p]
                xc = 0.5 * (e[1:] + e[:-1])
                ax.step(xc, tgt[sl]['all'].projected(p), where='mid', color='k', lw=1,
                        ls=':', label='data, all without %s' % sl)
                ax.step(xc, T.projected(p), where='mid', color='k', lw=1.6,
                        label='data, dead cells (target)')
                for k, c, ls in (('orig', '0.6', '--'), ('A', '#1b9e77', '-'),
                                 ('Braw', '#d95f02', '--'), ('Bfin', '#d95f02', '-')):
                    if var[sl][k].n > 0:
                        ax.step(xc, var[sl][k].projected(p, target=T), where='mid', color=c,
                                ls=ls, label=VLABEL[k])
                if mcnat[sl].n > 0:
                    ax.step(xc, mcnat[sl].projected(p, target=T), where='mid', color='#7570b3',
                            ls=':', label='MC, natural no-%s' % sl)
                ax.set_xlabel('log10 sigma(%s)' % p)
            axes.flat[0].legend(fontsize=8)
            axes.flat[-1].axis('off')
            axes.flat[-1].text(0, 0.9, 'muons that lost the %s hit in the emulation\n'
                               'MC reweighted to the data target in\n(pt, |eta|, pixel hits, '
                               'z0*sign(eta))' % sl, va='top')
            fig.suptitle('%s: covariance of the tracks without the hit, epoch %s'
                         % (sl, a.epoch), fontsize=13)
            fig.tight_layout()
            book.add(fig, '%s_routes' % sl)

            if ab[sl]['h'].sum() > 0:
                fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
                for j, p in enumerate(('dxy', 'dsz')):
                    hh = np.ma.masked_equal(ab[sl]['h'][j].T, 0)
                    with np.errstate(divide='ignore'):
                        axes[j].pcolormesh(AB_EDGES, AB_EDGES, np.log10(hh), cmap='viridis')
                    axes[j].plot(AB_EDGES, AB_EDGES, 'r--', lw=0.8)
                    axes[j].set_xlabel('log10 sigma(%s), route A' % p)
                    axes[j].set_ylabel('log10 sigma(%s), route B final' % p)
                xc = 0.5 * (D_EDGES[1:] + D_EDGES[:-1])
                for j, (p, c) in enumerate((('dxy', 'k'), ('dsz', '#d95f02'))):
                    d = ab[sl]['d'][j]
                    axes[2].step(xc, d / max(d.sum(), 1e-300), where='mid', color=c, label=p)
                axes[2].set_xlabel('log10 sigma(A) - log10 sigma(B final), per track')
                axes[2].legend()
                fig.suptitle('%s: route A against route B, track by track, epoch %s'
                             % (sl, a.epoch), fontsize=13)
                fig.tight_layout()
                book.add(fig, '%s_AvsB' % sl)

            if has_bs and sig_d[sl].h.sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                xc = 0.5 * (SIG_EDGES[1:] + SIG_EDGES[:-1])
                ax.step(xc, sig_d[sl].projected(), where='mid', color='k', lw=1.6,
                        label='data, all without %s' % sl)
                for k, c, ls in (('orig', '0.6', '--'), ('A', '#1b9e77', '-'),
                                 ('Braw', '#d95f02', '--'), ('Bfin', '#d95f02', '-')):
                    if sig_m[sl][k].h.sum() > 0:
                        ax.step(xc, sig_m[sl][k].projected(target=sig_d[sl]), where='mid',
                                color=c, ls=ls, label=VLABEL[k])
                ax.set_yscale('log')
                ax.set_xlabel('bs_dxy / sigma (MC: parameters smeared by N(0, C\' - C))')
                ax.legend(fontsize=9)
                ax.set_title('%s: beamspot IP significance of the muons without the hit, epoch %s'
                             % (sl, a.epoch))
                fig.tight_layout()
                book.add(fig, '%s_ipsig' % sl)
    book.close()


if __name__ == '__main__':
    main()
