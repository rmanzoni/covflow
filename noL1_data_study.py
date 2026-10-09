#!/usr/bin/env python3
"""
noL1_data_study.py -- step 4 of the hit-loss emulation, and the input of route B.

For one epoch, on data (default) or on MC (--sample mc), it compares tracks
that have / do not have a hit on L1 (and, separately, on D1), sorted by the
efficiency of the cell they cross, taken from the kill maps:

  W1  hit present,  cell working (eps >= --eps-working)   source of route B
  W0  hit missing,  cell working                          losses NOT from dead
                                                           readout: rejected hit,
                                                           scattering, ...
  D0  hit missing,  cell dead    (eps <= --eps-dead)      random losses: what the
                                                           emulation creates
A "missing" L1 hit means first BPix layer == 2 (L2 present), and likewise
first FPix disk == 2 for D1, so the three groups differ by one hit only.
For data the cell efficiency is that of the event's run range; for MC it is
eps_MC.

1. MIXTURE TEST. log10(sigma) of each track parameter, W0 against D0, with W0
   reweighted to D0 in (pt, |eta|, other pixel hits). If they agree, the data
   tracks without L1 are one population and route A's target is clean. If W0
   is wider or shifted, data no-L1 tracks are a mixture, and route A would give
   the emulated (random-loss) tracks the wrong tails.

2. ROUTE B HIT ERRORS. For each |eta| bin, the effective hit covariance
   V = diag(sigma_u^2, sigma_v^2) (u = r*phi, v = z on a layer / r on a disk)
   for which removing the hit from the W1 covariances,
       C' = (C^-1 - H^T V^-1 H)^-1,
   reproduces the median log10 sigma_dxy and sigma_dsz of D0 (W1 reweighted to
   D0 as above). H comes from the same helix as the maps. A two-parameter grid
   search, coarse then fine. The 16% and 84% quantiles are reported as a check
   that is not fitted. --split-test fits on the cells with even phi index and
   evaluates on the odd ones.
   -> hiterrors_<sample>_<epoch>.json, read by emulate_hit_loss.py --route B

Usage (tcsh, UI, covflow env):
  python noL1_data_study.py --epoch 2026 --killmaps killmaps_2026/killmaps_2026 --split-test
  python noL1_data_study.py --epoch 2026 --killmaps killmaps_2026/killmaps_2026 --sample mc
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
FAMILIES = {'L1': ('L1',), 'D1': ('D1+', 'D1-')}
PARAMS = list(H.F.PARAM_NAMES)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    H.add_common_args(p, DEFAULT_CFG)
    p.add_argument('--killmaps', required=True, help='stem of killmaps_<epoch>.npz/.json')
    p.add_argument('--sample', choices=('data', 'mc'), default='data')
    p.add_argument('--eps-working', type=float, default=0.95)
    p.add_argument('--eps-dead', type=float, default=0.4)
    p.add_argument('--eta-edges', type=float, nargs='+',
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    p.add_argument('--fit-tracks', type=int, default=20000,
                   help='W1 and D0 tracks kept per |eta| bin for the V fit')
    p.add_argument('--split-test', action='store_true')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', default=None, help='default: noL1_<sample>_<epoch>')
    p.add_argument('--png', action='store_true')
    return p.parse_args()


class Reservoir:
    """Uniform random subsample of at most `cap` rows over all chunks."""

    def __init__(self, cap, rng):
        self.cap, self.rng, self.d, self.key, self.seen = cap, rng, None, None, 0

    def add(self, cols):
        n = len(next(iter(cols.values())))
        if n == 0:
            return
        self.seen += n
        k = self.rng.random(n)
        if self.d is None:
            self.d, self.key = {c: np.asarray(v) for c, v in cols.items()}, k
        else:
            self.d = {c: np.concatenate([self.d[c], np.asarray(cols[c])]) for c in self.d}
            self.key = np.concatenate([self.key, k])
        if len(self.key) > self.cap:
            keep = np.argpartition(self.key, self.cap)[:self.cap]
            self.d = {c: v[keep] for c, v in self.d.items()}
            self.key = self.key[keep]

    def get(self):
        return self.d or {}


def rw_weights(src, dst):
    """Per-track weights taking src to dst's distribution in
    (pt, |eta|, npix, z0*sign(eta)); dst cells without src get no weight."""
    cs, shape = H.rw_cell(src['pt'], src['eta'], src['npix'], H.zsigned(src['z0'], src['eta']))
    cd, _ = H.rw_cell(dst['pt'], dst['eta'], dst['npix'], H.zsigned(dst['z0'], dst['eta']))
    nc = int(np.prod(shape))
    hs = np.bincount(cs[cs >= 0], minlength=nc).astype(float)
    hd = np.bincount(cd[cd >= 0], minlength=nc).astype(float)
    hd /= max(hd.sum(), 1)
    hs_n = hs / max(hs.sum(), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(hs_n > 0, hd / hs_n, 0.0)
    return np.where(cs >= 0, f[np.maximum(cs, 0)], 0.0)


def fine_match(src, dst, zs_width=1.0, eta_width=0.1):
    """
    Weights for the V fit: src (W1) reweighted to dst (D0) in fine cells of
    (log pt, |eta|, other pixel hits, z0*sign(eta)), and the dst tracks kept.

    Fine because the covariance of a track depends on exactly which layers and
    disks it crosses, i.e. on where it starts along the beam, and dead cells
    select narrow bands of that. dst tracks in cells without any src track are
    dropped from the comparison (both sides then cover the same tracks), and
    the fraction dropped is returned.
    """
    def cell(t):
        lp = np.clip(np.searchsorted(H.PT_RW, t['pt'], 'right') - 1, 0, len(H.PT_RW) - 2)
        ae = np.floor(np.abs(t['eta']) / eta_width).astype(np.int64)
        zs = np.floor(np.clip(H.zsigned(t['z0'], t['eta']), -30, 30) / zs_width).astype(np.int64)
        npx = np.clip(t['npix'], 0, 15).astype(np.int64)
        return ((lp * 1000 + ae) * 100 + (zs + 50)) * 16 + npx
    cs, cd = cell(src), cell(dst)
    us, inv_s, ns = np.unique(cs, return_inverse=True, return_counts=True)
    pos = np.searchsorted(us, cd)
    pos = np.clip(pos, 0, len(us) - 1)
    has = us[pos] == cd
    keep_dst = has
    ud, nd = np.unique(cd[keep_dst], return_counts=True)
    look = dict(zip(ud.tolist(), nd.tolist()))
    w = np.array([look.get(c, 0) for c in us.tolist()], float)[inv_s] / ns[inv_s]
    return w, keep_dst, 1.0 - keep_dst.mean()


def wquant(x, w, qs):
    ok = np.isfinite(x) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return [np.nan] * len(qs)
    o = np.argsort(x)
    c = np.cumsum(w[o])
    c /= c[-1]
    return [float(np.interp(q, c, x[o])) for q in qs]


def predict(src, su, sv):
    """log10 sigma_dxy, sigma_dsz of the W1 sample after removing the hit."""
    V = np.zeros((len(src['eta']), 2, 2))
    V[:, 0, 0], V[:, 1, 1] = su ** 2, sv ** 2
    Cp, ok = H.remove_hit(src['C'], src['H'], V)
    sg = H.sigmas(Cp)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.log10(sg[:, 3]), np.log10(sg[:, 4]), ok


def fit_bin(src, dst, rng):
    """Grid search for (sigma_u, sigma_v) in cm, coarse then fine."""
    w, keep, dropped = fine_match(src, dst)
    one = keep.astype(float)
    t_dxy = wquant(dst['lsig'][:, 3], one, (0.16, 0.5, 0.84))
    t_dsz = wquant(dst['lsig'][:, 4], one, (0.16, 0.5, 0.84))

    def loss(su, sv):
        ld, lz, ok = predict(src, su, sv)
        m_d = wquant(ld, w * ok, (0.5,))[0]
        m_z = wquant(lz, w * ok, (0.5,))[0]
        bad = 1 - (w * ok).sum() / max(w.sum(), 1)
        # A few tracks with V - H C H^T not positive definite are expected at
        # the true V (V is one number per |eta| bin, the tracks' true errors
        # vary inside it); they are left out of the medians. Only a sizeable
        # fraction means V is too small.
        return (m_d - t_dxy[1]) ** 2 + (m_z - t_dsz[1]) ** 2 + 10 * max(0.0, bad - 0.02), bad

    lo, hi = 2e-4, 5e-2          # 2 um .. 500 um
    gu = np.geomspace(lo, hi, 15)
    gv = np.geomspace(lo, hi, 15)
    best = (np.inf, None, None)
    for span in (2.0, 1.3, 1.08, 1.02, None):     # each pass zooms in
        for su in gu:
            for sv in gv:
                l, _b = loss(su, sv)
                if l < best[0]:
                    best = (l, su, sv)
        if span is None:
            break
        su0, sv0 = best[1], best[2]
        gu = np.geomspace(max(lo, su0 / span), min(hi, su0 * span), 11)
        gv = np.geomspace(max(lo, sv0 / span), min(hi, sv0 * span), 11)
    su, sv = best[1], best[2]
    # at the edge of the range = the data do not constrain that error
    edge = [n for n, x in (('sigma_u', su), ('sigma_v', sv))
            if x <= lo * 1.05 or x >= hi / 1.05]
    ld, lz, ok = predict(src, su, sv)
    q_d = wquant(ld, w * ok, (0.16, 0.5, 0.84))
    q_z = wquant(lz, w * ok, (0.16, 0.5, 0.84))
    _, bad = loss(su, sv)
    return dict(sigma_u_cm=float(su), sigma_v_cm=float(sv), loss=float(best[0]),
                unconstrained=edge, dst_dropped=float(dropped),
                frac_not_pd=float(bad), target_dxy=t_dxy, target_dsz=t_dsz,
                pred_dxy=q_d, pred_dsz=q_z, n_src=int(len(src['eta'])),
                n_dst=int(len(dst['eta'])))


def main():
    a = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    info = P.load_epoch(a)
    km = H.KillMaps.load(a.killmaps)
    if km.meta.get('epoch') != a.epoch:
        die('kill maps are for epoch %s, not %s' % (km.meta.get('epoch'), a.epoch))
    tmpl, run_branch = H.resolve_branches(a, need_cov=True)
    is_mc = a.sample == 'mc'
    files = info['mc'] if is_mc else info['data']
    sel_expr = H.selection_expr(info, is_mc)
    extra = ([info['mc_weight']] if (is_mc and info['mc_weight']) else []) + \
            ([] if is_mc else [run_branch])
    need = H.sample_branches(tmpl, a.muons, sel_expr, extra)
    entries = P.preflight({a.sample: files}, info['tree'], {a.sample: need})
    out = a.out or 'noL1_%s_%s' % (a.sample, a.epoch)
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, 'noL1_%s_%s' % (a.sample, a.epoch))

    pops = ('W1', 'W0', 'D0')
    hist = {f: {p: H.RwHist() for p in pops} for f in FAMILIES}
    neta = len(a.eta_edges) - 1
    halves = ('even', 'odd') if a.split_test else ('all',)
    res = {f: {h: {pop: [Reservoir(a.fit_tracks, rng) for _ in range(neta)]
                   for pop in ('W1', 'D0')} for h in halves} for f in FAMILIES}
    counts = {f: {p: 0.0 for p in pops + ('mid0', 'mid1')} for f in FAMILIES}

    print('no-L1 study, epoch %s, sample %s' % (a.epoch, a.sample))
    for chunk, n in H.iterate(files, info['tree'], need, a, entries, a.sample):
        sel = P.eval_selection(sel_expr, chunk, n)
        if is_mc and info['mc_weight']:
            wev = np.asarray(chunk[info['mc_weight']], np.float64)[:n]
            sel &= np.isfinite(wev)
            wev = wev[sel]
        else:
            wev = np.ones(int(sel.sum()))
        if not is_mc:
            rix = km.range_of_run(np.asarray(chunk[run_branch])[:n][sel])
        for mu in a.muons:
            v = H.muon_view(chunk, mu, tmpl, a, sel)
            good = H.good_track(v) & np.all(np.isfinite(v['cov']), 1)
            for fam, surfs in FAMILIES.items():
                eps = np.full(len(good), np.nan)
                cell_par = np.zeros(len(good), np.int64)
                which = np.full(len(good), '', dtype=object)
                for s in surfs:
                    x, y = H.cross_view(s, v)
                    ix, iy, ok = km.lookup(s, x, y)
                    ok &= good
                    e = km.eps_m[s][ix, iy] if is_mc else km.eps_d[s][rix, ix, iy]
                    eps = np.where(ok, e, eps)
                    cell_par = np.where(ok, iy % 2, cell_par)
                    which = np.where(ok, s, which)
                hit = H.has_hit(surfs[0], v)
                nxt = (v['first_b'] == 2) if fam == 'L1' else (v['first_e'] == 2)
                npix = v['n_pix'] - hit
                with np.errstate(invalid='ignore'):
                    work, dead = eps >= a.eps_working, eps <= a.eps_dead
                masks = dict(W1=work & hit, W0=work & ~hit & nxt, D0=dead & ~hit & nxt)
                counts[fam]['mid0'] += float(wev[~work & ~dead & np.isfinite(eps) & ~hit].sum())
                counts[fam]['mid1'] += float(wev[~work & ~dead & np.isfinite(eps) & hit].sum())
                M = None
                for pop, m in masks.items():
                    if not m.any():
                        continue
                    counts[fam][pop] += float(wev[m].sum())
                    if M is None:
                        M = H.packed_to_matrix(v['cov'])
                    hist[fam][pop].fill(v['pt'][m], v['eta'][m], npix[m],
                                        H.zsigned(v['z0'][m], v['eta'][m]), M[m], wev[m])
                for pop in ('W1', 'D0'):
                    m = masks[pop]
                    if not m.any():
                        continue
                    ae = np.abs(v['eta'])
                    for b in range(neta):
                        mb = m & (ae >= a.eta_edges[b]) & (ae < a.eta_edges[b + 1])
                        for h in halves:
                            mh = mb if h == 'all' else mb & (cell_par == (0 if h == 'even' else 1))
                            if not mh.any():
                                continue
                            cols = dict(pt=v['pt'][mh], eta=v['eta'][mh], npix=npix[mh],
                                        phi=v['phi'][mh], q=v['q'][mh], z0=v['z0'][mh],
                                        side=np.array([str(t) for t in which[mh]]))
                            if pop == 'W1':
                                cols['C'] = M[mh]
                            else:
                                with np.errstate(divide='ignore'):
                                    cols['lsig'] = np.log10(H.sigmas(M[mh]))
                            res[fam][h][pop][b].add(cols)

    # ---------------------------------------------------------- V fits
    print('\n[fit] route B hit errors')
    fit = {}
    test = {}
    for fam in FAMILIES:
        fit[fam] = []
        test[fam] = []
        for b in range(neta):
            fit_h = 'even' if a.split_test else 'all'
            src = res[fam][fit_h]['W1'][b].get()
            dst = res[fam][fit_h]['D0'][b].get()
            if len(src.get('eta', [])) < 200 or len(dst.get('eta', [])) < 200:
                fit[fam].append(None)
                test[fam].append(None)
                print('    %s |eta| %.1f-%.1f: too few tracks (W1 %d, D0 %d), no fit'
                      % (fam, a.eta_edges[b], a.eta_edges[b + 1],
                         len(src.get('eta', [])), len(dst.get('eta', []))))
                continue
            add_jacobian(src, fam)
            r = fit_bin(src, dst, rng)
            fit[fam].append(r)
            line = ('    %s |eta| %.1f-%.1f: sigma_u %6.1f um  sigma_v %6.1f um  '
                    'median log10 sigma dxy %.3f->%.3f (data %.3f)  dsz %.3f (data %.3f)'
                    % (fam, a.eta_edges[b], a.eta_edges[b + 1], 1e4 * r['sigma_u_cm'],
                       1e4 * r['sigma_v_cm'], wquant(np.log10(H.sigmas(src['C'])[:, 3]),
                                                     np.ones(len(src['eta'])), (0.5,))[0],
                       r['pred_dxy'][1], r['target_dxy'][1], r['pred_dsz'][1],
                       r['target_dsz'][1]))
            print(line)
            if a.split_test:
                s2, d2 = res[fam]['odd']['W1'][b].get(), res[fam]['odd']['D0'][b].get()
                if len(s2.get('eta', [])) >= 200 and len(d2.get('eta', [])) >= 200:
                    add_jacobian(s2, fam)
                    w, keep, _ = fine_match(s2, d2)
                    ld, lz, ok = predict(s2, r['sigma_u_cm'], r['sigma_v_cm'])
                    one = keep.astype(float)
                    test[fam].append(dict(
                        pred_dxy=wquant(ld, w * ok, (0.16, 0.5, 0.84)),
                        pred_dsz=wquant(lz, w * ok, (0.16, 0.5, 0.84)),
                        target_dxy=wquant(d2['lsig'][:, 3], one, (0.16, 0.5, 0.84)),
                        target_dsz=wquant(d2['lsig'][:, 4], one, (0.16, 0.5, 0.84))))
                else:
                    test[fam].append(None)
            else:
                test[fam].append(None)

    he = dict(epoch=a.epoch, sample=a.sample, killmaps=os.path.abspath(a.killmaps),
              eps_working=a.eps_working, eps_dead=a.eps_dead,
              split_test=a.split_test, surfaces={})
    for fam in FAMILIES:
        he['surfaces'][fam] = dict(
            abs_eta_edges=a.eta_edges,
            sigma_u_cm=[r['sigma_u_cm'] if r else None for r in fit[fam]],
            sigma_v_cm=[r['sigma_v_cm'] if r else None for r in fit[fam]],
            fit=fit[fam], split_test=test[fam])
    he_path = os.path.join(out, 'hiterrors_%s_%s.json' % (a.sample, a.epoch))
    with open(he_path, 'w') as fh:
        json.dump(he, fh, indent=1, default=lambda o: None if o is None else
                  (o.item() if hasattr(o, 'item') else str(o)))

    # ----------------------------------------------------- mixture verdict
    lines = summary_lines(a, counts, hist, fit, test)
    print('\n' + '\n'.join(lines))
    plot(a, hist, fit, test, res, lines, stem)
    print('\n[out] %s.pdf, %s   (%.1f min)' % (stem, he_path, (time.time() - t0) / 60))


def add_jacobian(src, fam):
    par = H.helix_to_curv(src['pt'], src['eta'], src['phi'], src['q'], src['z0'])
    Hm = np.full((len(par), 2, 5), np.nan)
    for s in FAMILIES[fam]:
        m = src['side'] == s
        if m.any():
            Hm[m] = H.jacobian(s, par[m])
    src['H'] = Hm
    ok = np.all(np.isfinite(Hm), axis=(1, 2))
    if not ok.all():
        for k in list(src):
            src[k] = src[k][ok]


def mixture_table(hist, fam, eta_bin=None):
    rows = []
    W0, D0 = hist[fam]['W0'], hist[fam]['D0']
    for p in PARAMS:
        e = H.LOGSIG_EDGES[p]
        h0 = W0.projected(p, target=D0, eta_bin=eta_bin)
        hd = D0.projected(p, eta_bin=eta_bin)
        q0, qd = H.quantiles(h0, e), H.quantiles(hd, e)
        rows.append((p, q0, qd, H.ks_distance(h0, hd)))
    return rows


def summary_lines(a, counts, hist, fit, test):
    L = ['No-L1 study, epoch %s, sample %s' % (a.epoch, a.sample), '',
         'W1 hit present, working cell (eps >= %.2f)   W0 hit missing, working cell'
         % a.eps_working,
         'D0 hit missing, dead cell (eps <= %.2f)       "missing" = next layer/disk is the first'
         '\neps = HIT EFFICIENCY of the cell in data (probes with a valid hit / probes)'
         % a.eps_dead, '']
    for fam in FAMILIES:
        c = counts[fam]
        L.append('%s tracks: W1 %.0f   W0 %.0f   D0 %.0f   (middle cells: %.0f with hit, %.0f without)'
                 % (fam, c['W1'], c['W0'], c['D0'], c['mid1'], c['mid0']))
    L += ['', 'MIXTURE TEST: log10 sigma, W0 reweighted to D0 in (pt, |eta|, other pixel hits)',
          'shift = median(W0) - median(D0) in log10 (0.01 = 2.3%); width = (q84-q16) ratio W0/D0',
          '%-4s %-7s %9s %9s %7s' % ('surf', 'param', 'shift', 'width', 'KS')]
    for fam in FAMILIES:
        if hist[fam]['W0'].n <= 0 or hist[fam]['D0'].n <= 0:
            L.append('%-4s (no W0 or no D0 tracks)' % fam)
            continue
        for p, q0, qd, ks in mixture_table(hist, fam):
            wr = (q0[2] - q0[0]) / (qd[2] - qd[0]) if qd[2] > qd[0] else np.nan
            L.append('%-4s %-7s %+9.4f %9.3f %7.3f' % (fam, p, q0[1] - qd[1], wr, ks))
    L += ['', 'ROUTE B HIT ERRORS (median log10 sigma of D0 matched; q16/q84 are a check)',
          '%-4s %-9s %8s %8s | %-27s | %-27s' % ('surf', '|eta|', 'su[um]', 'sv[um]',
                                                'dxy q16/q50/q84 pred-data',
                                                'dsz q16/q50/q84 pred-data')]
    for fam in FAMILIES:
        for b, r in enumerate(fit[fam]):
            lab = '%.1f-%.1f' % (a.eta_edges[b], a.eta_edges[b + 1])
            if r is None:
                L.append('%-4s %-9s   no fit' % (fam, lab))
                continue
            dd = [p - t for p, t in zip(r['pred_dxy'], r['target_dxy'])]
            dz = [p - t for p, t in zip(r['pred_dsz'], r['target_dsz'])]
            L.append('%-4s %-9s %8.1f %8.1f | %+.3f/%+.3f/%+.3f      | %+.3f/%+.3f/%+.3f%s'
                     % (fam, lab, 1e4 * r['sigma_u_cm'], 1e4 * r['sigma_v_cm'], *dd, *dz,
                        ('   UNCONSTRAINED: ' + ', '.join(r['unconstrained']))
                        if r['unconstrained'] else ''))
            t = test[fam][b]
            if t:
                dd = [p - q for p, q in zip(t['pred_dxy'], t['target_dxy'])]
                dz = [p - q for p, q in zip(t['pred_dsz'], t['target_dsz'])]
                L.append('%-4s %-9s %17s | %+.3f/%+.3f/%+.3f      | %+.3f/%+.3f/%+.3f'
                         % ('', '  odd cells', '(not fitted)', *dd, *dz))
    return L


def plot(a, hist, fit, test, res, lines, stem):
    plt = P.setup_mpl()
    book = P.Book(stem + '.pdf', png_prefix=stem if a.png else None)
    P.page_text(book, lines, 'summary')
    cols = {'W1': ('#7570b3', '-'), 'W0': ('#d95f02', '-'), 'D0': ('k', '-')}
    for fam in FAMILIES:
        if hist[fam]['D0'].n <= 0:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, p in zip(axes.flat, PARAMS):
            e = H.LOGSIG_EDGES[p]
            xc = 0.5 * (e[1:] + e[:-1])
            for pop in ('W1', 'W0', 'D0'):
                if hist[fam][pop].n <= 0:
                    continue
                h = hist[fam][pop].projected(p, target=None if pop == 'D0' else hist[fam]['D0'])
                c, ls = cols[pop]
                ax.step(xc, h, where='mid', color=c, ls=ls,
                        label={'W1': 'W1 with hit', 'W0': 'W0 no hit, working cell',
                               'D0': 'D0 no hit, dead cell'}[pop])
            ax.set_xlabel('log10 sigma(%s)' % p)
            ax.set_ylabel('fraction')
        axes.flat[0].legend(fontsize=9)
        axes.flat[-1].axis('off')
        axes.flat[-1].text(0, 0.9, 'W1 and W0 reweighted to D0\nin (pt, |eta|, other pixel hits)\n\n'
                           'W0 = D0: one no-hit population\nW0 != D0: a mixture', va='top')
        fig.suptitle('%s: tracks with and without the hit, epoch %s, %s'
                     % (fam, a.epoch, a.sample), fontsize=13)
        fig.tight_layout()
        book.add(fig, '%s_mixture' % fam)

    # fitted V vs |eta|
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ec = 0.5 * (np.array(a.eta_edges[1:]) + np.array(a.eta_edges[:-1]))
    for ax, key, lab in ((axes[0], 'sigma_u_cm', 'sigma_u (r*phi) [um]'),
                         (axes[1], 'sigma_v_cm', 'sigma_v (z on L1, r on D1) [um]')):
        for fam, c in (('L1', 'k'), ('D1', '#d95f02')):
            y = np.array([1e4 * r[key] if r else np.nan for r in fit[fam]])
            ax.plot(ec, y, 'o-', color=c, label=fam)
        ax.set_xlabel('|eta|')
        ax.set_ylabel(lab)
        ax.set_yscale('log')
        ax.legend()
    fig.suptitle('Route B effective hit errors, epoch %s, %s' % (a.epoch, a.sample))
    fig.tight_layout()
    book.add(fig, 'hiterrors')
    book.close()


if __name__ == '__main__':
    main()
