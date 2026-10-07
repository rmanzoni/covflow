#!/usr/bin/env python3
"""
pixel_eff_maps.py
=================

Hit-efficiency maps of the pixel detector, data vs MC, for one data-taking
epoch of configs/run3_epochs.py (the PPD-based MC <-> data map used for the
covflow trainings). Default: BPix layer 1 and FPix disk 1. `--surfaces all`
adds L2-L4 and D2-D3 (needs a per-layer hit mask, see below).

WHAT IS MEASURED
----------------
Probe   = a muon track (mu1 and mu2 of each selected event) whose helix crosses
          the surface S, and which has at least --min-other-hits valid pixel
          hits OTHER than a hit on S. The second condition makes the probe a
          track that exists independently of S.
Cell    = where the probe crosses S:
            barrel layer : (z, phi) on the cylinder r = r_L
            endcap disk  : (r, phi) on the plane |z| = z_D, one map per side
          computed track by track from the exact helix (pt, eta, phi, charge,
          z0 of the track; B = 3.8 T). This is why z0 enters: two tracks with
          the same eta and phi cross different modules if their z0 differ.
eps(cell) = (probes in the cell WITH a valid hit on S) / (probes in the cell)
          MC probes carry the pileup weight of the epoch; data probes weight 1.

Hit types are NOT needed: a dead ROC and an inefficient one both count as
"no valid hit". For L1 and D1 the valid hit is read from the first-hit
branches:  valid L1 hit  <=>  pix_first_b_layer == 1
           valid D1 hit  <=>  pix_first_e_disk  == 1   (on the track's side)
For L2-L4 / D2-D3 the first-hit branches are not enough (a track with an L1 hit
says nothing about L2), so --surfaces all requires a per-layer bit mask branch
({mu}_pix_valid_mask: bits 0-3 = BPix L1-L4, bits 4-6 = FPix D1-D3). That
branch does not exist yet in the ntuples; the script stops with this message
if it is asked for and missing.

DERIVED QUANTITIES (all per cell)
---------------------------------
  delta     = eps_data - eps_MC
  P_kill    = 1 - eps_data / eps_MC   when eps_data < eps_MC: the probability
              with which an MC hit must be removed to reproduce data;
              cells with eps_data > eps_MC cannot be fixed by killing and are
              listed separately (reweighting fallback).
  signif    = delta / sigma(delta)
  eps_MC on data illumination
            = sum_c eps_MC(c) N_data(c) / sum_c N_data(c): the MC efficiency
              integrated with the DATA distribution of crossing points, so that
              the integrated data/MC comparison is not biased by different
              z0 / eta / pt spectra.

OUTPUTS (in --out)
------------------
  effmaps_<epoch>.pdf        all plots, one surface after the other
  effmaps_<epoch>_*.png      the same pages as PNG, with --png
  effmaps_<epoch>.npz        edges + sum(w), sum(w^2) of numerator and
                             denominator, per surface and sample: the input
                             of the hit-killing emulation
  effmaps_<epoch>.json       integrated numbers, coverage, run table
  bad_cells_<epoch>.csv      cells with eps_data << eps_MC (dead in data)

USAGE (tcsh, on the UI, covflow env)
------------------------------------
  python pixel_eff_maps.py --epoch 2026 --list-branches
  python pixel_eff_maps.py --epoch 2026 --max-events 200000 --out test_2026
  python pixel_eff_maps.py --epoch 2026 |& tee effmaps_2026.log
  python pixel_eff_maps.py --epoch 2024 --surfaces all       (needs the mask)

GEOMETRY CONSTANTS are approximate Phase-1 values (see GEOM below). They set
where the helix is intersected and the acceptance used for integrated numbers.
The maps themselves extend beyond the nominal acceptance, so wrong edges show
up as a band of empty or inefficient cells at the border.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import runpy
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CFG = os.path.join(HERE, 'configs', 'run3_epochs.py')

# ---------------------------------------------------------------------------
# geometry (cm, T). Approximate Phase-1 values: CHECK against the CMSSW geometry
# ---------------------------------------------------------------------------
GEOM = dict(
    B_FIELD=3.8,
    BPIX_R={1: 2.9, 2: 6.8, 3: 10.9, 4: 16.0},      # nominal layer radii
    BPIX_HALF_Z=26.6,                              # |z| acceptance of the barrel
    FPIX_Z={1: 29.1, 2: 39.6, 3: 51.6},             # |z| of the disks
    FPIX_R=(4.5, 14.8),                            # radial acceptance of a disk
    ROC_PITCH_Z=0.8325,                            # 66.6 mm module / 8 ROCs
)

# branch templates, {mu} -> mu1 / mu2. Override with --branch KEY=TEMPLATE;
# an empty template disables an optional branch.
BRANCHES = dict(
    pt='{mu}_best_trk_pt',          # covflow context
    eta='{mu}_best_trk_eta',        # covflow context
    phi='{mu}_phi',                 # used by phi_map.py
    charge='{mu}_charge',           # ASSUMED name -- optional (straight line if empty)
    vz='{mu}_vz',                   # ASSUMED name -- track reference point z (z0)
    vx='{mu}_vx',                   # ASSUMED name -- optional (0 if empty)
    vy='{mu}_vy',                   # ASSUMED name -- optional (0 if empty)
    dz='{mu}_dz',                   # only with --z0-from dz_pv
    pv_z='pv_z',                    # only with --z0-from dz_pv (ASSUMED name)
    n_pix='{mu}_n_pix_hit',         # covflow context
    first_b='{mu}_pix_first_b_layer',
    first_e='{mu}_pix_first_e_disk',
    mask='{mu}_pix_valid_mask',     # does NOT exist yet; only for L2-4 / D2-3
    run='run',
)
OPTIONAL = {'charge', 'vx', 'vy'}

MASK_BIT = {'L1': 0, 'L2': 1, 'L3': 2, 'L4': 3, 'D1': 4, 'D2': 5, 'D3': 6}
SELECTION_FUNCS = {'abs': np.abs, 'sqrt': np.sqrt, 'log': np.log, 'exp': np.exp,
                   'cos': np.cos, 'sin': np.sin, 'minimum': np.minimum,
                   'maximum': np.maximum, 'where': np.where, 'np': np}


def die(msg):
    sys.stderr.write('\nERROR: ' + msg.rstrip() + '\n')
    sys.exit(1)


def warn(msg):
    sys.stderr.write('[warn] ' + msg.rstrip() + '\n')


class Progress:
    """Status line on stderr: done/total, rate, elapsed, ETA. Redrawn in place
    on a terminal, one line every ~10% in a log file."""

    def __init__(self, total, label, unit='ev'):
        self.total, self.label, self.unit = max(int(total), 1), label, unit
        self.t0 = time.time()
        self.tty = sys.stderr.isatty()
        self.last_pct = -10
        self.update(0)

    def update(self, done):
        frac = min(done / self.total, 1.0)
        el = time.time() - self.t0
        rate = done / el if el > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else float('nan')
        bar = '#' * int(30 * frac) + '.' * (30 - int(30 * frac))
        line = ('[%s] %5.1f%% %-22s %s/%s %s  %.0f %s/s  %ds elapsed  eta %s'
                % (bar, 100 * frac, self.label[:22], f'{done:,}', f'{self.total:,}',
                   self.unit, rate, self.unit, el,
                   '%ds' % eta if np.isfinite(eta) else '?'))
        if self.tty:
            sys.stderr.write('\r' + line)
            if done >= self.total:
                sys.stderr.write('\n')
        elif int(100 * frac) >= self.last_pct + 10 or done >= self.total:
            self.last_pct = int(100 * frac)
            sys.stderr.write(line + '\n')
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# arguments and configuration
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--epoch', required=True,
                   help='epoch of the config (2022_preEE ... 2026)')
    p.add_argument('--config', default=DEFAULT_CFG,
                   help='covflow epoch config (default: %(default)s)')
    p.add_argument('--surfaces', nargs='+', default=['L1', 'D1'],
                   help="L1..L4, D1..D3, or 'all' (default: L1 D1)")
    p.add_argument('--muons', nargs='+', default=['mu1', 'mu2'])
    p.add_argument('--branch', action='append', default=[], metavar='KEY=TEMPLATE',
                   help='override a branch template, e.g. charge={mu}_q or vx=')
    p.add_argument('--z0-from', choices=('vz', 'dz_pv'), default='vz',
                   help='z0 = {mu}_vz, or pv_z + {mu}_dz (default: vz)')
    p.add_argument('--min-other-hits', type=int, default=2,
                   help='valid pixel hits required besides the one on S '
                        '(default: %(default)s)')
    p.add_argument('--nbins-phi', type=int, default=96)
    p.add_argument('--nbins-r', type=int, default=30)
    p.add_argument('--min-den', type=float, default=20.,
                   help='minimum effective probes for a cell to be drawn '
                        '(default: %(default)s)')
    p.add_argument('--min-run-den', type=float, default=2000.,
                   help='probes per point of the run-dependence plot; '
                        'consecutive runs are merged until reached')
    p.add_argument('--bad-eps-data', type=float, default=0.5,
                   help='bad-cell list: eps_data below this ...')
    p.add_argument('--bad-eps-mc', type=float, default=0.9,
                   help='... while eps_MC above this')
    p.add_argument('--max-events', type=int, default=None,
                   help='stop each sample after this many entries (tests)')
    p.add_argument('--step-size', default='200 MB')
    p.add_argument('--xrootd', action='store_true',
                   help='read root://SE_HOST/ instead of the /pnfs mount')
    p.add_argument('--out', default=None, help='default: effmaps_<epoch>')
    p.add_argument('--png', action='store_true', help='also one PNG per page')
    p.add_argument('--list-branches', nargs='?', const='', default=None,
                   metavar='REGEX',
                   help='print the branches of the first data file matching '
                        'REGEX (default: candidates for the assumed names) '
                        'and exit')
    return p.parse_args()


def load_epoch(a):
    if not os.path.isfile(a.config):
        die('config not found: %s (use --config)' % a.config)
    cfg = runpy.run_path(a.config)
    for key in ('EPOCHS', 'INPUT_DIR', 'TRAINING'):
        if key not in cfg:
            die('%s does not define %s' % (a.config, key))
    if a.epoch not in cfg['EPOCHS']:
        die('unknown epoch %r; known: %s' % (a.epoch, list(cfg['EPOCHS'])))
    ep, T = cfg['EPOCHS'][a.epoch], cfg['TRAINING']
    base = cfg['INPUT_DIR']
    if a.xrootd:
        base = 'root://%s/%s' % (cfg.get('SE_HOST', 't3dcachedb03.psi.ch:1094'), base)
    fmt = dict(pu_year=ep['pu_year'], mu='{mu}')
    return dict(
        data=[os.path.join(base, f) for f in ep['data']],
        mc=[os.path.join(base, f) for f in ep['mc']],
        pu_year=ep['pu_year'],
        tree=T.get('tree', 'tree'),
        selection=T.get('selection') or '',
        data_selection=T.get('data_selection') or '',
        mc_selection=T.get('mc_selection') or '',
        mc_weight=(T.get('mc_weight') or '').format(**fmt),
    )


def resolve_surfaces(names):
    if names == ['all']:
        return ['L1', 'L2', 'L3', 'L4', 'D1', 'D2', 'D3']
    for s in names:
        if s not in MASK_BIT:
            die('unknown surface %r: use L1..L4, D1..D3 or all' % s)
    return list(dict.fromkeys(names))


def branch_templates(a, surfaces):
    br = dict(BRANCHES)
    for item in a.branch:
        key, sep, val = item.partition('=')
        if not sep or key not in br:
            die('--branch %r: expected KEY=TEMPLATE with KEY in %s' % (item, sorted(br)))
        br[key] = val
    needs_mask = any(s not in ('L1', 'D1') for s in surfaces)
    used = ['pt', 'eta', 'phi', 'charge', 'vx', 'vy', 'n_pix', 'first_b', 'first_e']
    used += ['vz'] if a.z0_from == 'vz' else ['dz', 'pv_z']
    if needs_mask:
        used.append('mask')
    tmpl = {k: br[k] for k in used if br[k]}
    for k in used:
        if not br[k] and k not in OPTIONAL:
            die('branch %r is required and cannot be disabled' % k)
    return tmpl, br['run'], needs_mask


def selection_branches(expr):
    """Identifiers of a selection string that are not functions."""
    if not expr:
        return set()
    names = set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()', expr))
    return {n for n in names if n not in SELECTION_FUNCS and n not in ('and', 'or', 'not')}


def eval_selection(expr, arrays, n):
    if not expr:
        return np.ones(n, bool)
    ns = dict(SELECTION_FUNCS)
    ns.update(arrays)
    try:
        out = eval(expr, {'__builtins__': {}}, ns)
    except Exception as e:
        die('cannot evaluate selection %r: %s' % (expr, e))
    return np.broadcast_to(np.asarray(out, bool), (n,)).copy()


# ---------------------------------------------------------------------------
# helix crossing with a barrel cylinder or an endcap plane
# ---------------------------------------------------------------------------

def helix_frame(pt, phi0, q, x0, y0, bfield):
    """Bending radius R (cm, inf if q == 0) and circle centre (cx, cy).
    With B along +z a positive track turns clockwise: the position azimuth
    decreases, phi(s) = phi0 - q s / (2R) for a track from the origin."""
    with np.errstate(divide='ignore'):
        R = np.where(q != 0, pt / (0.299792458 * bfield) * 100.0, np.inf)
    s = np.sign(q)
    cx = np.where(np.isfinite(R), x0 + s * R * np.sin(phi0), np.nan)
    cy = np.where(np.isfinite(R), y0 - s * R * np.cos(phi0), np.nan)
    return R, cx, cy


def helix_xy(alpha, q, x0, y0, cx, cy):
    """Transverse position after turning angle alpha = s / R."""
    a = -np.sign(q) * alpha
    dx, dy = x0 - cx, y0 - cy
    ca, sa = np.cos(a), np.sin(a)
    return cx + dx * ca - dy * sa, cy + dx * sa + dy * ca


def cross_barrel(r_l, pt, eta, phi0, q, x0, y0, z0, bfield):
    """(z, phi) where the track first reaches radius r_l; NaN if it never does."""
    R, cx, cy = helix_frame(pt, phi0, q, x0, y0, bfield)
    cot = np.sinh(eta)
    z = np.full(pt.shape, np.nan)
    ph = np.full(pt.shape, np.nan)

    straight = ~np.isfinite(R)
    if straight.any():
        ux, uy = np.cos(phi0[straight]), np.sin(phi0[straight])
        rx, ry = x0[straight], y0[straight]
        b = rx * ux + ry * uy
        disc = b * b - (rx * rx + ry * ry) + r_l * r_l
        s = -b + np.sqrt(np.maximum(disc, 0.0))
        z[straight] = z0[straight] + s * cot[straight]
        ph[straight] = np.arctan2(ry + s * uy, rx + s * ux)

    c = ~straight
    if c.any():
        Rc, qc = R[c], q[c]
        x0c, y0c, cxc, cyc = x0[c], y0[c], cx[c], cy[c]
        ref = np.hypot(x0c, y0c)
        hi = 2.0 * np.arcsin(np.clip((r_l + ref) / (2.0 * Rc), 0.0, 1.0)) + 1e-3
        hi = np.minimum(hi, np.pi)
        xh, yh = helix_xy(hi, qc, x0c, y0c, cxc, cyc)
        reach = np.hypot(xh, yh) >= r_l
        lo = np.zeros_like(hi)
        for _ in range(45):                      # bisection, |pos| grows with alpha
            mid = 0.5 * (lo + hi)
            xm, ym = helix_xy(mid, qc, x0c, y0c, cxc, cyc)
            out = np.hypot(xm, ym) >= r_l
            hi = np.where(out, mid, hi)
            lo = np.where(out, lo, mid)
        xa, ya = helix_xy(hi, qc, x0c, y0c, cxc, cyc)
        zc = z0[c] + Rc * hi * cot[c]
        z[c] = np.where(reach, zc, np.nan)
        ph[c] = np.where(reach, np.arctan2(ya, xa), np.nan)
    return z, ph


def cross_disk(z_d, pt, eta, phi0, q, x0, y0, z0, bfield):
    """(r, phi) where the track crosses the plane z = z_d (signed); NaN if the
    track goes the other way or curls up before reaching it."""
    R, cx, cy = helix_frame(pt, phi0, q, x0, y0, bfield)
    cot = np.sinh(eta)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (z_d - z0) / cot                     # transverse path length
    ok = np.isfinite(s) & (s > 0)
    r = np.full(pt.shape, np.nan)
    ph = np.full(pt.shape, np.nan)

    straight = ok & ~np.isfinite(R)
    if straight.any():
        xs = x0[straight] + s[straight] * np.cos(phi0[straight])
        ys = y0[straight] + s[straight] * np.sin(phi0[straight])
        r[straight], ph[straight] = np.hypot(xs, ys), np.arctan2(ys, xs)

    c = ok & np.isfinite(R)
    if c.any():
        alpha = s[c] / R[c]
        good = alpha < np.pi                     # no full half-turn
        xc, yc = helix_xy(alpha, q[c], x0[c], y0[c], cx[c], cy[c])
        r[c] = np.where(good, np.hypot(xc, yc), np.nan)
        ph[c] = np.where(good, np.arctan2(yc, xc), np.nan)
    return r, ph


# ---------------------------------------------------------------------------
# surfaces and accumulators
# ---------------------------------------------------------------------------

class Surface:
    """One map: a barrel layer, or one side of a disk."""

    def __init__(self, name, a):
        self.name = name                         # 'L1', 'D1+', 'D1-'
        self.kind = name[0]
        self.index = int(name[1])
        self.base = name[:2]
        self.side = 0 if self.kind == 'L' else (1 if name.endswith('+') else -1)
        pi = np.pi
        self.ye = np.linspace(-pi, pi, a.nbins_phi + 1)
        if self.kind == 'L':
            k = int(math.ceil((GEOM['BPIX_HALF_Z'] + 6.0) / GEOM['ROC_PITCH_Z']))
            self.xe = np.arange(-k, k + 1) * GEOM['ROC_PITCH_Z']
            self.xlabel = 'z at %s [cm]' % self.base
        else:
            self.xe = np.linspace(GEOM['FPIX_R'][0] - 1.5, GEOM['FPIX_R'][1] + 2.0,
                                  a.nbins_r + 1)
            self.xlabel = 'r on %s [cm]' % name
        self.ylabel = 'phi at %s [rad]' % name

    def crossing(self, v, bfield):
        if self.kind == 'L':
            return cross_barrel(GEOM['BPIX_R'][self.index], v['pt'], v['eta'],
                                v['phi'], v['q'], v['x0'], v['y0'], v['z0'], bfield)
        z_d = self.side * GEOM['FPIX_Z'][self.index]
        return cross_disk(z_d, v['pt'], v['eta'], v['phi'], v['q'], v['x0'],
                          v['y0'], v['z0'], bfield)

    def in_acceptance(self, x):
        if self.kind == 'L':
            return np.abs(x) < GEOM['BPIX_HALF_Z']
        return (x > GEOM['FPIX_R'][0]) & (x < GEOM['FPIX_R'][1])


ETA_EDGES = np.linspace(-2.5, 2.5, 51)
PT_EDGES = np.array([3., 4., 5., 6., 7., 8., 10., 12., 15., 20., 30., 50.])
Z0_EDGES = np.linspace(-25., 25., 101)


class Acc:
    """sum(w) and sum(w^2) for numerator and denominator of one sample."""

    def __init__(self, shape):
        self.den = np.zeros(shape)
        self.num = np.zeros(shape)
        self.den2 = np.zeros(shape)
        self.num2 = np.zeros(shape)


def fill2(acc, x, y, w, hit, xe, ye):
    for arr, ww in ((acc.den, w), (acc.den2, w * w),
                    (acc.num, w * hit), (acc.num2, w * w * hit)):
        h, _, _ = np.histogram2d(x, y, bins=(xe, ye), weights=ww)
        arr += h


def fill1(acc, x, w, hit, e):
    for arr, ww in ((acc.den, w), (acc.den2, w * w),
                    (acc.num, w * hit), (acc.num2, w * w * hit)):
        arr += np.histogram(x, bins=e, weights=ww)[0]


def efficiency(num, den, num2, den2, min_den=0.0):
    """eps and its error for weighted counts; NaN where the effective number of
    probes (den^2 / den2) is below min_den."""
    with np.errstate(divide='ignore', invalid='ignore'):
        eps = num / den
        fail2 = np.maximum(den2 - num2, 0.0)
        var = (num2 * (1 - eps) ** 2 + fail2 * eps ** 2) / den ** 2
        neff = den ** 2 / den2
    bad = ~(den > 0) | ~(neff >= min_den)
    eps = np.where(bad, np.nan, eps)
    err = np.where(bad, np.nan, np.sqrt(np.maximum(var, 0.0)))
    return eps, err, np.where(den2 > 0, neff, 0.0)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def list_branches(files, tree, regex):
    import uproot
    f0 = files[0]
    with uproot.open(f0) as fh:
        keys = sorted(fh[tree].keys())
    rx = re.compile(regex or r'(^|_)(vx|vy|vz|dz|charge|q|phi|pv_|run)|pix|trg|hlt|HLT|pu_weight')
    hits = [k for k in keys if rx.search(k)]
    print('%s: %d branches, %d matching %r' % (f0, len(keys), len(hits), rx.pattern))
    for k in hits:
        print('   ', k)


def preflight(files_by_sample, tree, needed_by_sample):
    """Open every file and check every branch before reading anything."""
    import uproot
    allfiles = [(s, f) for s, fs in files_by_sample.items() for f in fs]
    print('\n[check] %d files' % len(allfiles))
    prog = Progress(len(allfiles), 'branch check', unit='files')
    entries, problems = {}, []
    for i, (sample, f) in enumerate(allfiles, 1):
        try:
            with uproot.open(f) as fh:
                if tree not in fh:
                    problems.append('%s: no tree %r' % (f, tree))
                    continue
                t = fh[tree]
                have = set(t.keys())
                entries[f] = t.num_entries
        except Exception as e:
            problems.append('%s: cannot open (%s)' % (f, e))
            continue
        finally:
            prog.update(i)
        miss = sorted(needed_by_sample[sample] - have)
        if miss:
            problems.append('%s: missing %s' % (os.path.basename(f), ', '.join(miss)))
    if problems and any('_pix_valid_mask' in p_ for p_ in problems):
        problems.append('NOTE: the per-layer hit mask is needed for L2-L4 and D2-D3 and '
                        'is not produced by the current ntuplizer (TrackHitContent has '
                        'only the FIRST valid layer/disk). Add it there, or run on L1 '
                        'and D1 only.')
    if problems:
        die('branch check failed:\n  ' + '\n  '.join(problems) +
            '\n\nFind the right names with --list-branches, then pass them with '
            '--branch KEY=TEMPLATE (e.g. --branch "charge={mu}_q").')
    return entries


def muon_view(arr, mu, tmpl, z0_from, sel):
    """Per-muon arrays for the selected events, as float64."""
    g = lambda key: np.asarray(arr[tmpl[key].format(mu=mu)], dtype=np.float64)[sel]
    n = int(sel.sum())
    v = dict(pt=g('pt'), eta=g('eta'), phi=g('phi'),
             n_pix=g('n_pix'), first_b=g('first_b'), first_e=g('first_e'))
    v['q'] = np.sign(g('charge')) if 'charge' in tmpl else np.zeros(n)
    v['x0'] = g('vx') if 'vx' in tmpl else np.zeros(n)
    v['y0'] = g('vy') if 'vy' in tmpl else np.zeros(n)
    if z0_from == 'vz':
        v['z0'] = g('vz')
    else:
        v['z0'] = g('dz') + np.asarray(arr[tmpl['pv_z']], np.float64)[sel]
    if 'mask' in tmpl:
        m = g('mask')
        v['mask'] = np.where(np.isfinite(m), m, 0).astype(np.int64)
    return v


def has_hit(v, surf, use_mask):
    if use_mask:
        return ((v['mask'] >> MASK_BIT[surf.base]) & 1).astype(np.float64)
    if surf.base == 'L1':
        return (v['first_b'] == 1).astype(np.float64)
    if surf.base == 'D1':
        return (v['first_e'] == 1).astype(np.float64)
    raise RuntimeError('surface %s needs the hit mask' % surf.name)


def other_hits(v, hit, use_mask):
    if use_mask:
        m = v['mask']
        nlay = np.zeros(m.shape, np.int64)
        for b in range(7):
            nlay += (m >> b) & 1
        return nlay - hit
    return v['n_pix'] - hit


def process_sample(sample, files, info, tmpl, run_branch, a, surfaces, use_mask,
                   entries, is_mc):
    import uproot
    sel_expr = ' & '.join('(%s)' % s for s in
                          [info['selection'],
                           info['mc_selection'] if is_mc else info['data_selection']]
                          if s)
    branches = set(selection_branches(sel_expr))
    for mu in a.muons:
        branches |= {t.format(mu=mu) for k, t in tmpl.items() if '{mu}' in t}
    branches |= {t for t in tmpl.values() if '{mu}' not in t}
    weight = info['mc_weight'] if is_mc else ''
    if weight:
        branches.add(weight)
    if not is_mc:
        branches.add(run_branch)

    shapes = {s.name: (len(s.xe) - 1, len(s.ye) - 1) for s in surfaces}
    res = dict(
        maps={s.name: Acc(shapes[s.name]) for s in surfaces},
        eta={s.name: Acc(len(ETA_EDGES) - 1) for s in surfaces},
        pt={s.name: Acc(len(PT_EDGES) - 1) for s in surfaces},
        z0=np.zeros(len(Z0_EDGES) - 1),
        runs={s.base: {} for s in surfaces},
        n_events=0, n_selected=0, n_probes={s.name: 0.0 for s in surfaces},
        w_dropped=0,
    )

    total = sum(entries[f] for f in files)
    if a.max_events:
        total = min(total, a.max_events)
    prog = Progress(total, sample)
    done = 0
    for chunk in uproot.iterate([{f: info['tree']} for f in files],
                                expressions=sorted(branches), library='np',
                                step_size=a.step_size):
        n = len(next(iter(chunk.values())))
        if a.max_events and done + n > a.max_events:
            n = a.max_events - done
            chunk = {k: v[:n] for k, v in chunk.items()}
        sel = eval_selection(sel_expr, chunk, n)
        if weight:
            w_ev = np.asarray(chunk[weight], np.float64)
            fin = np.isfinite(w_ev)
            res['w_dropped'] += int((sel & ~fin).sum())
            sel &= fin
            w_ev = w_ev[sel]
        else:
            w_ev = np.ones(int(sel.sum()))
        run_ev = None if is_mc else np.asarray(chunk[run_branch])[sel].astype(np.int64)
        res['n_events'] += n
        res['n_selected'] += int(sel.sum())

        for mu in a.muons:
            v = muon_view(chunk, mu, tmpl, a.z0_from, sel)
            good = (np.isfinite(v['pt']) & (v['pt'] > 0) & np.isfinite(v['eta'])
                    & np.isfinite(v['phi']) & np.isfinite(v['z0']))
            res['z0'] += np.histogram(v['z0'][good], bins=Z0_EDGES, weights=w_ev[good])[0]
            for s in surfaces:
                x, y = s.crossing(v, GEOM['B_FIELD'])
                hit = has_hit(v, s, use_mask)
                probe = (good & np.isfinite(x) & np.isfinite(y)
                         & (other_hits(v, hit, use_mask) >= a.min_other_hits))
                if not probe.any():
                    continue
                xp, yp, wp, hp = x[probe], y[probe], w_ev[probe], hit[probe]
                fill2(res['maps'][s.name], xp, yp, wp, hp, s.xe, s.ye)
                acc = s.in_acceptance(xp)
                res['n_probes'][s.name] += float(wp[acc].sum())
                fill1(res['eta'][s.name], v['eta'][probe][acc], wp[acc], hp[acc], ETA_EDGES)
                fill1(res['pt'][s.name], v['pt'][probe][acc], wp[acc], hp[acc], PT_EDGES)
                if run_ev is not None:
                    r = run_ev[probe][acc]
                    h = hp[acc]
                    tab = res['runs'][s.base]
                    ur, inv = np.unique(r, return_inverse=True)
                    dn = np.bincount(inv, minlength=len(ur))
                    nm = np.bincount(inv, weights=h, minlength=len(ur))
                    for run, d_, n_ in zip(ur.tolist(), dn.tolist(), nm.tolist()):
                        t = tab.setdefault(run, [0.0, 0.0])
                        t[0] += d_
                        t[1] += n_
        done += n
        prog.update(done)
        if a.max_events and done >= a.max_events:
            break
    if is_mc and res['w_dropped']:
        frac = res['w_dropped'] / max(res['n_selected'] + res['w_dropped'], 1)
        print('    %s: %d selected events dropped for a non-finite %s (%.3f%%)'
              % (sample, res['w_dropped'], weight, 100 * frac))
        if res['n_selected'] == 0:
            die('%s: every selected event has a non-finite %s' % (sample, weight))
    return res


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        import mplhep as hep
        plt.style.use(hep.style.CMS)
        plt.rcParams.update({'font.size': 12, 'axes.titlesize': 12,
                             'axes.labelsize': 12, 'legend.fontsize': 10,
                             'xtick.labelsize': 10, 'ytick.labelsize': 10})
    except ImportError:
        plt.rcParams.update({'font.size': 11, 'axes.titlesize': 11})
    return plt


class Book:
    def __init__(self, path, png_prefix=None):
        from matplotlib.backends.backend_pdf import PdfPages
        self.pdf = PdfPages(path)
        self.png_prefix, self.n = png_prefix, 0

    def add(self, fig, tag):
        self.pdf.savefig(fig)
        if self.png_prefix:
            self.n += 1
            fig.savefig('%s_%02d_%s.png' % (self.png_prefix, self.n, tag), dpi=110)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def close(self):
        self.pdf.close()


def draw_map(fig, pos, surf, M, title, cmap, vmin, vmax, clabel):
    """One heatmap: a (z, phi) rectangle for a layer, a polar (r, phi) map for a disk."""
    import matplotlib.pyplot as plt
    Mm = np.ma.masked_invalid(M)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad('0.82')
    if surf.kind == 'L':
        ax = fig.add_subplot(*pos)
        im = ax.pcolormesh(surf.xe, surf.ye, Mm.T, cmap=cm, vmin=vmin, vmax=vmax,
                           shading='flat', rasterized=True)
        for zz in (-GEOM['BPIX_HALF_Z'], GEOM['BPIX_HALF_Z']):
            ax.axvline(zz, color='k', ls='--', lw=0.8)
        ax.set_xlabel(surf.xlabel)
        ax.set_ylabel(surf.ylabel)
    else:
        ax = fig.add_subplot(*pos, projection='polar')
        ax.set_facecolor('0.82')
        im = ax.pcolormesh(surf.ye, surf.xe, Mm, cmap=cm, vmin=vmin, vmax=vmax,
                           shading='flat', rasterized=True)
        th = np.linspace(-np.pi, np.pi, 361)
        for rr in GEOM['FPIX_R']:
            ax.plot(th, np.full_like(th, rr), 'k--', lw=0.8)
        ax.set_ylim(0, surf.xe[-1])
        ax.set_yticks([4, 8, 12, 16])
        ax.set_yticklabels(['4', '8', '12', '16 cm'], fontsize=8)
        th_ticks = np.linspace(-np.pi, np.pi, 8, endpoint=False)
        ax.set_xticks(th_ticks)
        ax.set_xticklabels(['phi=%.2f' % t if i == 0 else '%.2f' % t
                            for i, t in enumerate(th_ticks)], fontsize=8)
        # set_xticks widens the view to include the negative ticks, i.e. to
        # [-pi, 2pi], and a polar axis then draws only half of the disk.
        ax.set_thetalim(-np.pi, np.pi)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08 if surf.kind == 'D' else 0.02,
                      extend='both' if cmap in ('RdBu_r', 'PuOr_r') else 'neither')
    cb.set_label(clabel)
    return ax


def surface_maps(s, rd, rm, a):
    """eps, error and effective counts per cell for data and MC."""
    D, M = rd['maps'][s.name], rm['maps'][s.name]
    ed, sd, nd = efficiency(D.num, D.den, D.num2, D.den2, a.min_den)
    em, sm, nm = efficiency(M.num, M.den, M.num2, M.den2, a.min_den)
    return ed, sd, nd, em, sm, nm


def page_main(book, s, ed, em, epoch):
    """The requested plot: eps_data, eps_MC and their difference."""
    import matplotlib.pyplot as plt
    wide = s.kind == 'L'
    fig = plt.figure(figsize=(18, 5.6) if wide else (18, 6.2))
    delta = ed - em
    # Saturates at 0.2: dead cells still stand out (darkest blue), and
    # percent-level structure elsewhere is not washed out by them.
    lim = np.nanpercentile(np.abs(delta), 95) if np.isfinite(delta).any() else 0.1
    lim = float(np.clip(lim, 0.02, 0.2))
    draw_map(fig, (1, 3, 1), s, ed, 'data: eps = probes with a hit / probes', 'viridis', 0, 1, 'hit efficiency, data (eps_data)')
    draw_map(fig, (1, 3, 2), s, em, 'MC (PU-weighted)', 'viridis', 0, 1, 'hit efficiency, MC (eps_MC)')
    draw_map(fig, (1, 3, 3), s, delta, 'eps_data - eps_MC', 'RdBu_r', -lim, lim, 'eps_data - eps_MC')
    fig.suptitle('%s hit efficiency, epoch %s    (grey = fewer than the minimum '
                 'number of probes)' % (s.name, epoch), fontsize=13)
    fig.tight_layout()
    book.add(fig, '%s_eff' % s.name.replace('+', 'p').replace('-', 'm'))


def page_kill(book, s, ed, sd, em, sm, epoch):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18, 5.6) if s.kind == 'L' else (18, 6.2))
    with np.errstate(divide='ignore', invalid='ignore'):
        pk = np.where(ed < em, 1 - ed / em, np.nan)
        above = np.where(ed > em, ed - em, np.nan)
        sig = (ed - em) / np.sqrt(sd ** 2 + sm ** 2)
    draw_map(fig, (1, 3, 1), s, pk, 'P_kill = 1 - eps_data/eps_MC (where eps_data < eps_MC)',
             'magma_r', 0, 1, 'P_kill')
    draw_map(fig, (1, 3, 2), s, above,
             'eps_data - eps_MC where eps_data > eps_MC\n(not fixable by killing)',
             'Blues', 0, max(0.02, float(np.nanmax(above)) if np.isfinite(above).any() else 0.02),
             'eps_data - eps_MC')
    draw_map(fig, (1, 3, 3), s, np.clip(sig, -10, 10),
             'significance (eps_data - eps_MC) / sigma', 'RdBu_r', -10, 10, 'sigma (clipped at 10)')
    fig.suptitle('%s: what the hit-killing emulation would do, epoch %s' % (s.name, epoch),
                 fontsize=13)
    fig.tight_layout()
    book.add(fig, '%s_kill' % s.name.replace('+', 'p').replace('-', 'm'))


def page_illumination(book, s, rd, rm, epoch):
    """Where the probes cross S, data vs MC, each normalised to unit area."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18, 5.6) if s.kind == 'L' else (18, 6.2))
    Dd, Dm = rd['maps'][s.name].den, rm['maps'][s.name].den
    fd = Dd / Dd.sum() if Dd.sum() > 0 else Dd * np.nan
    fm = Dm / Dm.sum() if Dm.sum() > 0 else Dm * np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where((fd > 0) & (fm > 0), fd / fm, np.nan)
    vmax = float(np.nanmax([np.nanmax(fd), np.nanmax(fm)])) if Dd.sum() > 0 else 1
    draw_map(fig, (1, 3, 1), s, np.where(Dd > 0, fd, np.nan), 'data: fraction of probes per cell',
             'cividis', 0, vmax, 'fraction')
    draw_map(fig, (1, 3, 2), s, np.where(Dm > 0, fm, np.nan), 'MC: fraction of probes per cell',
             'cividis', 0, vmax, 'fraction')
    draw_map(fig, (1, 3, 3), s, np.log2(ratio), 'log2(data fraction / MC fraction)',
             'PuOr_r', -2, 2, 'log2 ratio')
    fig.suptitle('%s: where the probes cross (illumination), epoch %s. Different '
                 'patterns come from z0, eta and pt spectra, not from the detector'
                 % (s.name, epoch), fontsize=12)
    fig.tight_layout()
    book.add(fig, '%s_illum' % s.name.replace('+', 'p').replace('-', 'm'))


def eb(ax, x, xerr, eps, err, label, color, marker):
    m = np.isfinite(eps)
    ax.errorbar(x[m], eps[m], xerr=xerr[m] if xerr is not None else None, yerr=err[m],
                fmt=marker, ms=3.5, color=color, label=label, elinewidth=0.8, capsize=0)


def page_projections(book, s, rd, rm, epoch):
    """1D efficiency vs the in-surface coordinates, eta and pt, with data/MC."""
    import matplotlib.pyplot as plt
    D, M = rd['maps'][s.name], rm['maps'][s.name]
    acc_x = s.in_acceptance(0.5 * (s.xe[1:] + s.xe[:-1]))
    cases = [
        ('x', s.xlabel, s.xe, [x.sum(axis=1) for x in (D.num, D.den, D.num2, D.den2)],
         [x.sum(axis=1) for x in (M.num, M.den, M.num2, M.den2)]),
        ('phi', s.ylabel + ' (inside acceptance)', s.ye,
         [x[acc_x].sum(axis=0) for x in (D.num, D.den, D.num2, D.den2)],
         [x[acc_x].sum(axis=0) for x in (M.num, M.den, M.num2, M.den2)]),
        ('eta', 'track eta (inside acceptance)', ETA_EDGES,
         [getattr(rd['eta'][s.name], k) for k in ('num', 'den', 'num2', 'den2')],
         [getattr(rm['eta'][s.name], k) for k in ('num', 'den', 'num2', 'den2')]),
        ('pt', 'track pt [GeV] (inside acceptance)', PT_EDGES,
         [getattr(rd['pt'][s.name], k) for k in ('num', 'den', 'num2', 'den2')],
         [getattr(rm['pt'][s.name], k) for k in ('num', 'den', 'num2', 'den2')]),
    ]
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(5, 2, height_ratios=[3, 1, 0.75, 3, 1], hspace=0.08, wspace=0.22)
    for i, (tag, xl, e, dd, mm) in enumerate(cases):
        r0, c0 = 3 * (i // 2), i % 2
        ax = fig.add_subplot(gs[r0, c0])
        axr = fig.add_subplot(gs[r0 + 1, c0], sharex=ax)
        x = 0.5 * (e[1:] + e[:-1])
        xerr = 0.5 * (e[1:] - e[:-1])
        ed, sd, _ = efficiency(dd[0], dd[1], dd[2], dd[3], 5)
        em, sm, _ = efficiency(mm[0], mm[1], mm[2], mm[3], 5)
        eb(ax, x, xerr, ed, sd, 'data', 'k', 'o')
        eb(ax, x, xerr, em, sm, 'MC', '#d95f02', 's')
        lo = np.nanmin(np.concatenate([ed, em])) if np.isfinite(np.concatenate([ed, em])).any() else 0
        ax.set_ylim(max(0.0, lo - 0.05), 1.01)
        ax.set_ylabel('hit efficiency eps(%s)' % s.name)
        ax.legend(loc='lower left')
        plt.setp(ax.get_xticklabels(), visible=False)
        with np.errstate(divide='ignore', invalid='ignore'):
            rat = ed / em
            rerr = rat * np.sqrt((sd / ed) ** 2 + (sm / em) ** 2)
        eb(axr, x, xerr, rat, rerr, None, 'k', 'o')
        axr.axhline(1, color='0.5', lw=0.8)
        axr.set_ylabel('data/MC')
        axr.set_xlabel(xl)
        if tag == 'pt':
            ax.set_xscale('log')
        if s.kind == 'L' and tag == 'x':
            for zz in (-GEOM['BPIX_HALF_Z'], GEOM['BPIX_HALF_Z']):
                ax.axvline(zz, color='0.5', ls='--', lw=0.8)
        if s.kind == 'D' and tag == 'x':
            for rr in GEOM['FPIX_R']:
                ax.axvline(rr, color='0.5', ls='--', lw=0.8)
    fig.suptitle('%s hit efficiency, projections, epoch %s' % (s.name, epoch), fontsize=13)
    book.add(fig, '%s_proj' % s.name.replace('+', 'p').replace('-', 'm'))


def page_z0(book, rd, rm, epoch):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = 0.5 * (Z0_EDGES[1:] + Z0_EDGES[:-1])
    for res, lab, col, ls in ((rd, 'data', 'k', '-'), (rm, 'MC (PU-weighted)', '#d95f02', '--')):
        h = res['z0']
        if h.sum() > 0:
            ax.step(x, h / h.sum(), where='mid', color=col, ls=ls,
                    label='%s: mean %.2f cm, rms %.2f cm'
                    % (lab, np.average(x, weights=h),
                       np.sqrt(np.average((x - np.average(x, weights=h)) ** 2, weights=h))))
    ax.set_xlabel('z0 of the muon track [cm]')
    ax.set_ylabel('fraction of muons')
    ax.set_title('z0 of the selected muons, epoch %s' % epoch)
    ax.legend()
    fig.tight_layout()
    book.add(fig, 'z0')


def group_runs(tab, min_den):
    runs = sorted(tab)
    groups, cur = [], None
    for r in runs:
        d, n = tab[r]
        if cur is None:
            cur = [r, r, 0.0, 0.0]
        cur[1], cur[2], cur[3] = r, cur[2] + d, cur[3] + n
        if cur[2] >= min_den:
            groups.append(cur)
            cur = None
    if cur is not None:
        if groups and cur[2] < min_den:
            g = groups[-1]
            g[1], g[2], g[3] = cur[1], g[2] + cur[2], g[3] + cur[3]
        else:
            groups.append(cur)
    return groups


def page_runs(book, rd, bases, a, epoch):
    import matplotlib.pyplot as plt
    bases = [b for b in bases if rd['runs'].get(b)]
    if not bases:
        return {}
    fig, axes = plt.subplots(len(bases), 1, figsize=(12, 3.2 * len(bases)), squeeze=False)
    out = {}
    for ax, b in zip(axes[:, 0], bases):
        g = group_runs(rd['runs'][b], a.min_run_den)
        d = np.array([x[2] for x in g])
        n = np.array([x[3] for x in g])
        eps = n / d
        err = np.sqrt(eps * (1 - eps) / d)
        idx = np.arange(len(g))
        ax.errorbar(idx, eps, yerr=err, fmt='o', ms=3, color='k', elinewidth=0.8)
        tot = n.sum() / d.sum()
        ax.axhline(tot, color='#d95f02', lw=1, label='epoch average %.4f' % tot)
        ax.set_ylabel('hit efficiency eps_data(%s)' % b)
        step = max(1, len(g) // 12)
        ax.set_xticks(idx[::step])
        ax.set_xticklabels(['%d' % x[0] for x in g][::step], rotation=45, fontsize=8)
        ax.legend(loc='lower left')
        out[b] = [dict(first_run=x[0], last_run=x[1], probes=x[2], hits=x[3]) for x in g]
    axes[-1, 0].set_xlabel('first run of the group (consecutive runs merged to >= %d '
                           'probes)' % a.min_run_den)
    fig.suptitle('Data hit efficiency vs run, inside acceptance, epoch %s (both disk '
                 'sides merged)' % epoch, fontsize=12)
    fig.tight_layout()
    book.add(fig, 'runs')
    return out


def page_text(book, lines, tag):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(0.03, 0.97, '\n'.join(lines), va='top', ha='left', family='monospace',
             fontsize=8.5)
    book.add(fig, tag)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    a = parse_args()
    info = load_epoch(a)
    surfaces_req = resolve_surfaces(a.surfaces)
    tmpl, run_branch, use_mask = branch_templates(a, surfaces_req)

    if a.list_branches is not None:
        list_branches(info['data'], info['tree'], a.list_branches)
        return

    surfaces = []
    for b in surfaces_req:
        surfaces += [Surface(b, a)] if b[0] == 'L' else [Surface(b + '+', a), Surface(b + '-', a)]

    out = a.out or 'effmaps_%s' % a.epoch
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, 'effmaps_%s' % a.epoch)

    print('pixel hit-efficiency maps, epoch %s' % a.epoch)
    print('  data     : %d files' % len(info['data']))
    print('  MC       : %s  (weight %s)' % (', '.join(os.path.basename(f) for f in info['mc']),
                                            info['mc_weight'] or 'none'))
    print('  surfaces : %s' % ', '.join(s.name for s in surfaces))
    print('  muons    : %s, probes need >= %d other valid pixel hits'
          % (', '.join(a.muons), a.min_other_hits))
    print('  z0       : %s' % ('{mu}_vz' if a.z0_from == 'vz' else 'pv_z + {mu}_dz'))
    if 'charge' not in tmpl:
        warn('no charge branch: helices are straight lines (phi off by up to a few '
             'mm on the disks)')
    if 'vx' not in tmpl or 'vy' not in tmpl:
        warn('no transverse reference point: tracks start at x = y = 0, so phi at '
             'L1 is off by up to (beam offset)/(2.9 cm)')

    # every branch, every file, before any reading
    sel_all = ' & '.join(s for s in (info['selection'],) if s)
    needed = {}
    for sample, extra_sel, wt in (('data', info['data_selection'], ''),
                                  ('mc', info['mc_selection'], info['mc_weight'])):
        b = selection_branches(sel_all) | selection_branches(extra_sel)
        for mu in a.muons:
            b |= {t.format(mu=mu) for t in tmpl.values() if '{mu}' in t}
        b |= {t for t in tmpl.values() if '{mu}' not in t}
        if wt:
            b.add(wt)
        if sample == 'data':
            b.add(run_branch)
        needed[sample] = b
    if use_mask:
        print('  hit mask : %s (bits 0-3 BPix L1-L4, 4-6 FPix D1-D3)' % tmpl['mask'])
    entries = preflight({'data': info['data'], 'mc': info['mc']}, info['tree'], needed)

    print('\n[read] data')
    rd = process_sample('data', info['data'], info, tmpl, run_branch, a, surfaces,
                        use_mask, entries, is_mc=False)
    print('[read] MC')
    rm = process_sample('MC', info['mc'], info, tmpl, run_branch, a, surfaces,
                        use_mask, entries, is_mc=True)

    # ------------------------------------------------------------------ numbers
    summary = dict(epoch=a.epoch, surfaces={}, args=vars(a), geometry=GEOM,
                   n_events=dict(data=rd['n_events'], mc=rm['n_events']),
                   n_selected=dict(data=rd['n_selected'], mc=rm['n_selected']),
                   selection=info['selection'], mc_selection=info['mc_selection'],
                   data_selection=info['data_selection'], mc_weight=info['mc_weight'],
                   branches=tmpl)
    bad_rows = []
    text = [
        'Pixel hit-efficiency maps, epoch %s' % a.epoch,
        '',
        'eps(cell) = (probes crossing the cell WITH a valid hit on the surface) / (probes crossing the cell)',
        'probe     = mu1 or mu2 of a selected event, whose helix crosses the surface, with',
        '            >= %d valid pixel hits other than the one on the surface' % a.min_other_hits,
        'cell      = (z, phi) on a barrel layer, (r, phi) on one side of a disk, from the',
        '            exact helix (pt, eta, phi, charge, z0; B = %.1f T)' % GEOM['B_FIELD'],
        'MC        = weighted with %s; data unweighted' % (info['mc_weight'] or 'nothing'),
        '',
        'eps_MC on data illumination = sum_c eps_MC(c) N_data(c) / sum_c N_data(c)  (cells with',
        '            both numbers defined): MC efficiency with the DATA crossing-point spectrum.',
        'P_kill    = 1 - eps_data / eps_MC, where eps_data < eps_MC.',
        '',
        'selection : %s' % info['selection'],
        'MC only   : %s' % (info['mc_selection'] or '-'),
        'data only : %s' % (info['data_selection'] or '-'),
        '',
        'events read / selected   data %s / %s     MC %s / %s'
        % (f"{rd['n_events']:,}", f"{rd['n_selected']:,}",
           f"{rm['n_events']:,}", f"{rm['n_selected']:,}"),
        '',
        '%-6s %12s %12s %10s %10s %16s %12s %10s' % ('surf', 'probes data', 'probes MC',
                                                      'eps_data', 'eps_MC', 'eps_MC(data ill.)',
                                                      'data/MC', 'no-MC frac'),
    ]
    for s in surfaces:
        D, M = rd['maps'][s.name], rm['maps'][s.name]
        xc = 0.5 * (s.xe[1:] + s.xe[:-1])
        acc = s.in_acceptance(xc)[:, None] * np.ones((1, len(s.ye) - 1), bool)
        ed, sd, nd, em, sm, nm = surface_maps(s, rd, rm, a)
        Dd = D.den[acc].sum()
        eps_d = D.num[acc].sum() / Dd if Dd > 0 else float('nan')
        Md = M.den[acc].sum()
        eps_m = M.num[acc].sum() / Md if Md > 0 else float('nan')
        both = acc & np.isfinite(em) & (D.den > 0)
        eps_m_dill = ((em[both] * D.den[both]).sum() / D.den[both].sum()
                      if D.den[both].sum() > 0 else float('nan'))
        no_mc = (D.den[acc & ~np.isfinite(em)].sum() / Dd) if Dd > 0 else float('nan')
        summary['surfaces'][s.name] = dict(
            probes_data=float(Dd), probes_mc=float(Md), eps_data=eps_d, eps_mc=eps_m,
            eps_mc_on_data_illumination=eps_m_dill,
            data_over_mc=eps_d / eps_m_dill if eps_m_dill > 0 else float('nan'),
            frac_data_probes_in_cells_without_mc=no_mc)
        text.append('%-6s %12.0f %12.0f %10.4f %10.4f %16.4f %12.4f %10.4f'
                    % (s.name, Dd, Md, eps_d, eps_m, eps_m_dill,
                       eps_d / eps_m_dill if eps_m_dill > 0 else float('nan'), no_mc))
        bad = (np.isfinite(ed) & np.isfinite(em) & (ed < a.bad_eps_data)
               & (em > a.bad_eps_mc))
        for i, j in zip(*np.nonzero(bad)):
            bad_rows.append(dict(surface=s.name,
                                 x_lo=s.xe[i], x_hi=s.xe[i + 1],
                                 phi_lo=s.ye[j], phi_hi=s.ye[j + 1],
                                 eps_data=ed[i, j], eps_mc=em[i, j],
                                 probes_data=nd[i, j], probes_mc=nm[i, j]))
        summary['surfaces'][s.name]['n_bad_cells'] = int(bad.sum())
    text += ['', 'bad cells (eps_data < %.2f and eps_MC > %.2f): %d, listed in %s'
             % (a.bad_eps_data, a.bad_eps_mc, len(bad_rows),
                os.path.basename(stem.replace('effmaps_', 'bad_cells_')) + '.csv')]
    text += ['', 'geometry (approximate, CHECK): BPix r = %s cm, |z| < %.1f cm; FPix |z| = %s cm, '
             '%.1f < r < %.1f cm' % (GEOM['BPIX_R'], GEOM['BPIX_HALF_Z'], GEOM['FPIX_Z'],
                                    GEOM['FPIX_R'][0], GEOM['FPIX_R'][1])]
    text += ['branches  : ' + ', '.join('%s=%s' % kv for kv in sorted(tmpl.items()))]

    print('\n' + '\n'.join(text[17:]))

    # -------------------------------------------------------------------- plots
    plt = setup_mpl()
    book = Book(stem + '.pdf', png_prefix=stem if a.png else None)
    page_text(book, text, 'summary')
    page_z0(book, rd, rm, a.epoch)
    print('\n[plot] %s' % (stem + '.pdf'))
    pprog = Progress(len(surfaces), 'plots', unit='surf')
    for i, s in enumerate(surfaces, 1):
        ed, sd, nd, em, sm, nm = surface_maps(s, rd, rm, a)
        page_main(book, s, ed, em, a.epoch)
        page_kill(book, s, ed, sd, em, sm, a.epoch)
        page_illumination(book, s, rd, rm, a.epoch)
        page_projections(book, s, rd, rm, a.epoch)
        pprog.update(i)
    summary['runs'] = page_runs(book, rd, list(dict.fromkeys(s.base for s in surfaces)),
                                a, a.epoch)
    book.close()

    # ------------------------------------------------------------------ outputs
    arrays = {}
    for s in surfaces:
        k = s.name.replace('+', 'p').replace('-', 'm')
        arrays['%s_xedges' % k] = s.xe
        arrays['%s_phiedges' % k] = s.ye
        for sample, res in (('data', rd), ('mc', rm)):
            A = res['maps'][s.name]
            for f in ('num', 'den', 'num2', 'den2'):
                arrays['%s_%s_%s' % (k, sample, f)] = getattr(A, f)
    np.savez_compressed(stem + '.npz', **arrays)
    with open(stem + '.json', 'w') as fh:
        json.dump(summary, fh, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    bad_path = os.path.join(out, 'bad_cells_%s.csv' % a.epoch)
    with open(bad_path, 'w', newline='') as fh:
        cols = ['surface', 'x_lo', 'x_hi', 'phi_lo', 'phi_hi', 'eps_data', 'eps_mc',
                'probes_data', 'probes_mc']
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in bad_rows:
            wr.writerow({c: ('%.4f' % r[c] if isinstance(r[c], float) else r[c]) for c in cols})
    print('[out] %s.pdf, %s.npz, %s.json, %s' % (stem, stem, stem, bad_path))


if __name__ == '__main__':
    main()
