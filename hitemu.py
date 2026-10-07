"""
hitemu.py -- pixel hit-loss emulation for covflow: the shared library.

Used by
    build_kill_maps.py     step 1  run ranges + per-range efficiency / kill maps
    noL1_data_study.py     step 4  no-L1 mixture test + route B hit-error fit
    emulate_hit_loss.py    steps 2, 3, 5  hit killing in MC, context closure,
                           route A / route B covariances and their checks

and meant to be importable from the ntuplizer later: everything that acts on
tracks is plain numpy, vectorised over a leading axis, so the same functions
handle one track or ten million.

Conventions (identical to pixel_eff_maps.py, which this module imports)
-----------------------------------------------------------------------
surface   'L1'..'L4' (BPix layer), 'D1+','D1-',... (FPix disk, z > 0 / z < 0)
cell      (z, phi) on a barrel layer, (r, phi) on a disk side
eps       fraction of probes crossing a cell that have a valid hit there
range     a block of consecutive data runs with one efficiency map each

Curvilinear track parameters, CMSSW order (reco::TrackBase):
    0 qoverp [1/GeV]  1 lambda [rad]  2 phi [rad]  3 dxy [cm]  4 dsz [cm]
The packed 15-vector order of the covariance is features.PACK_NAMES.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

import pixel_eff_maps as P
from pixel_eff_maps import GEOM, die, warn, Progress
import features as F          # covflow/features.py: numpy only

KILL_SURFACES = ('L1', 'D1+', 'D1-')
PACK_NAMES = list(F.PACK_NAMES)

# The branch set Ric's 2026 runs used: transverse reference point = PV,
# z0 = pv_z + dz. Overridable with --branch / --z0-from.
DEFAULT_BRANCH_OVERRIDES = ['vx=pv_x', 'vy=pv_y']
DEFAULT_Z0_FROM = 'dz_pv'

# per-muon branches this module needs on top of pixel_eff_maps.BRANCHES
EXTRA_TEMPLATES = dict(
    n_pix_b='{mu}_n_pix_b_hit',
    n_pix_e='{mu}_n_pix_e_hit',
    bs_dxy='{mu}_bs_dxy',          # optional: only for the IP-significance check
    bs_dxy_e='{mu}_bs_dxy_e',      # optional
)
OPTIONAL_EXTRA = {'bs_dxy', 'bs_dxy_e'}


# ---------------------------------------------------------------------------
# command line shared by the three scripts
# ---------------------------------------------------------------------------

def add_common_args(p, default_config):
    p.add_argument('--epoch', required=True)
    p.add_argument('--config', default=default_config,
                   help='covflow epoch config (default: %(default)s)')
    p.add_argument('--muons', nargs='+', default=['mu1', 'mu2'])
    p.add_argument('--branch', action='append', default=None, metavar='KEY=TEMPLATE',
                   help='override a branch template (default: %s)'
                        % ' '.join(DEFAULT_BRANCH_OVERRIDES))
    p.add_argument('--z0-from', choices=('vz', 'dz_pv'), default=DEFAULT_Z0_FROM)
    p.add_argument('--min-other-hits', type=int, default=2,
                   help='probe definition: valid pixel hits besides the one '
                        'on the surface (default: %(default)s)')
    p.add_argument('--max-events', type=int, default=None,
                   help='entries per sample, for tests')
    p.add_argument('--step-size', default='200 MB')
    p.add_argument('--xrootd', action='store_true')
    return p


def resolve_branches(a, need_cov=False, need_bs=False):
    """(per-muon templates, run branch). Fail-loud on unknown keys."""
    if a.branch is None:
        a.branch = list(DEFAULT_BRANCH_OVERRIDES)
    base_items, extra = [], dict(EXTRA_TEMPLATES)
    for item in a.branch:
        key, sep, val = item.partition('=')
        if key in extra:
            extra[key] = val
        else:
            base_items.append(item)
    a2 = type('A', (), {})()
    a2.branch, a2.z0_from = base_items, a.z0_from
    tmpl, run_branch, _ = P.branch_templates(a2, ['L1', 'D1'])
    for k in ('n_pix_b', 'n_pix_e'):
        if not extra[k]:
            die('branch %r is required' % k)
        tmpl[k] = extra[k]
    if need_bs:
        for k in ('bs_dxy', 'bs_dxy_e'):
            if extra[k]:
                tmpl[k] = extra[k]
    if need_cov:
        for n in PACK_NAMES:
            tmpl['cov_' + n] = '{mu}_cov_' + n
    return tmpl, run_branch


def sample_branches(tmpl, muons, sel_expr, extra=()):
    b = set(P.selection_branches(sel_expr)) | set(extra)
    for mu in muons:
        b |= {t.format(mu=mu) for t in tmpl.values() if '{mu}' in t}
    b |= {t for t in tmpl.values() if '{mu}' not in t}
    return b


def selection_expr(info, is_mc):
    parts = [info['selection'], info['mc_selection'] if is_mc else info['data_selection']]
    return ' & '.join('(%s)' % s for s in parts if s)


def drop_missing_optional(tmpl, files, tree, muons):
    """Remove optional templates whose branches are absent in the first file."""
    import uproot
    with uproot.open(files[0]) as fh:
        have = set(fh[tree].keys())
    for k in list(tmpl):
        if k in OPTIONAL_EXTRA and tmpl[k].format(mu=muons[0]) not in have:
            warn('optional branch %s not found: the checks that need it are skipped'
                 % tmpl[k].format(mu=muons[0]))
            del tmpl[k]
    return tmpl


def iterate(files, tree, branches, a, entries, label):
    """Yield numpy chunks with a progress line (cap: a.max_events entries)."""
    import uproot
    total = sum(entries[f] for f in files)
    if a.max_events:
        total = min(total, a.max_events)
    prog = Progress(total, label)
    done = 0
    for chunk in uproot.iterate([{f: tree} for f in files],
                                expressions=sorted(branches), library='np',
                                step_size=a.step_size):
        n = len(next(iter(chunk.values())))
        if a.max_events and done + n > a.max_events:
            n = a.max_events - done
            chunk = {k: v[:n] for k, v in chunk.items()}
        yield chunk, n
        done += n
        prog.update(done)
        if a.max_events and done >= a.max_events:
            break


def muon_view(chunk, mu, tmpl, a, sel):
    """pixel_eff_maps.muon_view plus hit counts, covariance (N,15) if asked,
    and the optional beamspot IP."""
    v = P.muon_view(chunk, mu, {k: t for k, t in tmpl.items()
                                if k in P.BRANCHES}, a.z0_from, sel)
    g = lambda key: np.asarray(chunk[tmpl[key].format(mu=mu)], np.float64)[sel]
    v['n_pix_b'] = g('n_pix_b')
    v['n_pix_e'] = g('n_pix_e')
    for k in ('bs_dxy', 'bs_dxy_e'):
        if k in tmpl:
            v[k] = g(k)
    if 'cov_' + PACK_NAMES[0] in tmpl:
        v['cov'] = np.stack([g('cov_' + n) for n in PACK_NAMES], axis=1)
    return v


def good_track(v):
    return (np.isfinite(v['pt']) & (v['pt'] > 0) & np.isfinite(v['eta'])
            & np.isfinite(v['phi']) & np.isfinite(v['z0']))


# ---------------------------------------------------------------------------
# where a track crosses each pixel surface
# ---------------------------------------------------------------------------

def surface_kind(name):
    return ('L', int(name[1]), 0) if name[0] == 'L' else \
           ('D', int(name[1]), 1 if name.endswith('+') else -1)


def cross(name, pt, eta, phi, q, x0, y0, z0):
    """(x, phi): x = z on a layer, r on a disk; NaN if not crossed."""
    kind, k, side = surface_kind(name)
    if kind == 'L':
        return P.cross_barrel(GEOM['BPIX_R'][k], pt, eta, phi, q, x0, y0, z0, GEOM['B_FIELD'])
    return P.cross_disk(side * GEOM['FPIX_Z'][k], pt, eta, phi, q, x0, y0, z0, GEOM['B_FIELD'])


def in_acceptance(name, x):
    kind = name[0]
    with np.errstate(invalid='ignore'):
        if kind == 'L':
            return np.isfinite(x) & (np.abs(x) < GEOM['BPIX_HALF_Z'])
        return np.isfinite(x) & (x > GEOM['FPIX_R'][0]) & (x < GEOM['FPIX_R'][1])


def cross_view(name, v, sel=None):
    s = slice(None) if sel is None else sel
    return cross(name, v['pt'][s], v['eta'][s], v['phi'][s], v['q'][s],
                 v['x0'][s], v['y0'][s], v['z0'][s])


def has_hit(name, v):
    """Valid hit on L1 / D1 from the first-hit branches (no hit mask needed)."""
    if name == 'L1':
        return v['first_b'] == 1
    if name.startswith('D1'):
        return v['first_e'] == 1
    raise ValueError('has_hit(%s) needs a per-layer hit mask' % name)


def cell_index(xe, ye, x, y):
    ix = np.searchsorted(xe, x, side='right') - 1
    iy = np.searchsorted(ye, y, side='right') - 1
    ok = np.isfinite(x) & np.isfinite(y) & (ix >= 0) & (ix < len(xe) - 1) \
        & (iy >= 0) & (iy < len(ye) - 1)
    return np.where(ok, ix, 0), np.where(ok, iy, 0), ok


def surface_edges(name, nbins_phi, nbins_r):
    a = type('A', (), {})()
    a.nbins_phi, a.nbins_r = nbins_phi, nbins_r
    s = P.Surface(name, a)
    return s.xe, s.ye


# ---------------------------------------------------------------------------
# run ranges
# ---------------------------------------------------------------------------

def _seg_cost(cn, cd, i, j):
    """-2 lnL of the pooled binomial on runs [i, j), summed over surfaces."""
    out = 0.0
    for s in cn:
        n = cn[s][j] - cn[s][i]
        d = cd[s][j] - cd[s][i]
        if d <= 0:
            continue
        p = n / d
        if 0 < p < 1:
            out -= 2 * (n * np.log(p) + (d - n) * np.log(1 - p))
    return out


def _best_split(cn, cd, i, j, primary, min_den):
    """Best split point k in (i, j) and its gain in -2 lnL."""
    best_k, best_gain = None, 0.0
    parent = _seg_cost(cn, cd, i, j)
    dp = cd[primary]
    for k in range(i + 1, j):
        if dp[k] - dp[i] < min_den or dp[j] - dp[k] < min_den:
            continue
        g = parent - _seg_cost(cn, cd, i, k) - _seg_cost(cn, cd, k, j)
        if g > best_gain:
            best_k, best_gain = k, g
    return best_k, best_gain


def segment_runs(runs, num, den, min_den, max_ranges, min_gain, primary='L1'):
    """
    Split a sorted list of runs into at most `max_ranges` blocks of consecutive
    runs, by binary segmentation on the pooled binomial likelihood of the
    per-run efficiencies of all surfaces together.

    runs      (R,) sorted run numbers
    num, den  {surface: (R,) hits / probes per run, inside acceptance}
    min_den   minimum probes of the primary surface in every block
    min_gain  a split is made only if it lowers -2 lnL by more than this

    Each step splits the block whose best split gains most. Run-to-run
    variations are much larger than the statistical errors in these data, so in
    practice the number of blocks is set by max_ranges and min_den, and min_gain
    only stops splitting a flat epoch.

    Returns [(i0, i1), ...] index ranges into `runs` and the gains of the splits.
    """
    cn = {s: np.concatenate([[0.0], np.cumsum(num[s])]) for s in num}
    cd = {s: np.concatenate([[0.0], np.cumsum(den[s])]) for s in den}
    segs = [(0, len(runs))]
    gains = []
    while len(segs) < max_ranges:
        cands = [(_best_split(cn, cd, i, j, primary, min_den), (i, j)) for i, j in segs]
        cands = [(k, g, ij) for (k, g), ij in cands if k is not None]
        if not cands:
            break
        k, g, (i, j) = max(cands, key=lambda c: c[1])
        if g < min_gain:
            break
        segs.remove((i, j))
        segs += [(i, k), (k, j)]
        segs.sort()
        gains.append(dict(first_run=int(runs[k]), gain=float(g)))
    return segs, gains


def read_lumi_csv(path):
    """brilcalc csv (--output-style csv): 'run:fill,...,delivered,recorded'
    -> {run: recorded}. Lines starting with '#' are skipped."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith('#'):
                continue
            cols = line.strip().split(',')
            try:
                run = int(cols[0].split(':')[0])
                out[run] = out.get(run, 0.0) + float(cols[-1])
            except ValueError:
                continue
    if not out:
        die('no run lines read from %s' % path)
    return out


# ---------------------------------------------------------------------------
# kill maps
# ---------------------------------------------------------------------------

class KillMaps:
    """
    Per run range: data efficiency per cell. Once: MC efficiency per cell.
    From them, per range and cell:
        p_kill  = 1 - eps_data/eps_MC   where eps_data < eps_MC, else 0
        w_hit   = eps_data/eps_MC       where eps_data > eps_MC, else 1
        w_nohit = (1-eps_data)/(1-eps_MC) where eps_data > eps_MC, else 1
    Killing reproduces data where data is less efficient than MC; the two
    weights where it is more efficient. The weights are bounded by 1/eps_MC,
    which is small only where MC is efficient: cells where eps_data/eps_MC
    exceeds `max_weight` (typically a module dead in the MC conditions and
    alive in data) cannot be emulated -- MC has no hits there to reweight --
    and are left unchanged and flagged in `uncorrectable`.

    Cells with too few probes fall back, recorded per cell in `level_*`:
      data  0 = this range and cell
            1 = the nearest ranges in time around this one, widened
                symmetrically until the cell has enough probes, scaled by the
                ratio of the surface averages (this range / the window)
            2 = this range, surface average in acceptance
            3 = nothing (no probes anywhere): no kill, weight 1
      MC    0 = cell, 1 = surface average, 3 = nothing
    """

    def __init__(self, surfaces, edges, ranges, lumi, d_num, d_den, m_num, m_den,
                 m_num2, m_den2, meta):
        self.surfaces = list(surfaces)
        self.edges = edges                      # {s: (xe, ye)}
        self.ranges = [tuple(map(int, r)) for r in ranges]   # [(first, last)]
        self.lumi = np.asarray(lumi, float)
        self.lumi = self.lumi / self.lumi.sum()
        self.d_num, self.d_den = d_num, d_den   # {s: (R, nx, ny)}
        self.m_num, self.m_den = m_num, m_den   # {s: (nx, ny)}
        self.m_num2, self.m_den2 = m_num2, m_den2
        self.meta = dict(meta)
        self.finalised = False

    # ---------------------------------------------------------------- build
    def finalise(self, min_cell, max_weight=1.5, prior=None, min_weight=0.2):
        self.min_cell = float(min_cell)
        self.max_weight = float(max_weight)
        self.min_weight = float(min_weight)
        self.prior = float(0.0 if prior is None else prior)
        self.uncorrectable = {}
        self.eps_d, self.eps_m, self.level_d, self.level_m = {}, {}, {}, {}
        self.p_kill, self.w_hit, self.w_nohit = {}, {}, {}
        for s in self.surfaces:
            xe, ye = self.edges[s]
            xc = 0.5 * (xe[1:] + xe[:-1])
            acc = in_acceptance(s, xc)[:, None] & np.ones((1, len(ye) - 1), bool)
            dn, dd = self.d_num[s], self.d_den[s]
            R = dn.shape[0]
            td = dd.sum(0)
            with np.errstate(divide='ignore', invalid='ignore'):
                surf_r = np.array([dn[r][acc].sum() / dd[r][acc].sum()
                                   if dd[r][acc].sum() > 0 else np.nan for r in range(R)])
            mn, md, md2 = self.m_num[s], self.m_den[s], self.m_den2[s]
            with np.errstate(divide='ignore', invalid='ignore'):
                neff = md ** 2 / md2
                em_c = mn / md
                em_s = mn[acc].sum() / md[acc].sum() if md[acc].sum() > 0 else np.nan
            em = np.where(md > 0, em_s, np.nan)
            levm = np.where(md > 0, 1, 3).astype(np.int8)
            okm = (md > 0) & (neff >= self.min_cell)
            em = np.where(okm, em_c, em)
            levm = np.where(okm, 0, levm)

            # Optional: data efficiencies pulled toward eps_MC by `prior`
            # pseudo-probes, eps = (hits + prior * eps_MC) / (probes + prior).
            # Off by default: it also pulls truly dead cells with few probes
            # up (30 probes, prior 30: P_kill 0.5 instead of 1). The noise it
            # would cure -- a sparse cell fluctuating to eps_data = 1 gives the
            # MC tracks without the hit there weight 0 -- is handled by the
            # floor `min_weight` on that weight instead.
            em0 = np.where(np.isfinite(em), em, 0.0)

            def shrink(n, d):
                with np.errstate(divide='ignore', invalid='ignore'):
                    return np.where(np.isfinite(em), (n + self.prior * em0) / (d + self.prior),
                                    n / d)

            eps = np.full(dn.shape, np.nan)
            lev = np.full(dn.shape, 3, np.int8)
            cn = np.concatenate([np.zeros((1,) + dn.shape[1:]), np.cumsum(dn, 0)])
            cd = np.concatenate([np.zeros((1,) + dd.shape[1:]), np.cumsum(dd, 0)])
            for r in range(R):
                if dd[r].sum() > 0:
                    eps[r] = surf_r[r]
                    lev[r] = 2
                # level 1: nearest ranges, widened until enough probes
                filled = np.zeros(dn.shape[1:], bool)
                for k in range(R - 1, 0, -1):      # widest first, overwritten by narrower
                    lo, hi = max(0, r - k), min(R, r + k + 1)
                    wn, wd = cn[hi] - cn[lo], cd[hi] - cd[lo]
                    ok = wd >= self.min_cell
                    with np.errstate(divide='ignore', invalid='ignore'):
                        s_win = wn[acc].sum() / wd[acc].sum()
                        val = np.clip(shrink(wn, wd) * (surf_r[r] / s_win), 0, 1)
                    eps[r] = np.where(ok, val, eps[r])
                    filled |= ok
                lev[r] = np.where(filled, 1, lev[r])
                l0 = dd[r] >= self.min_cell
                eps[r] = np.where(l0, shrink(dn[r], dd[r]), eps[r])
                lev[r] = np.where(l0, 0, lev[r])
                # no probe of this range anywhere near: leave no-change
                lev[r] = np.where((td <= 0) & ~l0, 3, lev[r])
                eps[r] = np.where(lev[r] == 3, np.nan, eps[r])

            with np.errstate(divide='ignore', invalid='ignore'):
                emb = em[None]
                defined = np.isfinite(eps) & np.isfinite(emb)
                pk = np.where(defined & (eps < emb) & (emb > 0), 1 - eps / emb, 0.0)
                wh = np.where(defined & (eps > emb) & (emb > 0), eps / emb, 1.0)
                wn = np.where(defined & (eps > emb) & (emb < 1), (1 - eps) / (1 - emb), 1.0)
            self.eps_d[s], self.level_d[s] = eps, lev
            self.eps_m[s], self.level_m[s] = em, levm
            with np.errstate(invalid='ignore'):
                unc = defined & (eps > emb) & ((emb <= 0) | (eps > self.max_weight * emb))
            wh = np.where(unc, 1.0, wh)
            wn = np.where(unc, 1.0, wn)
            self.uncorrectable[s] = unc
            self.p_kill[s] = np.clip(np.nan_to_num(pk), 0, 1)
            self.w_hit[s] = np.nan_to_num(wh, nan=1.0)
            self.w_nohit[s] = np.clip(np.nan_to_num(wn, nan=1.0), self.min_weight, None)
        self.finalised = True
        return self

    # ---------------------------------------------------------------- use
    def range_of_run(self, runs):
        first = np.array([r[0] for r in self.ranges])
        idx = np.searchsorted(first, np.asarray(runs), side='right') - 1
        return np.clip(idx, 0, len(self.ranges) - 1)

    def lookup(self, s, x, y):
        xe, ye = self.edges[s]
        return cell_index(xe, ye, x, y)

    def lumi_avg_eps_data(self, s):
        e = self.eps_d[s]
        w = self.lumi[:, None, None] * np.isfinite(e)
        with np.errstate(invalid='ignore'):
            return np.nansum(e * self.lumi[:, None, None], 0) / w.sum(0)

    # ---------------------------------------------------------------- io
    def save(self, stem):
        arr = dict(lumi=self.lumi, ranges=np.array(self.ranges, np.int64))
        for s in self.surfaces:
            k = s.replace('+', 'p').replace('-', 'm')
            arr[k + '_xedges'], arr[k + '_phiedges'] = self.edges[s]
            arr[k + '_d_num'], arr[k + '_d_den'] = self.d_num[s], self.d_den[s]
            arr[k + '_m_num'], arr[k + '_m_den'] = self.m_num[s], self.m_den[s]
            arr[k + '_m_num2'], arr[k + '_m_den2'] = self.m_num2[s], self.m_den2[s]
        np.savez_compressed(stem + '.npz', **arr)
        meta = dict(self.meta, surfaces=self.surfaces, ranges=self.ranges,
                    lumi_fraction=self.lumi.tolist(),
                    min_cell=getattr(self, 'min_cell', None),
                    max_weight=getattr(self, 'max_weight', None),
                    min_weight=getattr(self, 'min_weight', None),
                    prior=getattr(self, 'prior', None))
        with open(stem + '.json', 'w') as fh:
            json.dump(meta, fh, indent=1, default=lambda o: o.item()
                      if hasattr(o, 'item') else str(o))

    @classmethod
    def load(cls, path, min_cell=None, max_weight=None):
        stem = path[:-4] if path.endswith('.npz') else path
        if not os.path.isfile(stem + '.npz') or not os.path.isfile(stem + '.json'):
            die('kill maps %s.npz/.json not found (run build_kill_maps.py first)' % stem)
        z = np.load(stem + '.npz')
        meta = json.load(open(stem + '.json'))
        surfaces = meta['surfaces']
        g = lambda s, f: z[s.replace('+', 'p').replace('-', 'm') + '_' + f]
        km = cls(surfaces, {s: (g(s, 'xedges'), g(s, 'phiedges')) for s in surfaces},
                 [tuple(r) for r in z['ranges']], z['lumi'],
                 {s: g(s, 'd_num') for s in surfaces}, {s: g(s, 'd_den') for s in surfaces},
                 {s: g(s, 'm_num') for s in surfaces}, {s: g(s, 'm_den') for s in surfaces},
                 {s: g(s, 'm_num2') for s in surfaces}, {s: g(s, 'm_den2') for s in surfaces},
                 meta)
        return km.finalise(min_cell if min_cell is not None else meta.get('min_cell') or 30,
                           max_weight if max_weight is not None else meta.get('max_weight') or 1.5,
                           meta.get('prior'), meta.get('min_weight') or 0.2)


# ---------------------------------------------------------------------------
# hit killing
# ---------------------------------------------------------------------------

def emulate(v, range_idx, km, rng):
    """
    Kill L1 / D1 hits of MC tracks according to the kill maps of each track's
    run range. Returns a dict of per-track arrays:

      kill_L1, kill_D1   bool, hit removed
      w                  weight from the cells where data beats MC (<= 1/eps_MC)
      cell_ok_L1/_D1     the track crosses a mapped cell
      n_pix, n_pix_b, n_pix_e, first_b, first_e   the context after killing

    Only hits that exist are removed; a removed L1 hit makes the first BPix
    layer the next one the track crosses in acceptance (L2, L3, L4), or 0 if no
    BPix hit is left; same for the disks. Without a per-layer hit mask this is
    an assumption: a track that had L1 and L3 but no L2 would get 2, not 3.
    """
    N = len(v['pt'])
    out = dict(kill_L1=np.zeros(N, bool), kill_D1=np.zeros(N, bool),
               w=np.ones(N), cell_ok_L1=np.zeros(N, bool), cell_ok_D1=np.zeros(N, bool))
    for s in km.surfaces:
        x, y = cross_view(s, v)
        ix, iy, ok = km.lookup(s, x, y)
        r = range_idx
        pk = np.where(ok, km.p_kill[s][r, ix, iy], 0.0)
        wh = np.where(ok, km.w_hit[s][r, ix, iy], 1.0)
        wn = np.where(ok, km.w_nohit[s][r, ix, iy], 1.0)
        hit = has_hit(s, v) & ok
        kill = hit & (rng.random(N) < pk)
        out['w'] *= np.where(ok, np.where(hit, wh, wn), 1.0)
        key = 'L1' if s == 'L1' else 'D1'
        out['kill_' + key] |= kill
        out['cell_ok_' + key] |= ok
    kL1, kD1 = out['kill_L1'], out['kill_D1']
    out['n_pix'] = v['n_pix'] - kL1 - kD1
    out['n_pix_b'] = v['n_pix_b'] - kL1
    out['n_pix_e'] = v['n_pix_e'] - kD1
    out['first_b'] = _next_first(v, kL1, out['n_pix_b'], v['first_b'],
                                 ['L2', 'L3', 'L4'], default=2)
    out['first_e'] = _next_first(v, kD1, out['n_pix_e'], v['first_e'],
                                 None, default=2)
    return out


def _next_first(v, killed, n_left, first, layers, default):
    new = first.copy().astype(np.float64)
    idx = np.nonzero(killed)[0]
    if len(idx) == 0:
        return new
    nxt = np.full(len(idx), np.nan)
    if layers is None:          # disks: the track's side
        side = np.where(v['eta'][idx] >= 0, '+', '-')
        for k in (2, 3):
            for sd in ('+', '-'):
                m = (side == sd) & ~np.isfinite(nxt)
                if not m.any():
                    continue
                x, _ = cross_view('D%d%s' % (k, sd), v, idx[m])
                acc = in_acceptance('D%d%s' % (k, sd), x)
                sub = np.nonzero(m)[0]
                nxt[sub[acc]] = k
    else:
        for name in layers:
            m = ~np.isfinite(nxt)
            if not m.any():
                break
            x, _ = cross_view(name, v, idx[m])
            acc = in_acceptance(name, x)
            sub = np.nonzero(m)[0]
            nxt[sub[acc]] = int(name[1])
    nxt = np.where(np.isfinite(nxt), nxt, default)
    new[idx] = np.where(n_left[idx] > 0, nxt, 0)
    return new


# ---------------------------------------------------------------------------
# covariance algebra
# ---------------------------------------------------------------------------

def packed_to_matrix(packed):
    return F.packed_to_matrix(np.asarray(packed, np.float64))


def matrix_to_packed(M):
    return F.matrix_to_packed(M)


def sigmas(M):
    return np.sqrt(np.clip(np.diagonal(M, axis1=-2, axis2=-1), 0, None))


def curv_to_helix(par):
    """(N,5) curvilinear (qoverp, lambda, phi, dxy, dsz) at the point of closest
    approach to the origin -> pt, eta, phi0, q, x0, y0, z0."""
    qop, lam, phi, dxy, dsz = [par[:, i] for i in range(5)]
    p = 1.0 / np.abs(qop)
    pt = p * np.cos(lam)
    q = np.sign(qop)
    theta = 0.5 * np.pi - lam
    eta = -np.log(np.tan(0.5 * theta))
    x0 = -dxy * np.sin(phi)
    y0 = dxy * np.cos(phi)
    z0 = dsz / np.cos(lam)
    return pt, eta, phi, q, x0, y0, z0


def helix_to_curv(pt, eta, phi, q, z0, dxy=None):
    lam = 0.5 * np.pi - 2 * np.arctan(np.exp(-eta))
    p = pt / np.cos(lam)
    dxy = np.zeros_like(pt) if dxy is None else dxy
    return np.stack([q / p, lam, phi, dxy, z0 * np.cos(lam)], axis=1)


def measure(name, par):
    """Local coordinates of the crossing: barrel (r*phi, z), disk (r*phi, r).
    Returns (u_phi, v, r): u is reported as the azimuth, scaled by r by the
    caller, so that differences can be wrapped."""
    pt, eta, phi, q, x0, y0, z0 = curv_to_helix(par)
    x, ph = cross(name, pt, eta, phi, q, x0, y0, z0)
    kind, k, _ = surface_kind(name)
    r = np.full_like(x, GEOM['BPIX_R'][k]) if kind == 'L' else x
    return ph, x, r


FD_STEPS = np.array([1e-3, 1e-5, 1e-5, 1e-4, 1e-4])   # qoverp relative, then absolute


def jacobian(name, par):
    """H = d(u, v)/d(curvilinear params), (N,2,5), by central differences of
    the same helix that places tracks on the maps. u = r*phi along the
    surface, v = z (barrel) or r (disk). NaN rows where the track does not
    cross the surface."""
    par = np.asarray(par, np.float64)
    N = len(par)
    H = np.full((N, 2, 5), np.nan)
    for i in range(5):
        h = FD_STEPS[i] * (np.abs(par[:, 0]) if i == 0 else 1.0)
        pp, pm = par.copy(), par.copy()
        pp[:, i] += h
        pm[:, i] -= h
        php, xp, rp = measure(name, pp)
        phm, xm, rm = measure(name, pm)
        dphi = np.angle(np.exp(1j * (php - phm)))
        H[:, 0, i] = 0.5 * (rp + rm) * dphi / (2 * h)
        H[:, 1, i] = (xp - xm) / (2 * h)
    return H


def remove_hit(C, H, V):
    """
    Covariance without one hit:  C' = (C^-1 - H^T V^-1 H)^-1
                                    = C + C H^T (V - H C H^T)^-1 H C
    C (N,5,5), H (N,2,5), V (N,2,2). V - H C H^T is the covariance of the
    smoothed residual of that hit and must be positive definite; where it is
    not (V too small for this track), the track is flagged and C' = NaN.
    Returns (C', ok).
    """
    HC = H @ C
    S = V - HC @ np.swapaxes(H, -1, -2)
    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
    ok = np.isfinite(det) & (det > 0) & (S[:, 0, 0] > 0)
    Sinv = np.zeros_like(S)
    Sinv[:, 0, 0] = S[:, 1, 1]
    Sinv[:, 1, 1] = S[:, 0, 0]
    Sinv[:, 0, 1] = -S[:, 0, 1]
    Sinv[:, 1, 0] = -S[:, 1, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        Sinv /= det[:, None, None]
    Cp = C + np.swapaxes(HC, -1, -2) @ Sinv @ HC
    Cp = 0.5 * (Cp + np.swapaxes(Cp, -1, -2))
    Cp[~ok] = np.nan
    return Cp, ok


def smear(Cp, C0, rng):
    """
    delta ~ N(0, C' - C0), so that theta + delta has covariance C' if theta had
    C0 (independent increments of nested fits). Negative eigenvalues of
    C' - C0 are clipped to zero. Returns (delta (N,5), clipped (N,) bool,
    relative size of the most negative eigenvalue (N,)).
    """
    D = Cp - C0
    D = 0.5 * (D + np.swapaxes(D, -1, -2))
    good = np.all(np.isfinite(D), axis=(1, 2))
    lam = np.zeros((len(D), 5))
    Q = np.tile(np.eye(5), (len(D), 1, 1))
    if good.any():
        lam[good], Q[good] = np.linalg.eigh(D[good])
    scale = np.abs(lam).max(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        neg_rel = np.where(scale > 0, np.minimum(lam.min(1), 0) / scale, 0.0)
    clipped = good & (neg_rel < -1e-3)      # negative part > 0.1% of the largest
    xi = rng.standard_normal((len(D), 5))
    delta = np.einsum('nij,nj->ni', Q, np.sqrt(np.clip(lam, 0, None)) * xi)
    delta[~good] = np.nan
    return delta, clipped, neg_rel


# ---------------------------------------------------------------------------
# route B: effective hit covariance V
# ---------------------------------------------------------------------------

class HitErrors:
    """
    Effective hit covariance V per surface family ('L1', 'D1') and |eta| bin,
    from noL1_data_study.py. Several files may be given: for each surface the
    first file with at least one fitted bin is used (e.g. the MC fit first, the
    data fit as fallback for a surface MC has no dead cells on).

    |eta| bins without a fit in the file used are filled, in this order:
      1. from a later file that has them, scaled by the median ratio
         (file used / later file) over the bins both have -- the |eta| shape
         from data, the scale from MC, when MC has dead cells in a few bins only;
      2. otherwise from the nearest fitted bin of the same file.
    """

    def __init__(self, paths):
        paths = [paths] if isinstance(paths, str) else list(paths)
        self.files = []
        for path in paths:
            if not os.path.isfile(path):
                die('route B hit-error file %s not found (run noL1_data_study.py)' % path)
            self.files.append((path, json.load(open(path))))
        self.path = paths[0]
        self.sample = self.files[0][1].get('sample')
        self.choice = {}
        self.how = {}
        for fam in ('L1', 'D1'):
            for i, (path, d) in enumerate(self.files):
                e = d['surfaces'].get(fam)
                if e and any(u is not None for u in e['sigma_u_cm']):
                    self.choice[fam] = (path, d.get('sample'), self._complete(fam, e, i))
                    break

    def _complete(self, fam, e, i_used):
        e = dict(e)
        su = [None if x is None else float(x) for x in e['sigma_u_cm']]
        sv = [None if x is None else float(x) for x in e['sigma_v_cm']]
        how = ['fit' if u is not None else None for u in su]
        for path, d in self.files[i_used + 1:]:
            f = d['surfaces'].get(fam)
            if not f or list(f['abs_eta_edges']) != list(e['abs_eta_edges']):
                continue
            common = [k for k in range(len(su)) if su[k] is not None and how[k] == 'fit'
                      and f['sigma_u_cm'][k] is not None]
            if not common:
                continue
            ru = float(np.median([su[k] / f['sigma_u_cm'][k] for k in common]))
            rv = float(np.median([sv[k] / f['sigma_v_cm'][k] for k in common]))
            for k in range(len(su)):
                if su[k] is None and f['sigma_u_cm'][k] is not None:
                    su[k], sv[k] = f['sigma_u_cm'][k] * ru, f['sigma_v_cm'][k] * rv
                    how[k] = '%s x %.3f/%.3f' % (os.path.basename(path), ru, rv)
        e['sigma_u_cm'], e['sigma_v_cm'] = su, sv
        self.how[fam] = how
        return e

    def describe(self):
        out = ['route B hit errors (sigma_u / sigma_v per |eta| bin):']
        for fam in ('L1', 'D1'):
            if fam not in self.choice:
                out.append('  %s: NO FIT in %s' % (fam, ', '.join(p for p, _ in self.files)))
                continue
            path, sample, e = self.choice[fam]
            edges = e['abs_eta_edges']
            fitted = [i for i, (u, v) in enumerate(zip(e['sigma_u_cm'], e['sigma_v_cm']))
                      if u is not None and v is not None]
            out.append('  %s (%s fit, %s):' % (fam, sample, os.path.basename(path)))
            for i in range(len(edges) - 1):
                lab = '%.1f-%.1f' % (edges[i], edges[i + 1])
                if i in fitted:
                    h = self.how[fam][i]
                    out.append('     |eta| %-8s %6.1f / %6.1f um  %s'
                               % (lab, 1e4 * e['sigma_u_cm'][i], 1e4 * e['sigma_v_cm'][i],
                                  '' if h == 'fit' else '(' + h + ')'))
                else:
                    j = min(fitted, key=lambda k: abs(k - i))
                    out.append('     |eta| %-8s as %.1f-%.1f' % (lab, edges[j], edges[j + 1]))
        return out

    def V(self, family, abs_eta):
        if family not in self.choice:
            die('no fitted hit errors for %s in %s'
                % (family, ', '.join(p for p, _ in self.files)))
        e = self.choice[family][2]
        edges = np.asarray(e['abs_eta_edges'])
        su = np.array([np.nan if x is None else x for x in e['sigma_u_cm']], float)
        sv = np.array([np.nan if x is None else x for x in e['sigma_v_cm']], float)
        b = np.clip(np.searchsorted(edges, abs_eta, side='right') - 1, 0, len(su) - 1)
        good = np.isfinite(su) & np.isfinite(sv)
        gi = np.nonzero(good)[0]
        near = np.array([gi[np.argmin(np.abs(gi - i))] for i in range(len(su))])
        b = near[b]
        V = np.zeros((len(abs_eta), 2, 2))
        V[:, 0, 0] = su[b] ** 2
        V[:, 1, 1] = sv[b] ** 2
        return V


INFLATE = (1.5, 2.0, 3.0, 5.0, 10.0, 30.0)


def route_b(cov_packed, v, killed_L1, killed_D1, hit_errors):
    """C' for the killed tracks: remove the L1 and/or D1 hit information.

    V is one number per |eta| bin; for a track whose own hit error is larger,
    V - H C H^T is not positive definite and the removal is undefined. Those
    tracks are retried with V scaled up by INFLATE in turn (flagged in
    `inflated`); only if all fail is the track marked not ok.
    Returns (packed C' (N,15), ok (N,), inflated (N,)) for all N tracks of v
    (unkilled tracks: C' = C)."""
    C = packed_to_matrix(cov_packed)
    Cp = C.copy()
    ok = np.ones(len(C), bool)
    inflated = np.zeros(len(C), bool)
    par = helix_to_curv(v['pt'], v['eta'], v['phi'], v['q'], v['z0'])
    for fam, killed in (('L1', killed_L1), ('D1', killed_D1)):
        idx = np.nonzero(killed)[0]
        if len(idx) == 0:
            continue
        if fam == 'L1':
            H = jacobian('L1', par[idx])
        else:
            H = np.full((len(idx), 2, 5), np.nan)
            pos = v['eta'][idx] >= 0
            for sd, m in (('+', pos), ('-', ~pos)):
                if m.any():
                    H[m] = jacobian('D1' + sd, par[idx][m])
        V = hit_errors.V(fam, np.abs(v['eta'][idx]))
        cp, okk = remove_hit(Cp[idx], H, V)
        infl = np.zeros(len(idx), bool)
        for f in INFLATE:
            bad = ~okk & np.all(np.isfinite(H), axis=(1, 2))
            if not bad.any():
                break
            cp2, ok2 = remove_hit(Cp[idx][bad], H[bad], V[bad] * f)
            sub = np.nonzero(bad)[0][ok2]
            cp[sub], okk[sub], infl[sub] = cp2[ok2], True, True
        Cp[idx] = np.where(okk[:, None, None], cp, np.nan)
        ok[idx] &= okk
        inflated[idx] |= infl
    return matrix_to_packed(Cp), ok, inflated


# ---------------------------------------------------------------------------
# route A: cross-context morph with the trained covflow flows
# ---------------------------------------------------------------------------

HIT_CONTEXT_SUFFIX = {'_n_pix_hit': 'n_pix', '_n_pix_b_hit': 'n_pix_b',
                      '_n_pix_e_hit': 'n_pix_e', '_pix_first_b_layer': 'first_b',
                      '_pix_first_e_disk': 'first_e'}
UNEMULATED_HIT_SUFFIX = ('_n_pix_layer', '_n_trk_layer', '_pix_first_layer',
                         '_n_pix_miss_inner', '_n_pix_inact_inner')


def _import_covflow():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(here) != 'covflow':
        die('the repository directory must be called "covflow" for the package '
            'import (it is %s)' % here)
    parent = os.path.dirname(here)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from covflow import flows as FL
    from covflow import data as D
    return FL, D


class CovFlow:
    """One trained covflow run (covflow.json + flows + scalers)."""

    def __init__(self, run_dir, mu, device='cpu'):
        man_path = os.path.join(run_dir, 'covflow.json')
        if not os.path.isfile(man_path):
            die('no covflow.json in %s' % run_dir)
        self.man = man = json.load(open(man_path))
        if man.get('schema') != 'covflow/1':
            die('%s: schema %r, expected covflow/1' % (man_path, man.get('schema')))
        FL, D = _import_covflow()
        self.FL = FL
        self.branches = list(man['context']['branches'])
        self.log_pt = man['context'].get('log_pt_branch')
        bad = [b for b in self.branches if not b.startswith(mu + '_') and '{mu}' not in b]
        if bad:
            die('%s was trained on context %s, not on %s' % (run_dir, self.branches, mu))
        for b in self.branches:
            if b.endswith(UNEMULATED_HIT_SUFFIX):
                die('context branch %s is a hit-pattern variable the emulation '
                    'does not update' % b)
        if man['cov_prefix'] != mu + '_cov_':
            die('%s corrects %s*, not %s_cov_*' % (run_dir, man['cov_prefix'], mu))
        self.idx = list(man['feature_indices'])
        self.param = man['param']
        f = man['flow']
        cfg = FL.FlowConfig(n_features=f['n_features'], n_context=f['n_context'],
                            transforms=f['transforms'], hidden=tuple(f['hidden']),
                            bins=f['bins'], seed=man.get('seed', 0))
        files = man['files']
        self.f_mc = FL.load_flow(cfg, os.path.join(run_dir, files['flow_mc']))
        self.f_data = FL.load_flow(cfg, os.path.join(run_dir, files['flow_data']))
        self.scaler = D.Standardiser.load(os.path.join(run_dir, files['scalers']))
        self.device = device
        self.run_dir = run_dir

    def needed_branches(self):
        return list(self.branches)

    def context(self, chunk, sel, emu=None, sub=None):
        """(N,k) context in training order, for the selected events `sel` of a
        chunk and, if given, the subset `sub` of them. emu (dict from emulate,
        over the same selected events) replaces the hit-pattern variables by
        their emulated values."""
        cols = []
        for b in self.branches:
            key = next((k for suf, k in HIT_CONTEXT_SUFFIX.items() if b.endswith(suf)), None)
            if emu is not None and key is not None:
                col = np.asarray(emu[key], np.float64)
            else:
                col = np.asarray(chunk[b], np.float64)[sel]
            if sub is not None:
                col = col[sub]
            if self.log_pt and b == self.log_pt:
                col = np.log(np.clip(col, 1e-6, None))
            cols.append(col)
        return np.stack(cols, axis=1)

    def morph(self, packed, C_from, C_to):
        """f_data^-1( f_MC(x; c_from); c_to ) on the feature subspace the flows
        were trained in; features outside it are copied unchanged."""
        to_feat, to_mat = F.get_transforms(self.param)
        y = F.packed_to_features(np.asarray(packed, np.float64), self.param)
        ys = self.scaler.x(y)
        z = self.FL.data_to_latent(self.f_mc, ys[:, self.idx], self.scaler.c(C_from),
                                   device=self.device)
        ys[:, self.idx] = self.FL.latent_to_data(self.f_data, z, self.scaler.c(C_to),
                                                 device=self.device)
        return F.matrix_to_packed(to_mat(self.scaler.x_inv(ys)))


# ---------------------------------------------------------------------------
# reweighted comparisons (shared by the two diagnostic scripts)
# ---------------------------------------------------------------------------

PT_RW = np.array([3., 4., 5., 6., 8., 10., 15., 30., 1e4])
ETA_RW = np.array([0., 0.5, 1.0, 1.5, 2.0, 2.6])
NPIX_RW = np.arange(-0.5, 9.5, 1.0)
# z0 * sign(eta): where along the beam the track starts, in its own direction.
# With |eta| and the hit count it fixes which layers and disks are crossed.
ZS_RW = np.array([-np.inf, -6., -3., 0., 3., 6., np.inf])


def zsigned(z0, eta):
    return z0 * np.where(eta >= 0, 1.0, -1.0)


def rw_cell(pt, eta, npix, zs):
    """Flat cell index in (pt, |eta|, npix, z0*sign(eta)); -1 outside."""
    ib = np.searchsorted(PT_RW, pt, 'right') - 1
    ie = np.searchsorted(ETA_RW, np.abs(eta), 'right') - 1
    ip = np.searchsorted(NPIX_RW, npix, 'right') - 1
    iz = np.searchsorted(ZS_RW, zs, 'right') - 1
    shape = (len(PT_RW) - 1, len(ETA_RW) - 1, len(NPIX_RW) - 1, len(ZS_RW) - 1)
    ok = (ib >= 0) & (ib < shape[0]) & (ie >= 0) & (ie < shape[1]) \
        & (ip >= 0) & (ip < shape[2]) & (iz >= 0) & (iz < shape[3])
    flat = np.ravel_multi_index((np.where(ok, ib, 0), np.where(ok, ie, 0),
                                 np.where(ok, ip, 0), np.where(ok, iz, 0)), shape)
    return np.where(ok, flat, -1), shape
LOGSIG_EDGES = {   # log10 of sigma, per parameter
    'qoverp': np.linspace(-4.5, -1.0, 71),
    'lambda': np.linspace(-5.0, -2.0, 61),
    'phi':    np.linspace(-5.0, -2.0, 61),
    'dxy':    np.linspace(-3.6, -1.0, 79),
    'dsz':    np.linspace(-3.6, -0.6, 76),
}


class RwHist:
    """Histograms of log10(sigma) per parameter in (pt, |eta|, n_pix,
    z0*sign(eta)) cells, so that any sample can be reweighted to any other
    at the end."""

    def __init__(self):
        _, self.shape = rw_cell(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
        nc = int(np.prod(self.shape))
        self.h = {p: np.zeros((nc, len(e) - 1)) for p, e in LOGSIG_EDGES.items()}
        self.n = 0.0

    def fill(self, pt, eta, npix, zs, M, w=None):
        if len(pt) == 0:
            return
        w = np.ones(len(pt)) if w is None else w
        sg = sigmas(M)
        cell, _ = rw_cell(pt, eta, npix, zs)
        ok = np.all(np.isfinite(sg), 1) & np.all(sg > 0, 1) & np.isfinite(w) & (cell >= 0)
        for j, p in enumerate(F.PARAM_NAMES):
            e = LOGSIG_EDGES[p]
            with np.errstate(divide='ignore', invalid='ignore'):
                ls = np.log10(sg[:, j])
            ik = np.searchsorted(e, ls, 'right') - 1
            okk = ok & (ik >= 0) & (ik < len(e) - 1)
            np.add.at(self.h[p], (cell[okk], ik[okk]), w[okk])
        self.n += float(w[ok].sum())

    def cells(self):
        return self.h['dxy'].sum(-1)

    def projected(self, p, target=None, eta_bin=None):
        """1D log10 sigma distribution, reweighted cell by cell to `target`
        (another RwHist) if given, normalised to unit area. eta_bin selects
        one |eta| bin of ETA_RW."""
        h = self.h[p]
        if target is not None:
            mine, theirs = self.cells(), target.cells()
            with np.errstate(divide='ignore', invalid='ignore'):
                f = np.where(mine > 0, theirs / mine, 0.0)
            h = h * f[:, None]
        if eta_bin is not None:
            ie = np.unravel_index(np.arange(h.shape[0]), self.shape)[1]
            h = h[ie == eta_bin]
        out = h.sum(0)
        s = out.sum()
        return out / s if s > 0 else out


def quantiles(hist, edges, qs=(0.16, 0.5, 0.84)):
    c = np.cumsum(hist)
    if c[-1] <= 0:
        return [np.nan] * len(qs)
    c = c / c[-1]
    xc = edges[1:]
    return [float(np.interp(q, c, xc)) for q in qs]


def ks_distance(h1, h2):
    if h1.sum() <= 0 or h2.sum() <= 0:
        return np.nan
    return float(np.max(np.abs(np.cumsum(h1) / h1.sum() - np.cumsum(h2) / h2.sum())))
