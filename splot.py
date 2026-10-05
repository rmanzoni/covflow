"""
covflow.splot
=============

J/psi -> mu mu mass fit and sPlot weights, written to be validated ON ITS OWN,
before anything is fed to the covflow training.

Where the model comes from
--------------------------
The model is the one in `compare_impact_parameters_splot.py` (the 2018
sigma_dxy quantile-morphing study), which fitted the charmonium sample
well: chi2/ndf = 210/66 with 5e5 signal candidates. Concretely:

  signal      two double-sided Crystal Balls with a COMMON mean and two core
              widths. The tail parameters (aL, nL, aR, nR) are SHARED between
              the two by default, as in the 2018 fit. Independent tails per
              component, or a single DSCB, are options.
  background  two exponentials; each slope may take either sign, so the sum
              can rise or fall on each side of the peak
  fit         extended maximum likelihood with iminuit (MIGRAD twice, then
              HESSE), strategy 2, tails FREE in data
  window      +- 250 MeV around the J/psi mass

Relative to the 2018 script, two things are re-parameterised, without changing
the family of shapes that can be described:

  s2 = rsig * s1 with rsig >= 1   -- the 2018 fit had s1 and s2 both free, so
                                    the two components could swap roles
                                    between fits. Forcing the second component
                                    to be the wider one removes that ambiguity.
                                    The functional family is identical.
  exp(-lam * (m - m_centre))      -- slopes are measured from the window
                                    centre, so the fit does not have to handle
                                    exp(+-100 * 3 GeV) numbers.

The fit also runs in two stages: first with the tails held at their starting
values, then with every parameter free. The final result is the second stage.

What sPlot needs, and what breaks it
------------------------------------
For each candidate the signal weight is

    w_s(m) = [V_ss f_s(m) + V_sb f_b(m)] / [N_s f_s(m) + N_b f_b(m)]

with V the inverse of  (V^-1)_ij = sum_e f_i(m_e) f_j(m_e) / D(m_e)^2, and the
sum running over EXACTLY the events the fit was performed on.

The background cancels only when the weighted events are summed over the WHOLE
fit window. Inside the signal region background events get positive signal
weights; the negative weights in the sidebands are what cancel them. So
applying the weights to a subset of the mass range -- e.g. fitting in +-250 MeV
and then training on the candidates within +-100 MeV -- does NOT subtract the
background. `SPlot.subrange_leakage` measures how much background survives in
that case.

Everything here is numpy + scipy + iminuit (+ matplotlib/mplhep for plots).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import numpy as np

JPSI_MASS = 3.0969

_TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ---------------------------------------------------------------------------
# status output (Ric: every long step must show it is alive)
# ---------------------------------------------------------------------------

class Progress:
    """Minimal progress bar on stderr, no dependency. Throttled to 2 Hz."""

    def __init__(self, total, label, unit="", every=0.5):
        self.total = max(int(total), 1)
        self.label, self.unit, self.every = label, unit, every
        self.n, self.t0, self.tlast = 0, time.time(), 0.0

    def update(self, k=1, extra=""):
        self.n += k
        now = time.time()
        if now - self.tlast < self.every and self.n < self.total:
            return
        self.tlast = now
        frac = min(self.n / self.total, 1.0)
        bar = "#" * int(30 * frac) + "." * (30 - int(30 * frac))
        el = now - self.t0
        eta = el / frac - el if frac > 0 else float("nan")
        sys.stderr.write(f"\r{self.label} [{bar}] {100*frac:5.1f}% "
                         f"{self.n:,}/{self.total:,}{self.unit} "
                         f"{el:6.0f}s elapsed, ~{eta:5.0f}s left {extra}   ")
        sys.stderr.flush()

    def close(self):
        self.tlast = 0.0
        self.n = max(self.n, 0)
        sys.stderr.write("\n")
        sys.stderr.flush()


class _Heartbeat:
    """The minimiser has no known length: report calls, NLL and time instead."""

    def __init__(self, label, every=2.0, enabled=True):
        self.label, self.every, self.enabled = label, every, enabled
        self.calls, self.t0, self.tlast, self.best = 0, time.time(), 0.0, np.inf

    def __call__(self, value):
        self.calls += 1
        self.best = min(self.best, value)
        if not self.enabled:
            return
        now = time.time()
        if now - self.tlast >= self.every:
            self.tlast = now
            sys.stderr.write(f"\r[fit] {self.label}: {self.calls:,} NLL calls, "
                             f"best NLL {self.best:.3f}, "
                             f"{now - self.t0:.0f}s   ")
            sys.stderr.flush()

    def done(self):
        if self.enabled and self.calls:
            sys.stderr.write(f"\r[fit] {self.label}: {self.calls:,} NLL calls, "
                             f"best NLL {self.best:.3f}, "
                             f"{time.time() - self.t0:.0f}s   \n")
            sys.stderr.flush()


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------

def dscb_unnorm(x, mu, s, aL, nL, aR, nR):
    """
    Double-sided Crystal Ball, unnormalised: Gaussian core, power-law tails
    below -aL and above +aR (in units of s), continuous in value and slope.

    Tails computed in log space so that large n with small alpha cannot
    overflow ((n/a)^n reaches 1e148 at the edge of the allowed range).
    """
    t = (np.asarray(x, float) - mu) / s
    out = np.exp(-0.5 * t * t)
    lo = t < -aL
    if np.any(lo):
        logA = nL * np.log(nL / aL) - 0.5 * aL * aL
        B = nL / aL - aL
        out[lo] = np.exp(logA - nL * np.log(B - t[lo]))
    hi = t > aR
    if np.any(hi):
        logA = nR * np.log(nR / aR) - 0.5 * aR * aR
        B = nR / aR - aR
        out[hi] = np.exp(logA - nR * np.log(B + t[hi]))
    return out


def dscb_primitive(x, mu, s, aL, nL, aR, nR):
    """
    Antiderivative of dscb_unnorm with respect to x, zero at x -> -inf
    (requires nL, nR > 1). Analytic, so the normalisation and the bin
    integrals are exact and SMOOTH in the parameters. A grid-based integral
    is not: as the tail junction mu - aL*s moves across grid nodes it adds
    small non-smooth steps to the NLL, which is harmless for the minimum but
    ruins HESSE's numerical second derivatives.
    """
    t = (np.asarray(x, float) - mu) / s
    r2 = np.sqrt(2.0)
    from scipy.special import erf
    BL, BR = nL / aL - aL, nR / aR - aR
    logAL = nL * np.log(nL / aL) - 0.5 * aL * aL
    logAR = nR * np.log(nR / aR) - 0.5 * aR * aR
    # left tail: A (B - t)^(1-n) / (n-1); at t = -aL, B - t = n/a
    G_at_mL = np.exp(logAL + (1 - nL) * np.log(nL / aL)) / (nL - 1)
    core_0 = np.sqrt(np.pi / 2) * erf(-aL / r2)
    G_at_R = G_at_mL + np.sqrt(np.pi / 2) * erf(aR / r2) - core_0
    out = np.empty_like(t)
    lo, hi = t <= -aL, t > aR
    mid = ~(lo | hi)
    out[lo] = np.exp(logAL + (1 - nL) * np.log(BL - t[lo])) / (nL - 1)
    out[mid] = G_at_mL + np.sqrt(np.pi / 2) * erf(t[mid] / r2) - core_0
    out[hi] = G_at_R + (np.exp(logAR + (1 - nR) * np.log(nR / aR))
                        - np.exp(logAR + (1 - nR) * np.log(BR + t[hi]))) / (nR - 1)
    return s * out


def _exp_norm(lam, ulo, uhi):
    """Integral of exp(-lam*u) over [ulo, uhi], stable at lam -> 0.
    `uhi` may be an array."""
    if abs(lam) < 1e-9:
        return uhi - ulo
    return np.exp(-lam * ulo) * (-np.expm1(-lam * (uhi - ulo))) / lam


SIGNAL_MODELS = ("2dscb", "dscb")
TAIL_MODES = ("shared", "independent")
BACKGROUND_MODELS = ("2exp", "exp")


class MassModel:
    """
    Parameter bookkeeping + normalised shapes on [lo, hi].

    Parameters (all in GeV where dimensionful):
      Ns, Nb                      yields in the window
      mu, s1                      common mean, width of the narrow core
      rsig, fcore                 (2dscb) s2 = rsig*s1, fraction of the narrow
      aL, nL, aR, nR              tails (shared); with independent tails these
                                  become aL1..nR1 and aL2..nR2
      lam1, lam2, fb              background slopes [1/GeV] and fraction of
                                  the first; the shape is exp(-lam*(m-centre))
    """

    def __init__(self, lo, hi, signal="2dscb", tails="shared",
                 background="2exp", ngrid=4001):
        if signal not in SIGNAL_MODELS:
            raise ValueError(f"signal must be one of {SIGNAL_MODELS}")
        if tails not in TAIL_MODES:
            raise ValueError(f"tails must be one of {TAIL_MODES}")
        if background not in BACKGROUND_MODELS:
            raise ValueError(f"background must be one of {BACKGROUND_MODELS}")
        if signal == "dscb" and tails == "independent":
            raise ValueError("independent tails need two components (2dscb)")
        self.lo, self.hi = float(lo), float(hi)
        self.centre = 0.5 * (self.lo + self.hi)
        self.signal, self.tails, self.background = signal, tails, background
        self.grid = np.linspace(self.lo, self.hi, ngrid)

        n = ["Ns", "Nb", "mu", "s1"]
        if signal == "2dscb":
            n += ["rsig", "fcore"]
        if tails == "shared":
            self.tail_names = ["aL", "nL", "aR", "nR"]
        else:
            self.tail_names = ["aL1", "nL1", "aR1", "nR1",
                               "aL2", "nL2", "aR2", "nR2"]
        n += self.tail_names
        n += ["lam1", "lam2", "fb"] if background == "2exp" else ["lam1"]
        self.names = n

    # -- description, for reports -----------------------------------------
    def describe(self):
        return {"window": [self.lo, self.hi], "signal": self.signal,
                "tails": self.tails, "background": self.background,
                "parameters": list(self.names)}

    # -- limits, taken from the 2018 fit where it had them -----------------
    def bounds(self, n_events):
        b = dict(Ns=(0.0, 3.0 * n_events), Nb=(0.0, 3.0 * n_events),
                 mu=(self.lo, self.hi), s1=(2e-3, 0.1),
                 rsig=(1.0, 10.0), fcore=(0.0, 1.0),
                 lam1=(-100.0, 100.0), lam2=(-100.0, 100.0), fb=(0.0, 1.0))
        for k in self.tail_names:
            b[k] = (0.2, 10.0) if k.startswith("a") else (1.01, 60.0)
        return {k: b[k] for k in self.names}

    def start(self, mass=None, counts=None, edges=None):
        """Data-driven yields/mean/width; everything else as in the 2018 fit."""
        if mass is not None:
            m = np.asarray(mass, float)
            cnt, e = np.histogram(m, bins=100, range=(self.lo, self.hi))
        else:
            cnt, e = np.asarray(counts, float), np.asarray(edges, float)
        c = 0.5 * (e[1:] + e[:-1])
        n = float(cnt.sum())
        mu0 = float(c[np.argmax(cnt)])
        # background level from the outer 20% on each side, extrapolated flat
        edge = 0.2 * (self.hi - self.lo)
        side = (c < self.lo + edge) | (c > self.hi - edge)
        nb0 = float(cnt[side].sum()) * (len(c) / max(side.sum(), 1))
        nb0 = min(max(nb0, 0.02 * n), 0.9 * n)
        # core width from the half-maximum of the background-subtracted peak
        peak = cnt - nb0 / len(c)
        above = c[peak > 0.5 * peak.max()]
        fwhm = (above.max() - above.min()) if above.size > 1 else 0.06
        s0 = float(np.clip(fwhm / 2.355, 0.005, 0.08))
        p = dict(Ns=n - nb0, Nb=nb0, mu=mu0, s1=0.8 * s0, rsig=2.0, fcore=0.6,
                 lam1=1.0, lam2=-1.0, fb=0.5)
        for k in self.tail_names:
            p[k] = 1.5 if k.startswith("a") else 3.0
        return {k: p[k] for k in self.names}

    # -- shapes ------------------------------------------------------------
    def _tails(self, p, comp):
        if self.tails == "shared":
            return p["aL"], p["nL"], p["aR"], p["nR"]
        s = str(comp)
        return p["aL" + s], p["nL" + s], p["aR" + s], p["nR" + s]

    def _dscb_norm(self, mu, s, tails):
        G = dscb_primitive(np.array([self.lo, self.hi]), mu, s, *tails)
        return G[1] - G[0]

    def _ndscb(self, x, mu, s, tails):
        return dscb_unnorm(x, mu, s, *tails) / self._dscb_norm(mu, s, tails)

    def _signal_cdf_parts(self, edges, p):
        """Normalised signal integral up to each edge, per component."""
        if self.signal == "dscb":
            comps = [(1.0, p["s1"], self._tails(p, 1))]
        else:
            comps = [(p["fcore"], p["s1"], self._tails(p, 1)),
                     (1 - p["fcore"], p["s1"] * p["rsig"], self._tails(p, 2))]
        out = 0.0
        for f, s_, t_ in comps:
            G = dscb_primitive(edges, p["mu"], s_, *t_)
            G0 = dscb_primitive(np.array([self.lo]), p["mu"], s_, *t_)[0]
            out = out + f * (G - G0) / self._dscb_norm(p["mu"], s_, t_)
        return out

    def _background_cdf(self, edges, p):
        u = np.asarray(edges, float) - self.centre
        ulo, uhi = self.lo - self.centre, self.hi - self.centre

        def ce(lam):
            return _exp_norm(lam, ulo, u) / _exp_norm(lam, ulo, uhi)

        if self.background == "exp":
            return ce(p["lam1"])
        return p["fb"] * ce(p["lam1"]) + (1 - p["fb"]) * ce(p["lam2"])

    def signal_components(self, x, p):
        """[(label, fraction * normalised pdf), ...] -- they sum to f_s."""
        if self.signal == "dscb":
            return [("DSCB", self._ndscb(x, p["mu"], p["s1"], self._tails(p, 1)))]
        s2 = p["s1"] * p["rsig"]
        return [
            (f"narrow DSCB, $\\sigma$={1e3*p['s1']:.1f} MeV",
             p["fcore"] * self._ndscb(x, p["mu"], p["s1"], self._tails(p, 1))),
            (f"wide DSCB, $\\sigma$={1e3*s2:.1f} MeV",
             (1 - p["fcore"]) * self._ndscb(x, p["mu"], s2, self._tails(p, 2))),
        ]

    def signal_pdf(self, x, p):
        return sum(c for _, c in self.signal_components(x, p))

    def background_pdf(self, x, p):
        u = np.asarray(x, float) - self.centre
        ulo, uhi = self.lo - self.centre, self.hi - self.centre

        def ne(lam):
            return np.exp(-lam * u) / _exp_norm(lam, ulo, uhi)

        if self.background == "exp":
            return ne(p["lam1"])
        return p["fb"] * ne(p["lam1"]) + (1 - p["fb"]) * ne(p["lam2"])

    def density(self, x, p):
        return p["Ns"] * self.signal_pdf(x, p) + p["Nb"] * self.background_pdf(x, p)

    def bin_expect(self, edges, p):
        """Expected counts per bin: exact integrals of the density."""
        e = np.asarray(edges, float)
        return (p["Ns"] * np.diff(self._signal_cdf_parts(e, p))
                + p["Nb"] * np.diff(self._background_cdf(e, p)))

    def sigma_eff(self, p):
        if self.signal == "dscb":
            return p["s1"]
        s2 = p["s1"] * p["rsig"]
        return float(np.sqrt(p["fcore"] * p["s1"] ** 2 + (1 - p["fcore"]) * s2 ** 2))


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    model: MassModel
    mode: str                       # "unbinned" | "binned"
    n_events: int
    values: dict
    errors: dict
    covariance: np.ndarray | None
    free: list
    fixed: dict
    valid: bool
    accurate_covariance: bool
    edm: float
    nfcn: int
    fmin: float
    at_limit: list
    stages: list = field(default_factory=list)

    @property
    def p(self):
        return self.values

    @property
    def error_reliable(self):
        """HESSE errors mean something only with an accurate covariance and
        no fraction pinned at 0/1 (a pinned fraction leaves the other
        component's shape parameters undetermined, and HESSE then reports
        nonsense such as +-117 on 4.3M signal events -- Run 3, 2 exp)."""
        pinned = [k for k in self.at_limit if k in ("fb", "fcore")]
        return self.accurate_covariance and not pinned

    def top_correlations(self, k=6):
        """Largest |correlations| between free parameters. A near-degenerate
        pair (|rho| > 0.95) is the usual reason for an invalid MIGRAD or a
        meaningless error, and names which parts of the model trade off."""
        if self.covariance is None:
            return []
        d = np.sqrt(np.clip(np.diag(self.covariance), 1e-300, None))
        rho = self.covariance / np.outer(d, d)
        pairs = [(self.free[i], self.free[j], float(rho[i, j]))
                 for i in range(len(self.free)) for j in range(i + 1, len(self.free))]
        pairs.sort(key=lambda t: -abs(t[2]))
        return pairs[:k]

    def summary(self):
        p = self.values
        out = {"mode": self.mode, "n_events_in_window": self.n_events,
               "model": self.model.describe(),
               "values": {k: float(v) for k, v in p.items()},
               "errors": {k: float(v) for k, v in self.errors.items()},
               "fixed": {k: float(v) for k, v in self.fixed.items()},
               "migrad_valid": self.valid,
               "accurate_covariance": self.accurate_covariance,
               "edm": self.edm, "nfcn": self.nfcn, "fmin": self.fmin,
               "parameters_at_limit": self.at_limit,
               "errors_reliable": self.error_reliable,
               "top_correlations": self.top_correlations(),
               "sigma_eff_MeV": 1e3 * self.model.sigma_eff(p),
               "purity_in_window": p["Ns"] / max(p["Ns"] + p["Nb"], 1e-9),
               "stages": self.stages}
        if self.covariance is not None:
            out["covariance_parameters"] = list(self.free)
            out["covariance"] = self.covariance.tolist()
        return out


def _import_minuit():
    try:
        from iminuit import Minuit
    except ImportError as e:
        raise SystemExit(
            "iminuit is required (the 2018 fit that worked used it; scipy's "
            "L-BFGS-B/Nelder-Mead give no covariance and no validity flag).\n"
            "  pip install iminuit        # inside the conda env you run in"
        ) from e
    return Minuit


def fit(model, mass=None, counts=None, edges=None, start=None, fixed=None,
        strategy=2, staged=True, verbose=True, label="data", heartbeat=True,
        max_passes=5, ncall=200_000):
    """
    Extended maximum-likelihood fit.

    Unbinned if `mass` is given; binned (Poisson, bin-integrated model) if
    `counts`+`edges` are given. With 1 MeV bins against a ~20-30 MeV
    resolution the binned fit loses nothing measurable and its cost does not
    grow with the number of candidates.

    `fixed`: dict name -> value held constant throughout.
    """
    Minuit = _import_minuit()
    fixed = dict(fixed or {})
    unknown = set(fixed) - set(model.names)
    if unknown:
        raise ValueError(f"cannot fix {sorted(unknown)}: not parameters of "
                         f"this model {model.names}")

    if mass is not None:
        m = np.asarray(mass, float)
        m = m[(m >= model.lo) & (m <= model.hi)]
        n = m.size
        mode = "unbinned"
    else:
        counts = np.asarray(counts, float)
        edges = np.asarray(edges, float)
        if not np.allclose(np.diff(edges), edges[1] - edges[0], rtol=1e-6):
            raise ValueError("binned fit needs uniform bin edges")
        n = int(round(counts.sum()))
        mode = "binned"
    if n < 500:
        raise ValueError(f"only {n} candidates in [{model.lo:.4f}, "
                         f"{model.hi:.4f}] -- too few for this fit")

    names = model.names
    p0 = (dict(start) if start is not None
          else model.start(mass=m if mode == "unbinned" else None,
                           counts=counts, edges=edges))
    p0.update(fixed)
    bounds = model.bounds(n)
    hb = _Heartbeat(label, enabled=heartbeat and verbose)

    def unpack(v):
        return dict(zip(names, v))

    if mode == "unbinned":
        def nll(v):
            p = unpack(v)
            d = model.density(m, p)
            if not np.all(np.isfinite(d)) or np.any(d <= 0):
                val = 1e30
            else:
                val = float(p["Ns"] + p["Nb"] - np.sum(np.log(d)))
            hb(val)
            return val
    else:
        nz = counts > 0
        # subtract the saturated-model constant so the NLL is O(nbins)
        const = float(np.sum(counts[nz] - counts[nz] * np.log(counts[nz])))

        def nll(v):
            p = unpack(v)
            mu = model.bin_expect(edges, p)
            if not np.all(np.isfinite(mu)) or np.any(mu <= 0):
                val = 1e30
            else:
                val = float(np.sum(mu) - np.sum(counts[nz] * np.log(mu[nz])) - const)
            hb(val)
            return val

    mi = Minuit(nll, np.array([p0[k] for k in names], float), name=names)
    mi.errordef = Minuit.LIKELIHOOD
    mi.strategy = strategy
    mi.print_level = 0
    for k in names:
        mi.limits[k] = bounds[k]
        # initial step: 10% of the value or of the range
        lo_b, hi_b = bounds[k]
        mi.errors[k] = max(0.1 * abs(p0[k]), 1e-3 * (hi_b - lo_b))
        mi.fixed[k] = k in fixed

    stages = []
    tail_like = [k for k in model.tail_names if k not in fixed]
    if staged and tail_like:
        for k in tail_like:
            mi.fixed[k] = True
        mi.migrad(ncall=ncall)
        stages.append({"stage": "tails held", "fval": float(mi.fval),
                       "valid": bool(mi.valid), "nfcn": int(mi.nfcn)})
        for k in tail_like:
            mi.fixed[k] = False
    # The 2018 fit ran MIGRAD twice. The narrow/wide core pair (s1, rsig,
    # fcore) is strongly correlated and converges slowly, so the default call
    # limit can end a pass before the EDM target: repeat until valid, with a
    # generous call budget, up to max_passes.
    for k in range(max_passes):
        mi.migrad(ncall=ncall)
        stages.append({"stage": f"all free, pass {k+1}", "fval": float(mi.fval),
                       "valid": bool(mi.valid), "edm": float(mi.fmin.edm),
                       "call_limit": bool(mi.fmin.has_reached_call_limit),
                       "nfcn": int(mi.nfcn)})
        if mi.valid and k >= 1:
            break
    mi.hesse()
    stages.append({"stage": "hesse", "valid": bool(mi.valid),
                   "accurate_covariance": bool(mi.fmin.has_accurate_covar),
                   "made_posdef": bool(mi.fmin.has_made_posdef_covar)})
    hb.done()

    free = [k for k in names if k not in fixed]
    vals = {k: float(mi.values[k]) for k in names}
    errs = {k: float(mi.errors[k]) if k not in fixed else 0.0 for k in names}
    cov = None
    if mi.covariance is not None:
        full = np.array(mi.covariance)
        idx = [names.index(k) for k in free]
        cov = full[np.ix_(idx, idx)]

    at_limit = []
    for k in free:
        lo_b, hi_b = bounds[k]
        tol = max(1e-4 * (hi_b - lo_b), 0.01 * errs[k])
        if abs(vals[k] - lo_b) < tol or abs(vals[k] - hi_b) < tol:
            at_limit.append(k)

    res = FitResult(model=model, mode=mode, n_events=n, values=vals,
                    errors=errs, covariance=cov, free=free, fixed=fixed,
                    valid=bool(mi.valid),
                    accurate_covariance=bool(mi.fmin.has_accurate_covar),
                    edm=float(mi.fmin.edm), nfcn=int(mi.nfcn),
                    fmin=float(mi.fval), at_limit=at_limit, stages=stages)
    if verbose:
        print_fit(res)
    return res


def print_fit(res):
    p, e = res.values, res.errors
    status = "VALID" if res.valid else "INVALID"
    print(f"[fit] {res.mode} fit of {res.n_events:,} candidates in "
          f"[{res.model.lo:.4f}, {res.model.hi:.4f}] GeV: MIGRAD {status}, "
          f"covariance {'accurate' if res.accurate_covariance else 'NOT accurate'}, "
          f"EDM {res.edm:.2e}, {res.nfcn} calls")
    for k in res.model.names:
        tag = "  (fixed)" if k in res.fixed else ""
        tag += "  <-- AT LIMIT" if k in res.at_limit else ""
        print(f"[fit]   {k:>6s} = {p[k]:14.6g} +- {e[k]:10.3g}{tag}")
    print(f"[fit]   purity in window {100*p['Ns']/(p['Ns']+p['Nb']):.1f}%, "
          f"S/B {p['Ns']/max(p['Nb'],1e-9):.2f}, "
          f"effective signal width {1e3*res.model.sigma_eff(p):.1f} MeV")
    print("[fit]   largest correlations: " + ", ".join(
        f"{a}-{b} {r:+.2f}" for a, b, r in res.top_correlations(4)))
    if not res.valid:
        print("[fit]   WARNING: MIGRAD did not converge. If a correlation "
              "above is close to +-1 the model is (nearly) degenerate: the "
              "data cannot tell those parameters apart, and the yields are "
              "not well defined. Run --toys to measure the consequence.")
    frac = [k for k in res.at_limit if k in ("fcore", "fb", "rsig")]
    # a power-law exponent n at its UPPER bound: the tail falls off as fast
    # as allowed, i.e. it is indistinguishable from the Gaussian core there
    # (seen on Run 3 data: nR -> 60). Harmless, unlike n at 1 or alpha at a bound.
    n_hi = [k for k in res.at_limit if k.startswith("n")
            and abs(res.values[k] - res.model.bounds(res.n_events)[k][1]) < 1e-3]
    shape = [k for k in res.at_limit if k not in frac and k not in n_hi]
    if n_hi:
        print(f"[fit]   NOTE: {n_hi} at the upper limit: that tail is "
              f"Gaussian-like, no power-law needed. Harmless.")
    if frac:
        print(f"[fit]   NOTE: {frac} at their limit: one of the two "
              f"components is not needed (fraction 0/1, or both widths equal). "
              f"Harmless for the curve, but that component's other parameters "
              f"are then undetermined -- consider the simpler model.")
    if shape:
        print(f"[fit]   WARNING: {shape} at their limit: the fit wanted to go "
              f"further; the shape cannot describe the data within its limits")


def goodness_of_fit(res, mass=None, counts=None, edges=None, nbins=100,
                    min_expected=5.0):
    """
    Pearson chi2 on a plotting binning, with sparse bins (expected < 5)
    merged into their neighbours so the chi2 stays meaningful in the tails.
    """
    model, p = res.model, res.values
    if mass is not None:
        edges = np.linspace(model.lo, model.hi, nbins + 1)
        counts, _ = np.histogram(mass, bins=edges)
    counts = np.asarray(counts, float)
    exp = model.bin_expect(np.asarray(edges, float), p)
    # merge low-expectation bins left to right
    oc, ec, acc_o, acc_e = [], [], 0.0, 0.0
    for o, x in zip(counts, exp):
        acc_o += o
        acc_e += x
        if acc_e >= min_expected:
            oc.append(acc_o); ec.append(acc_e); acc_o = acc_e = 0.0
    if acc_e > 0 and ec:
        oc[-1] += acc_o; ec[-1] += acc_e
    oc, ec = np.array(oc), np.array(ec)
    pulls = (oc - ec) / np.sqrt(ec)
    chi2 = float(np.sum(pulls ** 2))
    ndf = int(len(oc) - len(res.free))
    from scipy.stats import chi2 as _c2
    return {"chi2": chi2, "ndf": ndf, "chi2_ndf": chi2 / max(ndf, 1),
            "p_value": float(_c2.sf(chi2, max(ndf, 1))),
            "pull_mean": float(pulls.mean()), "pull_rms": float(pulls.std()),
            "n_bins_used": int(len(oc))}


# ---------------------------------------------------------------------------
# sPlot
# ---------------------------------------------------------------------------

class SPlot:
    """
    sWeights from a fitted model.

    `mass` must be EXACTLY the sample that was fitted (all candidates in the
    window). V is computed from it and stored; `weights()` then evaluates the
    weights for any array of masses using that V. Weights of events outside
    the window are zero.
    """

    def __init__(self, res, mass):
        self.res = res
        model, p = res.model, res.values
        m = np.asarray(mass, float)
        inside = (m >= model.lo) & (m <= model.hi)
        mi = m[inside]
        fs, fb = model.signal_pdf(mi, p), model.background_pdf(mi, p)
        D = p["Ns"] * fs + p["Nb"] * fb
        Vinv = np.array([[np.sum(fs * fs / D ** 2), np.sum(fs * fb / D ** 2)],
                         [np.sum(fb * fs / D ** 2), np.sum(fb * fb / D ** 2)]])
        self.V = np.linalg.inv(Vinv)
        self.n_fit_sample = int(inside.sum())
        if abs(self.n_fit_sample - res.n_events) > 0:
            print(f"[splot] WARNING: V computed on {self.n_fit_sample:,} "
                  f"candidates but the fit used {res.n_events:,} -- the "
                  f"weights are only valid on the fitted sample")
        sw_s, sw_b = self._eval(mi)
        Ns, Nb = p["Ns"], p["Nb"]
        # At the maximum of the extended likelihood the yield derivatives
        # vanish:  sum_e f_s/D = sum_e f_b/D = 1.  This is what makes the
        # weights sum to 1 per event, and it is the check that the fit really
        # sits at a maximum. (sum_e w_s = N_s, by contrast, is an algebraic
        # identity of the formula -- it holds even for a wrong fit.)
        u_s, u_b = float(np.sum(fs / D)), float(np.sum(fb / D))
        self.info = {
            "stationarity_sum_fs_over_D": u_s,
            "stationarity_sum_fb_over_D": u_b,
            "n_fit_sample": self.n_fit_sample,
            "V": self.V.tolist(),
            "sqrt_Vss_vs_Ns_error": [float(np.sqrt(self.V[0, 0])),
                                     float(res.errors.get("Ns", np.nan))],
            "sum_signal_weights": float(sw_s.sum()),
            "sum_background_weights": float(sw_b.sum()),
            "fitted_Ns": float(Ns), "fitted_Nb": float(Nb),
            # consequence of stationarity; a binned fit satisfies it to its
            # (small) binning precision, an unconverged fit visibly does not
            "max_abs_sw_s_plus_sw_b_minus_1": float(np.max(np.abs(sw_s + sw_b - 1.0))),
            "negative_signal_weight_fraction": float((sw_s < 0).mean()),
            "min_signal_weight": float(sw_s.min()),
            "max_signal_weight": float(sw_s.max()),
            "effective_n_signal": float(sw_s.sum() ** 2 / np.sum(sw_s ** 2)),
        }

    def _eval(self, m):
        model, p = self.res.model, self.res.values
        fs, fb = model.signal_pdf(m, p), model.background_pdf(m, p)
        D = np.clip(p["Ns"] * fs + p["Nb"] * fb, 1e-300, None)
        V = self.V
        return (V[0, 0] * fs + V[0, 1] * fb) / D, (V[1, 0] * fs + V[1, 1] * fb) / D

    def weights(self, mass):
        m = np.asarray(mass, float)
        model = self.res.model
        inside = (m >= model.lo) & (m <= model.hi)
        sw_s, sw_b = np.zeros(m.size), np.zeros(m.size)
        a, b = self._eval(m[inside])
        sw_s[inside], sw_b[inside] = a, b
        return sw_s, sw_b

    def subrange_leakage(self, sub_lo, sub_hi, n=20001):
        """
        What happens if the weights are applied only to candidates in
        [sub_lo, sub_hi] (e.g. training with a narrower mass cut than the fit).

        Returns the weighted signal and background content of that subrange:
          S_w = N_s * integral f_s w_s,  B_w = N_b * integral f_b w_s
        Over the full window B_w = 0 exactly; in a subrange it is not, and
        B_w / (S_w + B_w) is the background fraction that survives.
        """
        model, p = self.res.model, self.res.values
        x = np.linspace(max(sub_lo, model.lo), min(sub_hi, model.hi), n)
        ws, _ = self._eval(x)
        S = p["Ns"] * _TRAPZ(model.signal_pdf(x, p) * ws, x)
        B = p["Nb"] * _TRAPZ(model.background_pdf(x, p) * ws, x)
        xf = np.linspace(model.lo, model.hi, n)
        wf, _ = self._eval(xf)
        Bfull = p["Nb"] * _TRAPZ(model.background_pdf(xf, p) * wf, xf)
        return {"subrange": [float(sub_lo), float(sub_hi)],
                "weighted_signal": float(S), "weighted_background": float(B),
                "background_fraction": float(B / (S + B)) if S + B != 0 else np.nan,
                "signal_fraction_of_Ns": float(S / p["Ns"]),
                "full_window_weighted_background": float(Bfull)}


# ---------------------------------------------------------------------------
# regions for the control-variable checks
# ---------------------------------------------------------------------------

def regions(res, sr_nsigma=2.5, sb_nsigma=5.0):
    """Signal region |m-mu| < sr*sigma_eff; sidebands |m-mu| > sb*sigma_eff."""
    p, model = res.values, res.model
    se = model.sigma_eff(p)
    sr = (p["mu"] - sr_nsigma * se, p["mu"] + sr_nsigma * se)
    sbl = (model.lo, p["mu"] - sb_nsigma * se)
    sbh = (p["mu"] + sb_nsigma * se, model.hi)
    if sbl[1] <= sbl[0] or sbh[0] >= sbh[1]:
        raise ValueError(f"window too narrow for sidebands at {sb_nsigma} "
                         f"sigma_eff ({1e3*se:.1f} MeV)")

    def integ(fn, a, b):
        x = np.linspace(a, b, 4001)
        return float(_TRAPZ(fn(x, p), x))

    out = {"sigma_eff": se, "SR": sr, "SB_low": sbl, "SB_high": sbh}
    for nm, (a, b) in (("SR", sr), ("SB_low", sbl), ("SB_high", sbh)):
        out[f"S_{nm}"] = p["Ns"] * integ(model.signal_pdf, a, b)
        out[f"B_{nm}"] = p["Nb"] * integ(model.background_pdf, a, b)
    return out


def _chi2_shapes(h1, e1, h2, e2):
    """Chi2 between two unit-normalised histograms with their errors."""
    ok = (e1 ** 2 + e2 ** 2) > 0
    c = float(np.sum((h1[ok] - h2[ok]) ** 2 / (e1[ok] ** 2 + e2[ok] ** 2)))
    return c, int(ok.sum() - 1)


def _norm(h, e):
    s = h.sum()
    return (h / s, e / abs(s)) if s != 0 else (h, e)


def control_check(mass, values, sp, reg, edges):
    """
    Three background-subtracted views of one control variable, all as
    unit-normalised shapes:

      splot     sum of signal sWeights over the whole window
      sideband  (SR) - r * (SB_low + SB_high), with r = B_SR / B_SB from the fit
      sb_lo/hi  the two sidebands separately, each with its expected signal
                content removed (fitted S in that sideband x sPlot signal
                shape), to test whether the BACKGROUND shape of the variable
                depends on the mass -- the assumption both subtractions rely on

    sPlot and sideband subtraction make different assumptions and share only
    the SR events, so their agreement is a real cross-check (the chi2 between
    them is indicative only, the samples are correlated). The low/high
    sideband comparison uses disjoint samples, so its chi2 is a proper test.
    """
    m = np.asarray(mass, float)
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    m, v = m[ok], v[ok]
    sw_s, _ = sp.weights(m)

    def h(sel, w=None):
        c, _ = np.histogram(v[sel], bins=edges, weights=None if w is None else w[sel])
        e2, _ = np.histogram(v[sel], bins=edges,
                             weights=None if w is None else w[sel] ** 2)
        return c.astype(float), np.sqrt(e2 if w is not None else c)

    allw = np.ones(m.size, bool)
    in_sr = (m > reg["SR"][0]) & (m < reg["SR"][1])
    in_lo = (m >= reg["SB_low"][0]) & (m < reg["SB_low"][1])
    in_hi = (m > reg["SB_high"][0]) & (m <= reg["SB_high"][1])

    hs, es = h(allw, sw_s)
    hsr, esr = h(in_sr)
    hlo, elo = h(in_lo)
    hhi, ehi = h(in_hi)
    r = reg["B_SR"] / (reg["B_SB_low"] + reg["B_SB_high"])
    hsb, esb = hsr - r * (hlo + hhi), np.sqrt(esr ** 2 + r ** 2 * (elo ** 2 + ehi ** 2))

    # Low vs high sideband uses ITS OWN binning: equal-population bins of the
    # sideband events. On the signal-region binning the background populates
    # the edge bins with a handful of entries each and the Gaussian chi2 runs
    # high even for identical shapes (measured: 54/39 for a background that is
    # mass-independent by construction).
    in_sb = in_lo | in_hi
    nsb = int(np.clip(in_sb.sum() // 400, 5, 25))
    sb_edges = np.unique(np.percentile(v[in_sb], np.linspace(0, 100, nsb + 1)))
    sb_edges[0] -= 1e-9 * max(1.0, abs(sb_edges[0]))
    sb_edges[-1] += 1e-9 * max(1.0, abs(sb_edges[-1]))

    def hb_(sel, w=None, e_=sb_edges):
        c, _ = np.histogram(v[sel], bins=e_, weights=None if w is None else w[sel])
        e2, _ = np.histogram(v[sel], bins=e_, weights=None if w is None else w[sel] ** 2)
        return c.astype(float), np.sqrt(e2 if w is not None else c)
    hlo2, elo2 = hb_(in_lo)
    hhi2, ehi2 = hb_(in_hi)
    # The sidebands are not pure background: the radiative tail reaches the
    # low one in particular. Remove the signal each is expected to contain,
    # using the fitted signal yield there and the sPlot signal shape.
    gs, egs = _norm(*hb_(allw, sw_s))
    blo = hlo2 - reg["S_SB_low"] * gs
    bhi = hhi2 - reg["S_SB_high"] * gs
    eblo = np.sqrt(elo2 ** 2 + (reg["S_SB_low"] * egs) ** 2)
    ebhi = np.sqrt(ehi2 ** 2 + (reg["S_SB_high"] * egs) ** 2)
    out = {"splot": _norm(hs, es), "sideband": _norm(hsb, esb),
           "sb_edges": sb_edges,
           "sb_low": _norm(blo, eblo), "sb_high": _norm(bhi, ebhi),
           "sb_low_raw": _norm(hlo2, elo2), "sb_high_raw": _norm(hhi2, ehi2),
           "signal_fraction_SB_low": float(reg["S_SB_low"] / max(hlo.sum(), 1)),
           "signal_fraction_SB_high": float(reg["S_SB_high"] / max(hhi.sum(), 1)),
           "r_bkg_SR_over_SB": float(r),
           "clipped_fraction": float(np.mean((v < edges[0]) | (v > edges[-1])))}
    c, nd = _chi2_shapes(*out["splot"], *out["sideband"])
    out["chi2_splot_vs_sideband"] = (c, nd)
    # Primary: raw sidebands (no model input). Secondary: expected signal
    # removed -- that depends on the fitted signal tail extrapolated into the
    # sidebands, the least certain part of the fit, and on a synthetic test
    # it made a mass-independent background look WORSE (fitted signal in the
    # high sideband 600 vs 246 true).
    c, nd = _chi2_shapes(*out["sb_low_raw"], *out["sb_high_raw"])
    out["chi2_sblow_vs_sbhigh"] = (c, nd)
    c, nd = _chi2_shapes(*out["sb_low"], *out["sb_high"])
    out["chi2_sblow_vs_sbhigh_signal_removed"] = (c, nd)
    return out


# ---------------------------------------------------------------------------
# toys: bias and coverage of the fitter itself
# ---------------------------------------------------------------------------

def sample_model(res, n, rng, ngrid=20001):
    """Draw n masses from the fitted density by inverse CDF on a fine grid."""
    model, p = res.model, res.values
    x = np.linspace(model.lo, model.hi, ngrid)
    d = model.density(x, p)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, x)


def run_toys(res, n_toys, n_events=None, bin_width=0.001, seed=12345,
             watch=("Ns", "Nb", "mu", "s1")):
    """
    Generate from the fitted model, refit (binned, started at the truth),
    return the pulls. Mean ~0 and width ~1 means the fitter is unbiased and
    its HESSE errors are right, for THIS model at THIS statistics.

    n_events rescales the toys (same S/B) -- useful because the real sample
    may be too large for many toys; note the pull is then measured at the
    toy statistics, not the real one.
    """
    rng = np.random.default_rng(seed)
    model, p = res.model, res.values
    ntot = p["Ns"] + p["Nb"]
    scale = (n_events / ntot) if n_events else 1.0
    truth = dict(p)
    truth["Ns"], truth["Nb"] = p["Ns"] * scale, p["Nb"] * scale
    nb = int(round((model.hi - model.lo) / bin_width))
    edges = np.linspace(model.lo, model.hi, nb + 1)
    pulls = {k: [] for k in watch}
    fitted = {k: [] for k in watch}
    errs = {k: [] for k in watch}
    failed = 0
    bar = Progress(n_toys, "[toys]", " toys")
    for _ in range(n_toys):
        n = rng.poisson(ntot * scale)
        cnt, _ = np.histogram(sample_model(res, n, rng), bins=edges)
        try:
            r = fit(model, counts=cnt, edges=edges, start=truth,
                    fixed=res.fixed, staged=False, verbose=False,
                    heartbeat=False)
        except Exception:
            failed += 1
            bar.update(1)
            continue
        if not r.valid:
            failed += 1
        else:
            for k in watch:
                if k in r.errors and r.errors[k] > 0:
                    pulls[k].append((r.values[k] - truth[k]) / r.errors[k])
                    fitted[k].append(r.values[k])
                    errs[k].append(r.errors[k])
        bar.update(1, extra=f"failed {failed}")
    bar.close()
    summ = {"n_toys": int(n_toys), "failed": int(failed),
            "events_per_toy": float(ntot * scale)}
    # Quantities that stay meaningful when HESSE errors are not (a degenerate
    # model can report a tiny error along a flat direction, and a handful of
    # toys with pulls of 1e3 then dominate a plain mean/RMS):
    #   bias_percent        mean(fitted - truth) / truth
    #   spread_over_error   actual scatter of the fitted value / median reported
    #                       error -> 1 if the errors are right, > 1 if too small
    #   robust pull         median and half-width of the central 68%
    for k, v in pulls.items():
        v = np.asarray(v)
        if v.size < 2:
            continue
        fv, ev = np.asarray(fitted[k]), np.asarray(errs[k])
        q16, q50, q84 = np.percentile(v, [16, 50, 84])
        d = {"n": int(v.size),
             "spread_over_error": float(fv.std(ddof=1) / np.median(ev)),
             "pull_median": float(q50),
             "pull_68_halfwidth": float(0.5 * (q84 - q16)),
             "pull_mean": float(v.mean()), "pull_width": float(v.std(ddof=1)),
             "n_abs_pull_above_5": int(np.sum(np.abs(v) > 5))}
        if abs(truth[k]) > 0 and k in ("Ns", "Nb"):
            d["bias_percent"] = float(100 * (fv.mean() / truth[k] - 1))
            d["bias_percent_err"] = float(100 * fv.std(ddof=1)
                                          / np.sqrt(fv.size) / truth[k])
        summ[k] = d
    return summ, {k: np.asarray(v) for k, v in pulls.items()}


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import mplhep as hep
        plt.style.use(hep.style.CMS)
        return plt, hep
    except Exception:
        return plt, None


def plot_fit(res, path, mass=None, counts=None, edges=None, nbins=100,
             gof=None, rlabel="", sp_regions=None):
    """
    The 2018 layout (CMS label, N_S/N_B/S-B/chi2 box, pull panel with 1-2 sigma
    bands), plus the two signal components, and a log-y page on which the
    tails and the background shape can actually be judged.
    """
    plt, hep = _plt()
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.gridspec import GridSpec
    model, p = res.model, res.values
    lo, hi = model.lo, model.hi
    if mass is not None:
        e = np.linspace(lo, hi, nbins + 1)
        c, _ = np.histogram(mass, bins=e)
    else:
        # rebin the fit histogram to the plotting binning if it divides
        k = max(1, (len(edges) - 1) // nbins)
        nb = (len(edges) - 1) // k
        c = np.asarray(counts)[: nb * k].reshape(nb, k).sum(1)
        e = np.asarray(edges)[: nb * k + 1: k]
    ctr, bw = 0.5 * (e[1:] + e[:-1]), e[1] - e[0]
    expb = model.bin_expect(e, p)
    pull = (c - expb) / np.sqrt(np.clip(expb, 1e-9, None))
    xs = np.linspace(lo, hi, 800)
    sig = p["Ns"] * model.signal_pdf(xs, p) * bw
    bkg = p["Nb"] * model.background_pdf(xs, p) * bw
    comps = model.signal_components(xs, p)
    gof = gof or goodness_of_fit(res, counts=c, edges=e)

    with PdfPages(path) as pdf:
        for logy in (False, True):
            fig = plt.figure(figsize=(9, 8))
            gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
            ax, axp = fig.add_subplot(gs[0]), None
            axp = fig.add_subplot(gs[1], sharex=ax)
            ax.errorbar(ctr, c, yerr=np.sqrt(c), fmt="o", ms=3, color="black",
                        label="Data")
            ax.plot(xs, sig + bkg, color="#c1272d", lw=2, label="fit")
            ax.plot(xs, bkg, color="#3f7fbf", lw=1.6, ls="--", label="background")
            ax.fill_between(xs, bkg, sig + bkg, color="#c1272d", alpha=0.12,
                            label="signal")
            if len(comps) > 1:
                for (lab, cc), col in zip(comps, ("#e76300", "#832db6")):
                    ax.plot(xs, p["Ns"] * cc * bw + bkg, color=col, lw=1.1,
                            ls=":", label=lab + " + bkg")
            if sp_regions is not None:
                for a in (*sp_regions["SR"], sp_regions["SB_low"][1],
                          sp_regions["SB_high"][0]):
                    ax.axvline(a, color="0.5", lw=0.8, ls="-.")
            ax.set_ylabel(f"Events / {bw*1e3:.0f} MeV")
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(max(0.5, 0.3 * c[c > 0].min()), 30 * c.max())
            else:
                ax.set_ylim(0, 1.3 * c.max())
            ax.set_xlim(lo, hi)
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.legend(loc="upper right", fontsize=10)
            sob = p["Ns"] / max(p["Nb"], 1e-9)
            ax.text(0.03, 0.95,
                    (rf"$N_S={p['Ns']:.0f}\pm{res.errors['Ns']:.0f}$" "\n"
                     rf"$N_B={p['Nb']:.0f}\pm{res.errors['Nb']:.0f}$" "\n"
                     if res.error_reliable else
                     rf"$N_S={p['Ns']:.0f}$ (error n/a)" "\n"
                     rf"$N_B={p['Nb']:.0f}$" "\n") +
                    rf"$S/B={sob:.1f}$" "\n"
                    rf"$\chi^2/\mathrm{{ndf}}={gof['chi2']:.0f}/{gof['ndf']}$" "\n"
                    f"{res.mode}, MIGRAD {'valid' if res.valid else 'INVALID'}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=10)
            if hep is not None:
                hep.cms.label("Preliminary", ax=ax, data=True, rlabel=rlabel,
                              fontsize=13)
            axp.axhspan(-2, 2, color="0.85", zorder=0)
            axp.axhspan(-1, 1, color="0.72", zorder=0)
            axp.axhline(0, color="#c1272d", lw=1)
            axp.plot(ctr, pull, "o", ms=3, color="black")
            axp.set_ylim(-5, 5)
            axp.set_ylabel("pull")
            axp.set_xlabel(r"$m(\mu\mu)$ [GeV]")
            axp.set_xlim(lo, hi)
            pdf.savefig(fig, bbox_inches="tight")
            if not logy:
                fig.savefig(str(path).rsplit(".", 1)[0] + ".png", dpi=140,
                            bbox_inches="tight")
            plt.close(fig)
    return gof


def plot_scan(counts, edges, window, path, rlabel=""):
    """Mass spectrum well beyond the fit window after the relaxed selection:
    shows whether a trigger or skim mass cut sculpts the window."""
    plt, hep = _plt()
    ctr = 0.5 * (edges[1:] + edges[:-1])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, logy in zip(axes, (False, True)):
        ax.errorbar(ctr, counts, yerr=np.sqrt(counts), fmt="o", ms=2,
                    color="black")
        for a in window:
            ax.axvline(a, color="#c1272d", lw=1.2, ls="--")
        ax.set_xlabel(r"$m(\mu\mu)$ [GeV]")
        ax.set_ylabel(f"Events / {1e3*(edges[1]-edges[0]):.0f} MeV")
        if logy:
            ax.set_yscale("log")
        if hep is not None:
            hep.cms.label("Preliminary", ax=ax, data=True, rlabel=rlabel,
                          fontsize=11)
    axes[0].set_title("red: fit window", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_weights(sp, reg, path):
    plt, _ = _plt()
    model = sp.res.model
    x = np.linspace(model.lo, model.hi, 1000)
    ws, wb = sp._eval(x)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, ws, color="#c1272d", lw=2, label="signal sWeight")
    ax.plot(x, wb, color="#3f7fbf", lw=2, ls="--", label="background sWeight")
    ax.plot(x, ws + wb, color="0.4", lw=1, ls=":", label="sum (should be 1)")
    for a in (*reg["SR"], reg["SB_low"][1], reg["SB_high"][0]):
        ax.axvline(a, color="0.5", lw=0.8, ls="-.")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel(r"$m(\mu\mu)$ [GeV]")
    ax.set_ylabel("weight")
    ax.legend(fontsize=11)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_controls(results, path, mc=None):
    """One page per control variable: shapes on top, ratios below."""
    plt, _ = _plt()
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.gridspec import GridSpec
    with PdfPages(path) as pdf:
        for name, (edges, r) in results.items():
            ctr = 0.5 * (edges[1:] + edges[:-1])
            fig = plt.figure(figsize=(16, 7))
            gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.06, wspace=0.25)
            # left: signal extraction, sPlot vs sideband (vs MC)
            ax, axr = fig.add_subplot(gs[0, 0]), None
            axr = fig.add_subplot(gs[1, 0], sharex=ax)
            hs, es = r["splot"]
            hb, eb = r["sideband"]
            ax.errorbar(ctr, hs, yerr=es, fmt="o", ms=3, color="black",
                        label="data, sPlot signal")
            ax.errorbar(ctr, hb, yerr=eb, fmt="s", ms=3, color="#e76300",
                        mfc="none", label="data, sideband-subtracted")
            ref, ref_lab = (hb, "sideband")
            if mc is not None and name in mc:
                hm, em = mc[name]
                ax.stairs(hm, edges, color="#3f7fbf", lw=1.6, label="MC")
                ref, ref_lab = (hm, "MC")
            c, nd = r["chi2_splot_vs_sideband"]
            ax.set_title(f"sPlot vs sideband: $\\chi^2$/ndf = {c:.0f}/{nd} "
                         f"(correlated, indicative)", fontsize=11)
            ax.set_ylabel("normalised")
            ax.legend(fontsize=10)
            plt.setp(ax.get_xticklabels(), visible=False)
            for (h, e, st, col) in ((hs, es, "o", "black"), (hb, eb, "s", "#e76300")):
                rr = np.divide(h, ref, out=np.full_like(h, np.nan), where=ref > 0)
                re = np.divide(e, ref, out=np.full_like(h, np.nan), where=ref > 0)
                axr.errorbar(ctr, rr, yerr=re, fmt=st, ms=3, color=col,
                             mfc="none" if st == "s" else col)
            axr.axhline(1, color="0.4", lw=1)
            axr.set_ylim(0.5, 1.5)
            axr.set_ylabel(f"/ {ref_lab}")
            axr.set_xlabel(name)
            # right: background shape, low vs high sideband
            ax2 = fig.add_subplot(gs[0, 1])
            ax2r = fig.add_subplot(gs[1, 1], sharex=ax2)
            se = r["sb_edges"]
            ctr = 0.5 * (se[1:] + se[:-1])
            wid = np.diff(se)
            # equal-population bins: show density (per unit of the variable)
            hl, el = r["sb_low_raw"][0] / wid, r["sb_low_raw"][1] / wid
            hh, eh = r["sb_high_raw"][0] / wid, r["sb_high_raw"][1] / wid
            ax2.errorbar(ctr, hl, yerr=el, fmt="o", ms=3, color="#3f7fbf",
                         label=f"low sideband (fit: {100*r['signal_fraction_SB_low']:.1f}% "
                               f"signal)")
            ax2.errorbar(ctr, hh, yerr=eh, fmt="o", ms=3, color="#c1272d",
                         label=f"high sideband (fit: {100*r['signal_fraction_SB_high']:.1f}% "
                               f"signal)")
            c, nd = r["chi2_sblow_vs_sbhigh"]
            c0, _ = r["chi2_sblow_vs_sbhigh_signal_removed"]
            ax2.set_title(f"sidebands: $\\chi^2$/ndf = {c:.0f}/{nd} "
                          f"(fitted signal removed: {c0:.0f})", fontsize=11)
            ax2.set_ylabel("normalised density")
            ax2.legend(fontsize=10)
            plt.setp(ax2.get_xticklabels(), visible=False)
            rr = np.divide(hl, hh, out=np.full_like(hl, np.nan), where=hh > 0)
            # after signal removal a sparse bin can go negative: use |.|
            re = np.abs(rr) * np.sqrt(
                np.divide(el, hl, out=np.zeros_like(hl), where=hl != 0) ** 2
                + np.divide(eh, hh, out=np.zeros_like(hh), where=hh > 0) ** 2)
            ax2r.errorbar(ctr, rr, yerr=re, fmt="o", ms=3, color="black")
            ax2r.axhline(1, color="0.4", lw=1)
            ax2r.set_ylim(0.5, 1.5)
            ax2r.set_ylabel("low / high")
            ax2r.set_xlabel(name)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def plot_toys(pulls, path):
    plt, _ = _plt()
    from scipy.stats import norm
    keys = [k for k, v in pulls.items() if v.size]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 5),
                             squeeze=False)
    for ax, k in zip(axes[0], keys):
        v = pulls[k]
        out = int(np.sum(np.abs(v) > 5))
        ax.hist(np.clip(v, -4.99, 4.99), bins=np.linspace(-5, 5, 26),
                color="#9ec9e2", edgecolor="#3f7fbf")
        x = np.linspace(-5, 5, 200)
        ax.plot(x, v.size * 0.4 * norm.pdf(x), color="#c1272d")
        q16, q50, q84 = np.percentile(v, [16, 50, 84])
        ax.set_title(f"{k}: median {q50:+.2f}, 68% half-width "
                     f"{0.5*(q84-q16):.2f}\n{out} toys beyond |5| (in edge bins)",
                     fontsize=10)
        ax.set_xlabel("pull")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# self-test on synthetic data with KNOWN truth -- no ROOT needed
# ---------------------------------------------------------------------------

def _check_shapes(verbose=True):
    """Every model variant: pdfs integrate to 1 (scipy.quad), bin integrals
    match quad of the density, and they add up to Ns + Nb."""
    from scipy.integrate import quad
    lo, hi = JPSI_MASS - 0.25, JPSI_MASS + 0.25
    worst = 0.0
    base = dict(Ns=1000.0, Nb=400.0, mu=3.0955, s1=0.022, rsig=1.9, fcore=0.6,
                aL=1.4, nL=3.5, aR=1.8, nR=4.0, aL1=1.4, nL1=3.5, aR1=1.8,
                nR1=4.0, aL2=0.7, nL2=8.0, aR2=2.5, nR2=2.0, lam1=4.0,
                lam2=-6.0, fb=0.7)
    for sig, tl, bkg in (("dscb", "shared", "exp"), ("2dscb", "shared", "2exp"),
                         ("2dscb", "independent", "2exp"), ("2dscb", "shared", "exp")):
        m = MassModel(lo, hi, signal=sig, tails=tl, background=bkg)
        p = {k: base[k] for k in m.names}
        for fn in (m.signal_pdf, m.background_pdf):
            v = quad(lambda x: fn(np.array([x]), p)[0], lo, hi, limit=400)[0]
            worst = max(worst, abs(v - 1))
        e = np.linspace(lo, hi, 26)
        be = m.bin_expect(e, p)
        for k in (0, 7, 12, 25 - 1):
            v = quad(lambda x: m.density(np.array([x]), p)[0], e[k], e[k + 1],
                     limit=200)[0]
            worst = max(worst, abs(be[k] / v - 1))
        worst = max(worst, abs(be.sum() / (p["Ns"] + p["Nb"]) - 1))
    return worst


def selftest(outdir=None, n_sig=150_000, n_bkg=50_000, seed=7, verbose=True):
    """
    Tests the CODE on synthetic samples with KNOWN truth; no ROOT needed.

      0. every model variant: pdfs normalised, exact bin integrals
      1. identifiable configuration (2 DSCB, tails fixed at truth, one
         exponential): unbinned and binned fits valid, yields within 3 sigma
      2. stationarity sum f_s/D = sum f_b/D = 1, hence w_s + w_b = 1
      3. sPlot recovers the TRUE signal distribution of a control variable
         y that differs between signal N(0,1) and background N(1.5,1.3)
      4. applying the weights to +-100 MeV only leaves background in, by the
         amount subrange_leakage() predicts
      5. the full 2018 model (2 DSCB shared tails + 2 exp, all 13 free)
         reaches a likelihood at least as good as the truth

    and REPORTS, without failing, how the full 2018 model behaves on this
    sample: its yield shift and correlations are properties of the model,
    not of the code.
    """
    from scipy.stats import chi2 as _c2
    rng = np.random.default_rng(seed)
    lo, hi = JPSI_MASS - 0.25, JPSI_MASS + 0.25
    model = MassModel(lo, hi, background="exp")
    truth = dict(Ns=float(n_sig), Nb=float(n_bkg), mu=3.0955, s1=0.022,
                 rsig=1.9, fcore=0.6, aL=1.4, nL=3.5, aR=1.8, nR=4.0,
                 lam1=3.0)

    def as_res(mdl, vals):
        return FitResult(model=mdl, mode="truth", n_events=0, values=vals,
                         errors={}, covariance=None, free=[], fixed={},
                         valid=True, accurate_covariance=True, edm=0, nfcn=0,
                         fmin=0, at_limit=[])
    ms = sample_model(as_res(model, dict(truth, Nb=0.0)), n_sig, rng)
    mb = sample_model(as_res(model, dict(truth, Ns=0.0)), n_bkg, rng)
    mass = np.concatenate([ms, mb])
    lab = np.concatenate([np.ones(n_sig, bool), np.zeros(n_bkg, bool)])
    y = np.concatenate([rng.normal(0.0, 1.0, n_sig), rng.normal(1.5, 1.3, n_bkg)])

    checks = []

    def check(name, ok, detail):
        checks.append((name, bool(ok), detail))
        if verbose:
            print(f"[selftest] {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    worst = _check_shapes()
    check("shapes normalised, bin integrals exact (4 model variants)",
          worst < 1e-6, f"worst relative deviation from scipy.quad {worst:.1e}")

    tails = {k: truth[k] for k in model.tail_names}
    edges = np.linspace(lo, hi, 501)
    cnt, _ = np.histogram(mass, bins=edges)
    r_u = fit(model, mass=mass, fixed=tails, verbose=False, heartbeat=verbose,
              label="selftest unbinned")
    r_b = fit(model, counts=cnt, edges=edges, fixed=tails, verbose=False,
              heartbeat=verbose, label="selftest binned")
    for tag, r in (("unbinned", r_u), ("binned", r_b)):
        pull = (r.values["Ns"] - n_sig) / r.errors["Ns"]
        check(f"{tag}: MIGRAD valid, Ns within 3 sigma",
              r.valid and abs(pull) < 3,
              f"Ns = {r.values['Ns']:.0f} +- {r.errors['Ns']:.0f} "
              f"(true {n_sig}), pull {pull:+.2f}")
    rel = abs(r_u.errors["Ns"] / r_b.errors["Ns"] - 1)
    check("binned (1 MeV) and unbinned give the same Ns error", rel < 0.05,
          f"{r_u.errors['Ns']:.0f} vs {r_b.errors['Ns']:.0f}")

    sp = SPlot(r_u, mass)
    i = sp.info
    check("stationarity: sum f_s/D = sum f_b/D = 1",
          abs(i["stationarity_sum_fs_over_D"] - 1) < 1e-3
          and abs(i["stationarity_sum_fb_over_D"] - 1) < 1e-3,
          f"{i['stationarity_sum_fs_over_D']:.6f}, "
          f"{i['stationarity_sum_fb_over_D']:.6f}")
    check("w_s + w_b = 1 per event", i["max_abs_sw_s_plus_sw_b_minus_1"] < 1e-3,
          f"max deviation {i['max_abs_sw_s_plus_sw_b_minus_1']:.2e}")

    sw, _ = sp.weights(mass)
    ye = np.linspace(-4, 5, 46)
    hw, _ = np.histogram(y, bins=ye, weights=sw)
    ew2, _ = np.histogram(y, bins=ye, weights=sw ** 2)
    ht, _ = np.histogram(y[lab], bins=ye)
    ht = ht * hw.sum() / ht.sum()
    ok = ew2 > 0
    c2 = float(np.sum((hw[ok] - ht[ok]) ** 2 / ew2[ok]))
    nd = int(ok.sum() - 1)
    check("sPlot y-shape = true signal y-shape", _c2.sf(c2, nd) > 1e-3,
          f"chi2/ndf = {c2:.1f}/{nd}, p = {_c2.sf(c2, nd):.3f}; weighted "
          f"mean y {np.average(y, weights=sw):+.4f} vs true {y[lab].mean():+.4f}")

    sub = np.abs(mass - JPSI_MASS) < 0.1
    naive = float(np.sum(sw[sub] * y[sub]) / np.sum(sw[sub]))
    leak = sp.subrange_leakage(JPSI_MASS - 0.1, JPSI_MASS + 0.1)
    pred = leak["background_fraction"] * 1.5          # signal mean is 0
    err = 1.3 / np.sqrt(sub.sum())
    check("weights restricted to +-100 MeV leak background, as predicted",
          abs(naive - pred) < 4 * err and leak["background_fraction"] > 0.01,
          f"mean y = {naive:+.4f} (pure signal: 0); predicted surviving "
          f"background {100*leak['background_fraction']:.1f}% -> "
          f"{pred:+.4f} +- {err:.4f}")

    full = MassModel(lo, hi)                      # the 2018 model
    r_f = fit(full, counts=cnt, edges=edges, verbose=False, heartbeat=verbose,
              label="selftest 2018 model, all free")
    tf = dict(truth, rsig=truth["rsig"], lam2=0.0, fb=1.0)

    def bnll(mdl, p):
        mu = mdl.bin_expect(edges, p)
        return float(np.sum(mu) - np.sum(cnt * np.log(mu)))
    dn = bnll(full, r_f.values) - bnll(full, {k: tf[k] for k in full.names})
    check("2018 model, all free: minimiser reaches the truth likelihood",
          dn < 0.5, f"NLL(fit) - NLL(truth) = {dn:+.2f}")
    if verbose:
        shift = r_f.values["Ns"] / n_sig - 1
        print(f"[selftest] INFO  2018 model, all free: Ns = "
              f"{r_f.values['Ns']:.0f} ({100*shift:+.1f}% vs truth, "
              f"{shift*n_sig/r_f.errors['Ns']:+.1f} reported sigma), MIGRAD "
              f"{'valid' if r_f.valid else 'INVALID'}; largest correlations "
              + ", ".join(f"{a}-{b} {c:+.2f}" for a, b, c in r_f.top_correlations(3))
              + ". A background with more freedom than the data can "
                "constrain gives an invalid MIGRAD and unstable errors even "
                "when the curve is right; what matters for sPlot is whether "
                "the signal/background SPLIT is stable -> compare model "
                "variations on the real data.")

    if outdir:
        import os
        os.makedirs(outdir, exist_ok=True)
        plot_fit(r_u, os.path.join(outdir, "selftest_fit.pdf"), mass=mass,
                 rlabel="synthetic")
        plot_fit(r_f, os.path.join(outdir, "selftest_fit_2018model.pdf"),
                 counts=cnt, edges=edges, rlabel="synthetic, 2018 model")
    passed = all(ok for _, ok, _ in checks)
    if verbose:
        print(f"[selftest] {'ALL PASSED' if passed else 'SOME CHECKS FAILED'}")
    return passed, checks
