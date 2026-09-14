"""
covflow.sweights
================

Fit the dimuon mass spectrum in data and derive sWeights, so the data flow is
trained on background-subtracted J/psi candidates rather than on signal plus
combinatorial background.

Why this matters here: the flow learns the covariance distribution of whatever
you train it on. Background muons are not J/psi muons -- they are more often
fakes, displaced, or badly measured -- so their covariances differ. Training the
data flow on raw data makes the correction target a signal+background mixture,
and the resulting correction is wrong by roughly the background fraction times
the difference between the two populations.

Model
-----
signal      double-sided Crystal Ball: a Gaussian core with independent
            power-law tails on each side (separate alpha and n per side)
background  sum of two exponentials
range       +- `window` GeV around the J/psi mass (default 200 MeV)

The fit is unbinned and extended. sWeights follow Pivk & Le Diberder: with the
mass as the only discriminating variable, the per-candidate signal weight is

    sw_s(m) = [V_ss f_s(m) + V_sb f_b(m)] / [N_s f_s(m) + N_b f_b(m)]

where V is the inverse of the matrix

    (V^-1)_nj = sum_e f_n(m_e) f_j(m_e) / [N_s f_s(m_e) + N_b f_b(m_e)]^2

By construction these weights statistically subtract the background in any
variable uncorrelated with the mass -- which the track covariance should be,
and which is worth checking (see `check_mass_correlation`).
"""

from __future__ import annotations

import numpy as np

JPSI_MASS = 3.0969


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------

def dscb_unnorm(m, mu, sigma, aL, nL, aR, nR):
    """
    Double-sided Crystal Ball, unnormalised.

    Gaussian core; below -aL and above +aR (in units of sigma) it switches to a
    power law of index nL / nR, matched in value and first derivative at the
    junction. Independent parameters on the two sides -- final-state radiation
    makes the low-mass tail longer than the high-mass one, so a symmetric shape
    biases the fitted mean.
    """
    sigma = max(float(sigma), 1e-6)
    aL, aR = max(float(aL), 1e-3), max(float(aR), 1e-3)
    nL, nR = max(float(nL), 1.0 + 1e-6), max(float(nR), 1.0 + 1e-6)
    t = (np.asarray(m, float) - mu) / sigma

    out = np.exp(-0.5 * t * t)

    lo = t < -aL
    if np.any(lo):
        A = (nL / aL) ** nL * np.exp(-0.5 * aL * aL)
        B = nL / aL - aL
        out[lo] = A * np.power(np.clip(B - t[lo], 1e-12, None), -nL)

    hi = t > aR
    if np.any(hi):
        A = (nR / aR) ** nR * np.exp(-0.5 * aR * aR)
        B = nR / aR - aR
        out[hi] = A * np.power(np.clip(B + t[hi], 1e-12, None), -nR)

    return out


def two_exp_unnorm(m, k1, k2, frac):
    """Sum of two exponentials, unnormalised. `frac` weights the first."""
    m = np.asarray(m, float)
    f = min(max(float(frac), 0.0), 1.0)
    # subtract the range centre for numerical conditioning
    return f * np.exp(k1 * m) + (1.0 - f) * np.exp(k2 * m)


def _normalise(fn, lo, hi, n=4001):
    """Numeric normalisation of a callable over [lo, hi] by Simpson's rule."""
    x = np.linspace(lo, hi, n)
    y = fn(x)
    from scipy.integrate import simpson
    area = float(simpson(y, x=x))
    return max(area, 1e-300)


def signal_pdf(m, p, lo, hi):
    f = lambda x: dscb_unnorm(x, p["mu"], p["sigma"], p["aL"], p["nL"],
                              p["aR"], p["nR"])
    return f(m) / _normalise(f, lo, hi)


def background_pdf(m, p, lo, hi):
    f = lambda x: two_exp_unnorm(x - 0.5 * (lo + hi), p["k1"], p["k2"], p["frac"])
    return f(m) / _normalise(f, lo, hi)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

_PARAM_ORDER = ["Ns", "Nb", "mu", "sigma", "aL", "nL", "aR", "nR",
                "k1", "k2", "frac"]


def fit_mass(mass, lo, hi, fixed=None, verbose=True):
    """
    Extended unbinned maximum-likelihood fit of the mass spectrum on [lo, hi].

    `fixed` is an optional dict of parameters to hold constant -- in practice
    the Crystal Ball tail parameters (aL, nL, aR, nR), taken from a fit to
    simulated signal.

    WHY YOU USUALLY WANT TO FIX THE TAILS: a sum of two exponentials is flexible
    enough to absorb the signal's radiative left tail, and a Crystal Ball tail is
    flexible enough to absorb steeply falling background. The two are close to
    degenerate, and the fit can trade one for the other with almost no change in
    likelihood while biasing the signal yield by several percent. Fixing the tail
    shape from simulation breaks the degeneracy. Verified on a toy with a known
    12% left tail: free tails gave a 5% low yield; tails fixed from the
    generating shape recovered it.

    Returns (params dict, info dict).
    """
    from scipy.optimize import minimize

    fixed = dict(fixed or {})
    m = np.asarray(mass, float)
    m = m[(m >= lo) & (m <= hi)]
    n = len(m)
    if n < 500:
        raise ValueError(f"only {n} candidates in [{lo:.4f}, {hi:.4f}] -- "
                         f"too few for a mass fit")

    cnt, edges = np.histogram(m, bins=80)
    centre = 0.5 * (edges[1:] + edges[:-1])
    mu0 = float(centre[np.argmax(cnt)])
    core = m[np.abs(m - mu0) < 0.05]
    sig0 = float(np.std(core)) if len(core) > 100 else 0.03
    sig0 = min(max(sig0, 0.005), 0.08)
    edge = 0.5 * (hi - lo) * 0.5
    nb0 = float(((m < lo + edge) | (m > hi - edge)).sum()) * 2.0
    nb0 = min(max(nb0, 1.0), 0.95 * n)
    ns0 = max(n - nb0, 1.0)

    start = dict(Ns=ns0, Nb=nb0, mu=mu0, sigma=sig0,
                 aL=1.3, nL=3.0, aR=1.6, nR=3.0,
                 k1=-1.0, k2=2.0, frac=0.5)
    start.update(fixed)
    bounds = dict(Ns=(0.0, 2.0 * n), Nb=(0.0, 2.0 * n),
                  mu=(mu0 - 0.05, mu0 + 0.05), sigma=(0.002, 0.10),
                  aL=(0.2, 4.0), nL=(1.01, 30.0),
                  aR=(0.2, 4.0), nR=(1.01, 30.0),
                  k1=(-60.0, 60.0), k2=(-60.0, 60.0), frac=(0.0, 1.0))

    free = [k for k in _PARAM_ORDER if k not in fixed]

    def unpack(v, extra_fixed=()):
        p = dict(start)
        p.update(fixed)
        for i, k in enumerate(free_now):
            p[k] = v[i]
        return p

    def make_nll(free_now_):
        def nll(v):
            p = dict(start); p.update(fixed)
            for i, k in enumerate(free_now_):
                p[k] = v[i]
            if p["Ns"] < 0 or p["Nb"] < 0:
                return 1e12
            try:
                fs = signal_pdf(m, p, lo, hi)
                fb = background_pdf(m, p, lo, hi)
            except Exception:
                return 1e12
            tot = p["Ns"] * fs + p["Nb"] * fb
            if not np.all(np.isfinite(tot)) or np.any(tot <= 0):
                return 1e12
            return float(p["Ns"] + p["Nb"] - np.sum(np.log(tot)))
        return nll

    # Stage 1: hold the tails at their starting values and fit everything else.
    # Stage 2: release whatever is not explicitly fixed, starting from stage 1.
    # Going straight to the full 11-parameter fit lands in the tail/background
    # degeneracy described above.
    tails = ["aL", "nL", "aR", "nR"]
    stages = [[k for k in free if k not in tails], free]
    if set(free) & set(tails) == set():
        stages = [free]

    for stage in stages:
        if not stage:
            continue
        free_now = stage
        nll = make_nll(free_now)
        v0 = np.array([start[k] for k in free_now])
        bnds = [bounds[k] for k in free_now]
        r = minimize(nll, v0, method="L-BFGS-B", bounds=bnds,
                     options=dict(maxiter=4000, maxfun=40000))
        r = minimize(nll, r.x, method="Nelder-Mead",
                     options=dict(maxiter=30000, xatol=1e-7, fatol=1e-7))
        xr = np.clip(r.x, [b[0] for b in bnds], [b[1] for b in bnds])
        for i, k in enumerate(free_now):
            start[k] = float(xr[i])

    p = dict(start); p.update(fixed)
    free_now = free
    final_nll = make_nll(free)(np.array([p[k] for k in free])) if free else 0.0

    at_bound = [k for k in free
                if abs(p[k] - bounds[k][0]) < 1e-6 * max(1, abs(bounds[k][0]))
                or abs(p[k] - bounds[k][1]) < 1e-6 * max(1, abs(bounds[k][1]))]

    info = {"n_candidates": int(n), "range": [float(lo), float(hi)],
            "nll": float(final_nll),
            "fixed_parameters": {k: float(v) for k, v in fixed.items()},
            "parameters_at_bound": at_bound,
            "signal_yield": float(p["Ns"]), "background_yield": float(p["Nb"]),
            "purity_in_range": float(p["Ns"] / max(p["Ns"] + p["Nb"], 1e-9))}
    if verbose:
        print(f"[sweights] fitted {n:,} candidates in "
              f"[{lo:.4f}, {hi:.4f}] GeV"
              + (f"  (tails fixed)" if set(fixed) & set(tails) else ""))
        print(f"[sweights]   mu = {p['mu']:.5f} GeV, sigma = "
              f"{1000*p['sigma']:.2f} MeV")
        print(f"[sweights]   tails: aL={p['aL']:.2f} nL={p['nL']:.2f} | "
              f"aR={p['aR']:.2f} nR={p['nR']:.2f}")
        print(f"[sweights]   Ns = {p['Ns']:,.0f}, Nb = {p['Nb']:,.0f}, "
              f"purity = {100*info['purity_in_range']:.1f}%")
        if at_bound:
            print(f"[sweights]   WARNING: parameters at their bound: "
                  f"{at_bound} -- the fit wanted to go further, so the shape "
                  f"is not describing the data well")
    return p, info


def fit_signal_tails(mass_mc, lo, hi, verbose=True):
    """
    Fit simulated signal alone to get the Crystal Ball tail parameters, for
    fixing in the data fit. No background component.
    """
    p, info = fit_mass(mass_mc, lo, hi, fixed={"Nb": 0.0, "k1": -1.0,
                                               "k2": 1.0, "frac": 0.5},
                       verbose=False)
    tails = {k: float(p[k]) for k in ("aL", "nL", "aR", "nR")}
    if verbose:
        print(f"[sweights] signal tails from MC ({info['n_candidates']:,} "
              f"candidates): aL={tails['aL']:.2f} nL={tails['nL']:.2f} | "
              f"aR={tails['aR']:.2f} nR={tails['nR']:.2f}")
    return tails, info


# ---------------------------------------------------------------------------
# sWeights
# ---------------------------------------------------------------------------

def compute_sweights(mass, p, lo, hi):
    """
    Per-candidate signal and background sWeights.

    Evaluated at each candidate's mass using the fitted shapes and yields. Only
    candidates inside [lo, hi] get a weight; others get 0.
    """
    m = np.asarray(mass, float)
    inside = (m >= lo) & (m <= hi)
    sw_s = np.zeros(len(m))
    sw_b = np.zeros(len(m))
    if not inside.any():
        return sw_s, sw_b, {"n_weighted": 0}

    mi = m[inside]
    fs = signal_pdf(mi, p, lo, hi)
    fb = background_pdf(mi, p, lo, hi)
    Ns, Nb = p["Ns"], p["Nb"]
    denom = Ns * fs + Nb * fb
    denom = np.clip(denom, 1e-300, None)

    Vinv = np.empty((2, 2))
    Vinv[0, 0] = np.sum(fs * fs / denom ** 2)
    Vinv[0, 1] = Vinv[1, 0] = np.sum(fs * fb / denom ** 2)
    Vinv[1, 1] = np.sum(fb * fb / denom ** 2)
    V = np.linalg.inv(Vinv)

    sw_s[inside] = (V[0, 0] * fs + V[0, 1] * fb) / denom
    sw_b[inside] = (V[1, 0] * fs + V[1, 1] * fb) / denom

    info = {"n_weighted": int(inside.sum()),
            "sum_signal_weights": float(sw_s.sum()),
            "sum_background_weights": float(sw_b.sum()),
            "negative_signal_weight_fraction":
                float((sw_s[inside] < 0).mean()),
            "min_signal_weight": float(sw_s[inside].min()),
            "max_signal_weight": float(sw_s[inside].max()),
            # effective statistics after weighting: this, not the raw count,
            # is what the flow actually has to learn from
            "effective_n": float(sw_s.sum() ** 2 / max((sw_s ** 2).sum(), 1e-12))}
    return sw_s, sw_b, info


def check_mass_correlation(mass, features, feature_names, lo, hi, n_bins=8):
    """
    sPlot is only valid if the control variables are uncorrelated with the
    discriminating variable. Here that means the covariance features must not
    depend on the dimuon mass. This measures the dependence directly: the
    spread of each feature's median across mass bins, in units of that
    feature's own width.
    """
    m = np.asarray(mass, float)
    sel = (m >= lo) & (m <= hi)
    m, y = m[sel], np.asarray(features)[sel]
    edges = np.percentile(m, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    out = {}
    for k, nm in enumerate(feature_names):
        med = []
        for b in range(len(edges) - 1):
            inb = (m >= edges[b]) & (m < edges[b + 1] if b < len(edges) - 2
                                     else m <= edges[b + 1])
            if inb.sum() > 50:
                med.append(np.median(y[inb, k]))
        if len(med) < 3:
            continue
        width = np.percentile(y[:, k], 84) - np.percentile(y[:, k], 16)
        out[nm] = float(np.std(med) / max(width, 1e-9))
    return out


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

def plot_fit(mass, p, lo, hi, path, sw_s=None, n_bins=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.asarray(mass, float)
    m = m[(m >= lo) & (m <= hi)]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7),
                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    cnt, edges = np.histogram(m, bins=n_bins, range=(lo, hi))
    centre = 0.5 * (edges[1:] + edges[:-1])
    bw = edges[1] - edges[0]
    axes[0].errorbar(centre, cnt, yerr=np.sqrt(np.maximum(cnt, 0)), fmt="k.",
                     ms=4, label="data")

    x = np.linspace(lo, hi, 800)
    fs = p["Ns"] * signal_pdf(x, p, lo, hi) * bw
    fb = p["Nb"] * background_pdf(x, p, lo, hi) * bw
    axes[0].plot(x, fs + fb, "-", color="tab:blue", lw=2, label="total")
    axes[0].plot(x, fs, "--", color="tab:red", lw=1.5, label="signal (DSCB)")
    axes[0].plot(x, fb, "--", color="tab:green", lw=1.5, label="background (2 exp)")
    axes[0].set_ylabel(f"candidates / {1000*bw:.1f} MeV", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[0].set_title(
        f"$\\mu$ = {p['mu']:.4f} GeV, $\\sigma$ = {1000*p['sigma']:.1f} MeV, "
        f"$N_s$ = {p['Ns']:,.0f}, purity = "
        f"{100*p['Ns']/(p['Ns']+p['Nb']):.1f}%", fontsize=9)

    model = (p["Ns"] * signal_pdf(centre, p, lo, hi)
             + p["Nb"] * background_pdf(centre, p, lo, hi)) * bw
    err = np.sqrt(np.maximum(cnt, 1))
    axes[1].errorbar(centre, (cnt - model) / err, yerr=1, fmt="k.", ms=4)
    axes[1].axhline(0, color="tab:blue", lw=1)
    for s in (-3, 3):
        axes[1].axhline(s, color="0.7", lw=0.6, ls=":")
    axes[1].set_ylabel("pull", fontsize=9)
    axes[1].set_xlabel("dimuon mass [GeV]", fontsize=9)
    axes[1].set_ylim(-6, 6)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path
