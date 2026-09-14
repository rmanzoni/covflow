"""
covflow.validate
================

The suite that decides whether the correction is believable. Rough order of how
easily each check can fail (hardest last):

  roundtrip_error              parameterisation invertible on THESE matrices
  invalid_matrix_fraction      inputs already PD (do not mask if not)
  latent_check_mc              MC pushed through flow_mc ~ N(0,1) + exact inverse
  pd_fraction_after_correction 1.0 by construction; verified numerically
  latent_clip_fraction         MC tracks landing where the data flow saw nothing
  marginal_w1  before -> after large gain per feature (1D morphing gets these)
  corr_max_dabs before -> after the part 1D morphing CANNOT reach
  classifier_auc before -> after ~0.5 after: the only test that can really fail
  coverage                     MC out of data range / in starved bins

All distances are weighted (data carries sWeights). The classifier is a small
weighted MLP; AUC is a weighted Mann-Whitney estimate, so no sklearn dependency.
"""

from __future__ import annotations

import numpy as np

from . import features as F


# ---------------------------------------------------------------------------
# weighted primitives
# ---------------------------------------------------------------------------

def _wq(x, w, qs):
    """Weighted quantiles."""
    x = np.asarray(x); w = np.clip(np.asarray(w, float), 0, None)
    o = np.argsort(x); x, w = x[o], w[o]
    cw = np.cumsum(w) - 0.5 * w
    cw /= max(w.sum(), 1e-12)
    return np.interp(qs, cw, x)


def weighted_w1(xa, wa, xb, wb, n=512):
    """Weighted 1-Wasserstein via quantile functions (scipy if present)."""
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(xa, xb,
                                          u_weights=np.clip(wa, 0, None),
                                          v_weights=np.clip(wb, 0, None)))
    except Exception:
        qs = (np.arange(n) + 0.5) / n
        return float(np.mean(np.abs(_wq(xa, wa, qs) - _wq(xb, wb, qs))))


def weighted_corr(Y, w):
    """(N,d) -> (d,d) weighted Pearson correlation."""
    w = np.clip(np.asarray(w, float), 0, None)
    sw = w.sum()
    mu = (w[:, None] * Y).sum(0) / sw
    Yc = Y - mu
    cov = (w[:, None] * Yc).T @ Yc / sw
    d = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    return cov / d[:, None] / d[None, :]


def weighted_auc(score, label, weight):
    """
    Weighted AUC = P(score(data) > score(mc)) with ties at 0.5, via ranks.
    label: 1 for data (positive), 0 for mc.
    """
    score = np.asarray(score); label = np.asarray(label)
    weight = np.clip(np.asarray(weight, float), 0, None)
    o = np.argsort(score, kind="mergesort")
    s, lab, w = score[o], label[o], weight[o]
    # average ranks in weight units, tie-aware
    # mid-rank in weight units: (weight strictly below) + 0.5*(weight in tie block)
    ranks = np.empty(len(s))
    i = 0
    cum = 0.0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        wsum = w[i:j + 1].sum()
        ranks[i:j + 1] = cum + 0.5 * wsum
        cum += wsum
        i = j + 1
    wp = (w * (lab == 1)).sum()
    wn = (w * (lab == 0)).sum()
    if wp <= 0 or wn <= 0:
        return float("nan")
    sum_ranks_pos = (ranks * w * (lab == 1)).sum()
    auc = (sum_ranks_pos - 0.5 * wp * wp) / (wp * wn)
    return float(np.clip(auc, 0.0, 1.0))


# ---------------------------------------------------------------------------
# classifier test (MC vs data on the 15 features)
# ---------------------------------------------------------------------------

def classifier_auc(Y_mc, w_mc, Y_data, w_data, scaler=None,
                   epochs=40, lr=2e-3, hidden=(64, 64), device="cpu", seed=0):
    """
    Train a small weighted MLP to separate MC (0) from data (1) on the feature
    vectors; return held-out weighted AUC. ~0.5 means indistinguishable.
    Pass corrected-MC features to get the 'after' number.
    """
    import torch
    rng = np.random.default_rng(seed)
    if scaler is not None:
        Y_mc = scaler.x(Y_mc); Y_data = scaler.x(Y_data)
    X = np.concatenate([Y_mc, Y_data], 0).astype(np.float32)
    y = np.concatenate([np.zeros(len(Y_mc)), np.ones(len(Y_data))]).astype(np.float32)
    # balance classes by weight so AUC is not driven by yield
    w = np.concatenate([np.clip(w_mc, 0, None) / max(np.clip(w_mc,0,None).sum(),1e-9),
                        np.clip(w_data, 0, None) / max(np.clip(w_data,0,None).sum(),1e-9)]).astype(np.float32)
    idx = rng.permutation(len(X))
    X, y, w = X[idx], y[idx], w[idx]
    ntr = int(0.7 * len(X))
    Xtr, Xte = X[:ntr], X[ntr:]
    ytr, yte = y[:ntr], y[ntr:]
    wtr, wte = w[:ntr], w[ntr:]

    torch.manual_seed(seed)
    layers, d = [], X.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]; d = h
    layers += [torch.nn.Linear(d, 1)]
    net = torch.nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.as_tensor(Xtr); yt = torch.as_tensor(ytr); wt = torch.as_tensor(wtr)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(epochs):
        opt.zero_grad()
        logit = net(Xt.to(device)).squeeze(-1)
        loss = (bce(logit, yt.to(device)) * wt.to(device)).sum() / wt.sum().clamp_min(1e-9)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        sc = net(torch.as_tensor(Xte).to(device)).squeeze(-1).cpu().numpy()
    return weighted_auc(sc, yte, wte)


# ---------------------------------------------------------------------------
# full report
# ---------------------------------------------------------------------------

def run(packed_mc, C_mc, w_mc,
        packed_data, C_data, w_data,
        packed_corr, y_mc, y_corr,
        scaler, param="logsigma_corr",
        classifier=True, device="cpu"):
    """
    Assemble the report dict. `packed_corr, y_mc, y_corr` come from correct.morph.
    """
    to_feat, to_mat = F.get_transforms(param)
    names = F.feature_names(param)
    rep = {}

    # roundtrip on the actual matrices
    rep["roundtrip_error"] = F.roundtrip_error(packed_mc, param)

    # inputs already PD?
    rep["invalid_matrix_fraction_mc"] = float(1 - F.is_positive_definite(
        F.packed_to_matrix(packed_mc)).mean())
    rep["invalid_matrix_fraction_data"] = float(1 - F.is_positive_definite(
        F.packed_to_matrix(packed_data)).mean())

    # PD after correction
    rep["pd_fraction_after_correction"] = float(F.is_positive_definite(
        F.packed_to_matrix(packed_corr)).mean())

    # data features (for the marginals / correlations / classifier)
    y_data = to_feat(F.packed_to_matrix(packed_data))

    # marginal W1 before -> after, per feature
    w1_before, w1_after = {}, {}
    for k, nm in enumerate(names):
        w1_before[nm] = weighted_w1(y_mc[:, k], w_mc, y_data[:, k], w_data)
        w1_after[nm] = weighted_w1(y_corr[:, k], w_mc, y_data[:, k], w_data)
    rep["marginal_w1_before"] = w1_before
    rep["marginal_w1_after"] = w1_after
    rep["marginal_w1_mean_before"] = float(np.mean(list(w1_before.values())))
    rep["marginal_w1_mean_after"] = float(np.mean(list(w1_after.values())))

    # correlation structure: max |rho_mc - rho_data|, before vs after
    Rd = weighted_corr(y_data, w_data)
    Rb = weighted_corr(y_mc, w_mc)
    Ra = weighted_corr(y_corr, w_mc)
    tri = np.triu_indices(F.N_FEATURES, 1)
    rep["corr_max_dabs_before"] = float(np.max(np.abs(Rb - Rd)[tri]))
    rep["corr_max_dabs_after"] = float(np.max(np.abs(Ra - Rd)[tri]))

    # coverage: MC feature outside data feature range, per feature
    cov_out = {}
    for k, nm in enumerate(names):
        lo, hi = _wq(y_data[:, k], w_data, [0.001, 0.999])
        cov_out[nm] = float(((y_mc[:, k] < lo) | (y_mc[:, k] > hi)).mean())
    rep["coverage_mc_out_of_data_range"] = cov_out
    rep["coverage_worst"] = float(max(cov_out.values()))

    # classifier AUC before / after -- the decisive test
    if classifier:
        rep["classifier_auc_before"] = classifier_auc(
            y_mc, w_mc, y_data, w_data, scaler=scaler, device=device)
        rep["classifier_auc_after"] = classifier_auc(
            y_corr, w_mc, y_data, w_data, scaler=scaler, device=device)

    # median sigma ratios (interpretable, per scale feature)
    sig_ratio = {}
    for k in range(F.DIM):
        med_mc = _wq(y_mc[:, k], w_mc, [0.5])[0]
        med_dat = _wq(y_data[:, k], w_data, [0.5])[0]
        med_cor = _wq(y_corr[:, k], w_mc, [0.5])[0]
        sig_ratio[names[k]] = {"mc_over_data": float(np.exp(med_mc - med_dat)),
                               "corr_over_data": float(np.exp(med_cor - med_dat))}
    rep["median_sigma_ratio"] = sig_ratio

    return rep


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# context binning: closure at fixed kinematics
# ---------------------------------------------------------------------------
#
# The morph corrects p(y | c). It does NOT and cannot correct a difference in
# p(c) itself. So if MC and data have different (pt, eta, ...) spectra -- which
# they always do when MC is a signal sample and data is an inclusive trigger
# stream -- the *marginal* comparison mixes two things:
#
#     (i)  residual error of the flows                    <- what we want
#     (ii) different kinematics being averaged over       <- confound
#
# Two ways out, both provided here:
#   * bin in context and evaluate closure cell by cell  -> binned_closure()
#   * reweight MC to data's context spectrum and redo
#     the global metrics                                -> context_weights()
# The second gives one clean number; the first shows where it works and where
# it does not.

def context_bin_edges(C_pool, w_pool=None, n_bins=4):
    """
    Quantile edges per context dimension, from the POOLED sample so that data
    and MC land in the same cells. n_bins may be an int (same for every
    dimension) or a per-dimension sequence.
    """
    C_pool = np.asarray(C_pool, float)
    k = C_pool.shape[1]
    nb = [n_bins] * k if np.isscalar(n_bins) else list(n_bins)
    if len(nb) != k:
        raise ValueError(f"n_bins has {len(nb)} entries for {k} context dims")
    edges = []
    for d in range(k):
        n = max(int(nb[d]), 1)
        if n == 1:
            edges.append(np.array([-np.inf, np.inf]))
            continue
        qs = np.linspace(0, 1, n + 1)[1:-1]
        w = np.ones(len(C_pool)) if w_pool is None else w_pool
        inner = _wq(C_pool[:, d], w, qs)
        # collapse duplicate edges (discrete variables like nPV or nHits)
        inner = np.unique(inner)
        edges.append(np.concatenate([[-np.inf], inner, [np.inf]]))
    return edges


def assign_bins(C, edges):
    """Flat cell index per row, plus the per-dimension shape."""
    C = np.asarray(C, float)
    shape = tuple(len(e) - 1 for e in edges)
    idx = np.zeros(len(C), dtype=np.int64)
    for d, e in enumerate(edges):
        i = np.clip(np.digitize(C[:, d], e[1:-1]), 0, shape[d] - 1)
        idx = idx * shape[d] + i
    return idx, shape


def cell_label(flat_idx, shape, names, edges):
    """Human-readable description of a flat cell index."""
    sub = []
    rem = flat_idx
    for d in reversed(range(len(shape))):
        sub.append(rem % shape[d])
        rem //= shape[d]
    sub = sub[::-1]
    parts = []
    for d, i in enumerate(sub):
        lo, hi = edges[d][i], edges[d][i + 1]
        lo_s = "-inf" if not np.isfinite(lo) else f"{lo:.3g}"
        hi_s = "inf" if not np.isfinite(hi) else f"{hi:.3g}"
        parts.append(f"{names[d]}[{lo_s},{hi_s})")
    return " ".join(parts)


def context_weights(C_mc, w_mc, C_data, w_data, edges, max_weight=50.0):
    """
    Per-track weights that make MC's context spectrum match data's.

    Histogram ratio in the given cells, normalised so the total MC weight is
    preserved. Cells where MC has no entries but data does cannot be reweighted
    -- their data fraction is reported as `uncovered`, and it is a hard limit on
    how well any conditional correction can reproduce data marginals.
    """
    im, shape = assign_bins(C_mc, edges)
    idd, _ = assign_bins(C_data, edges)
    ncell = int(np.prod(shape))
    hm = np.bincount(im, weights=np.clip(w_mc, 0, None), minlength=ncell)
    hd = np.bincount(idd, weights=np.clip(w_data, 0, None), minlength=ncell)
    hm_n = hm / max(hm.sum(), 1e-12)
    hd_n = hd / max(hd.sum(), 1e-12)
    uncovered = float(hd_n[(hm <= 0) & (hd > 0)].sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(hm_n > 0, hd_n / np.maximum(hm_n, 1e-12), 0.0)
    ratio = np.clip(ratio, 0.0, max_weight)
    w = np.clip(w_mc, 0, None) * ratio[im]
    if w.sum() > 0:
        w *= np.clip(w_mc, 0, None).sum() / w.sum()
    info = {"uncovered_data_fraction": uncovered,
            "max_ratio": float(ratio.max()),
            "clipped_cells": int((ratio >= max_weight).sum()),
            "n_cells": ncell,
            "effective_n": float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))}
    return w, info


def binned_closure(y_mc, w_mc, C_mc, y_corr,
                   y_data, w_data, C_data,
                   edges, context_names, feature_indices=None,
                   min_count=400, classifier=True, device="cpu",
                   param="logsigma_corr"):
    """
    Repeat the closure test inside each context cell.

    Within a cell the kinematics of data and MC are (approximately) the same, so
    a residual difference is attributable to the flows rather than to the
    spectra. Returns a dict with a per-cell list and an aggregate.
    """
    names = F.feature_names(param)
    idx = list(range(len(names))) if feature_indices is None else list(feature_indices)
    im, shape = assign_bins(C_mc, edges)
    idd, _ = assign_bins(C_data, edges)
    ncell = int(np.prod(shape))

    cells, skipped = [], 0
    for c in range(ncell):
        sm = im == c
        sd = idd == c
        nm, nd = int(sm.sum()), int(sd.sum())
        if nm < min_count or nd < min_count:
            skipped += 1
            continue
        ent = {"cell": c,
               "label": cell_label(c, shape, context_names, edges),
               "n_mc": nm, "n_data": nd}
        w1b, w1a = {}, {}
        for k in idx:
            w1b[names[k]] = weighted_w1(y_mc[sm, k], w_mc[sm],
                                        y_data[sd, k], w_data[sd])
            w1a[names[k]] = weighted_w1(y_corr[sm, k], w_mc[sm],
                                        y_data[sd, k], w_data[sd])
        ent["w1_before"], ent["w1_after"] = w1b, w1a
        ent["w1_mean_before"] = float(np.mean(list(w1b.values())))
        ent["w1_mean_after"] = float(np.mean(list(w1a.values())))
        if classifier:
            ent["auc_before"] = classifier_auc(
                y_mc[sm][:, idx], w_mc[sm], y_data[sd][:, idx], w_data[sd],
                device=device)
            ent["auc_after"] = classifier_auc(
                y_corr[sm][:, idx], w_mc[sm], y_data[sd][:, idx], w_data[sd],
                device=device)
        cells.append(ent)

    out = {"cells": cells, "n_cells_total": ncell, "n_cells_skipped": skipped,
           "min_count": min_count}
    if cells:
        out["w1_mean_before"] = float(np.mean([c["w1_mean_before"] for c in cells]))
        out["w1_mean_after"] = float(np.mean([c["w1_mean_after"] for c in cells]))
        if classifier:
            out["auc_before_mean"] = float(np.mean([c["auc_before"] for c in cells]))
            out["auc_after_mean"] = float(np.mean([c["auc_after"] for c in cells]))
            out["auc_after_worst"] = float(np.max([c["auc_after"] for c in cells]))
        # per-feature aggregate across cells
        agg_b = {names[k]: float(np.mean([c["w1_before"][names[k]] for c in cells]))
                 for k in idx}
        agg_a = {names[k]: float(np.mean([c["w1_after"][names[k]] for c in cells]))
                 for k in idx}
        out["w1_before_per_feature"] = agg_b
        out["w1_after_per_feature"] = agg_a
    return out


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def block_feature_correlation(y, C, edges, w=None, param="logsigma_corr"):
    """
    Does the FLOW factorise by block? -- a different question from whether the
    covariance matrix does.

    The measured r-phi / r-z separation says each track's 5x5 matrix is block
    diagonal. It does NOT say the *features* of the two blocks are independent
    ACROSS tracks: sigma_phi and sigma_lambda both grow for a track with few
    hits or lots of material, so they can be strongly correlated track to track
    even though the matrix never mixes them.

    Only if the cross-block feature correlation is small -- at fixed context,
    since shared kinematic dependence is already handled by conditioning -- can
    the 9-D flow safely be split into independent 6-D (r-phi) and 3-D (r-z)
    flows. This measures that, cell by cell, and returns the worst case.
    """
    names = F.feature_names(param)
    rphi, rz = F.subset_indices("rphi"), F.subset_indices("rz")
    idx = F.subset_indices("block")
    pos = {f: n for n, f in enumerate(idx)}

    cells, _ = assign_bins(C, edges)
    w = np.ones(len(y)) if w is None else w

    per_cell = []
    for c in np.unique(cells):
        m = cells == c
        if m.sum() < 200:
            continue
        R = weighted_corr(y[m][:, idx], w[m])
        cross = [abs(R[pos[i], pos[j]]) for i in rphi for j in rz]
        within = [abs(R[pos[a], pos[b]])
                  for grp in (rphi, rz)
                  for ai, a in enumerate(grp) for b in grp[ai + 1:]]
        per_cell.append({"cell": int(c), "n": int(m.sum()),
                         "max_cross": float(np.max(cross)),
                         "median_cross": float(np.median(cross)),
                         "median_within": float(np.median(within))})
    if not per_cell:
        return {"n_cells": 0}

    worst = max(p["max_cross"] for p in per_cell)
    out = {"n_cells": len(per_cell),
           "max_cross_block_feature_corr": worst,
           "median_cross_block_feature_corr":
               float(np.median([p["median_cross"] for p in per_cell])),
           "median_within_block_feature_corr":
               float(np.median([p["median_within"] for p in per_cell])),
           "separable": bool(worst < 0.15),
           "rphi_features": [names[i] for i in rphi],
           "rz_features": [names[i] for i in rz]}
    # the single worst pair, named
    Rall = weighted_corr(y[:, idx], w)
    best = max(((abs(Rall[pos[i], pos[j]]), names[i], names[j])
                for i in rphi for j in rz), key=lambda t: t[0])
    out["worst_pair"] = {"a": best[1], "b": best[2], "abs_corr": float(best[0])}
    return out


def correction_noise_ratio(y_mc, y_corr, w_mc, y_data, w_data,
                           feature_indices=None, param="logsigma_corr",
                           w_mc_reweighted=None):
    """
    How big is the per-track correction compared with the disagreement it is
    meant to fix?

    dy = y_corr - y_mc has a systematic part (the shift the feature needs) and a
    track-to-track spread. Reported per feature:

      shift    |median dy|
      scatter  half the 16-84 range of dy
      ratio    scatter / max(shift, conditional discrepancy)

    IMPORTANT -- two limitations, both learned the hard way:

    1. The denominator MUST be the discrepancy at fixed context. Using the
       inclusive marginal instead inflates it by the data/MC kinematic spectrum
       mismatch, which for sigma_phi is a factor ten and flips the verdict.
       Pass `w_mc_reweighted` (from context_weights) to get this right; without
       it the ratio is only indicative.

    2. Even then this is NOT a reliable predictor of whether a feature will
       improve. A large spread is not necessarily noise: different tracks
       genuinely need different corrections, and the features with the biggest
       real mismodelling (sigma_dxy, sigma_dsz) need large structured
       corrections. Measured on real data, features with ratios above 2 have
       both improved threefold and got worse. Separating legitimate
       track-to-track variation from noise requires comparing two
       independently trained correctors on the same tracks, not an interquartile
       range. Treat this table as descriptive, not as a decision rule.
    """
    names = F.feature_names(param)
    idx = list(range(len(names))) if feature_indices is None else list(feature_indices)
    out = {}
    for k in idx:
        dy = y_corr[:, k] - y_mc[:, k]
        lo, hi = np.percentile(dy, [16, 84])
        scatter = 0.5 * (hi - lo)
        shift = abs(float(np.median(dy)))
        w1_marg = weighted_w1(y_mc[:, k], w_mc, y_data[:, k], w_data)
        if w_mc_reweighted is not None:
            w1_cond = weighted_w1(y_mc[:, k], w_mc_reweighted,
                                  y_data[:, k], w_data)
        else:
            w1_cond = float("nan")
        signal = w1_cond if np.isfinite(w1_cond) else w1_marg
        denom = max(shift, signal, 1e-9)
        r = scatter / denom
        out[names[k]] = {
            "median_shift": float(np.median(dy)),
            "scatter_1sigma": float(scatter),
            "w1_before_marginal": float(w1_marg),
            "w1_before_conditional": float(w1_cond),
            "scatter_over_signal": float(r),
            "signal_used": "conditional" if np.isfinite(w1_cond) else "marginal"}
    return out


def plot_binned_marginals(y_mc, w_mc, C_mc, y_corr,
                          y_data, w_data, C_data,
                          edges, context_names, feature_indices=None,
                          path="marginals_binned.pdf", min_count=200,
                          param="logsigma_corr", max_cells=64):
    """
    One page per feature; on each page, a grid of context cells showing
    data / MC / corrected MC.

    This is the visual form of binned_closure(). Within a cell the kinematics
    of data and MC are matched, so what you see is the flow's residual error
    rather than the spectrum difference that dominates the inclusive overlay.
    Panel titles carry the cell's W1 before -> after, and go red when the
    correction made that cell worse.

    The grid is laid out with the LAST context dimension along columns and all
    earlier dimensions folded into rows, so a 2-D context reads as a natural
    (dim0 x dim1) matrix.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    names = F.feature_names(param)
    idx = list(range(len(names))) if feature_indices is None else list(feature_indices)

    im, shape = assign_bins(C_mc, edges)
    idd, _ = assign_bins(C_data, edges)
    ncell = int(np.prod(shape))
    if ncell > max_cells:
        raise ValueError(f"{ncell} context cells exceeds max_cells={max_cells}; "
                         f"use coarser --closure-bins for the plots")

    # Grid layout: ignore context dimensions that have only one bin, otherwise
    # a trailing singleton (e.g. --plot-bins 3,3,1,1) collapses the grid to a
    # single column. Columns follow the LAST non-singleton dimension.
    multi = [d for d, s in enumerate(shape) if s > 1]
    if len(multi) >= 2:
        ncols = shape[multi[-1]]
    elif len(multi) == 1:
        ncols = int(np.ceil(np.sqrt(ncell)))
    else:
        ncols = 1
    ncols = max(1, min(ncols, ncell))
    nrows = int(np.ceil(ncell / ncols))

    # cell occupancy, so sparse cells are drawn but visibly marked
    counts = [(int((im == c).sum()), int((idd == c).sum())) for c in range(ncell)]

    with PdfPages(path) as pdf:
        for k in idx:
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(2.6 * ncols, 2.3 * nrows),
                                     squeeze=False)
            for c in range(nrows * ncols):
                ax = axes[c // ncols][c % ncols]
                if c >= ncell:
                    ax.axis("off")
                    continue
                nm, nd = counts[c]
                if nm < min_count or nd < min_count:
                    ax.text(0.5, 0.5, f"n={nm}/{nd}\ntoo few",
                            ha="center", va="center", fontsize=7,
                            color="0.6", transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(cell_label(c, shape, context_names, edges),
                                 fontsize=5.5, color="0.6")
                    continue
                sm, sd = im == c, idd == c
                lo, hi = _wq(y_data[sd, k], w_data[sd], [0.005, 0.995])
                if not np.isfinite(lo) or hi <= lo:
                    lo, hi = float(y_data[sd, k].min()), float(y_data[sd, k].max())
                bins = np.linspace(lo, hi, 30)
                ax.hist(y_data[sd, k], bins=bins, weights=w_data[sd], density=True,
                        histtype="stepfilled", alpha=0.35, color="C0")
                ax.hist(y_mc[sm, k], bins=bins, weights=w_mc[sm], density=True,
                        histtype="step", lw=1.1, color="C1")
                ax.hist(y_corr[sm, k], bins=bins, weights=w_mc[sm], density=True,
                        histtype="step", lw=1.1, color="C2")
                wb = weighted_w1(y_mc[sm, k], w_mc[sm], y_data[sd, k], w_data[sd])
                wa = weighted_w1(y_corr[sm, k], w_mc[sm], y_data[sd, k], w_data[sd])
                worse = wa > 1.1 * wb
                ax.set_title(f"{cell_label(c, shape, context_names, edges)}\n"
                             f"W1 {wb:.3f}$\\to${wa:.3f}  n={nm}/{nd}",
                             fontsize=5.5, color=("firebrick" if worse else "black"))
                ax.tick_params(labelsize=5)
            # one legend for the page
            handles = [plt.Line2D([], [], color="C0", lw=6, alpha=0.35, label="data"),
                       plt.Line2D([], [], color="C1", lw=1.5, label="MC"),
                       plt.Line2D([], [], color="C2", lw=1.5, label="MC corr")]
            fig.legend(handles=handles, loc="upper right", fontsize=8, ncol=3)
            fig.suptitle(names[k], fontsize=12, y=0.999)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)
    return path


def plot_marginals(y_mc, w_mc, y_data, w_data, y_corr, param="logsigma_corr",
                   path="marginals.pdf"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = F.feature_names(param)
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for k, ax in enumerate(axes.ravel()):
        lo, hi = _wq(y_data[:, k], w_data, [0.005, 0.995])
        bins = np.linspace(lo, hi, 60)
        ax.hist(y_data[:, k], bins=bins, weights=w_data, density=True,
                histtype="stepfilled", alpha=0.35, label="data")
        ax.hist(y_mc[:, k], bins=bins, weights=w_mc, density=True,
                histtype="step", lw=1.5, label="MC")
        ax.hist(y_corr[:, k], bins=bins, weights=w_mc, density=True,
                histtype="step", lw=1.5, ls="--", label="MC corr")
        ax.set_title(names[k], fontsize=9)
        if k == 0:
            ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_correction_size(y_mc, y_corr, param="logsigma_corr",
                         path="correction_size.pdf"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = F.feature_names(param)
    d = y_corr - y_mc
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for k, ax in enumerate(axes.ravel()):
        lo, hi = np.percentile(d[:, k], [0.5, 99.5])
        ax.hist(d[:, k], bins=np.linspace(lo, hi, 60), histtype="stepfilled", alpha=0.6)
        ax.set_title(f"{names[k]}  (dy)", fontsize=9)
        ax.axvline(0, color="k", lw=0.6)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path