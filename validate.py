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
