"""
covflow.data
============

Turn ROOT ntuples (or a synthetic toy) into the three arrays the flows need:

    X    (N, 15)  packed covariance elements, in features.PACK_NAMES order
    C    (N, k)   context / conditioning variables
    w    (N,)     per-track weights (sWeights for data, gen weight or 1 for MC)

plus a shared Standardiser fitted once on the pooled sample so MC and data live
in the same standardised frame -- that is what makes the latent codes of the two
flows directly comparable in the morph.

The loader copes with both ntuple layouts in Bmmm:
  * one-candidate-per-event with fixed roles (mu1_/mu2_/mu3_ scalar branches);
  * per-track vectors (trk_cov_* as std::vector<float>).
It reads every requested branch as an awkward array, broadcasts the context to
the covariance's jagged structure, flattens, and hands back flat numpy. Scalar
branches are just the length-1 case of that, so the same path serves both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

from . import features as F


# ---------------------------------------------------------------------------
# shared standardiser
# ---------------------------------------------------------------------------

@dataclass
class Standardiser:
    """Affine (x - mu)/sd applied column-wise, with a JSON round-trip.

    One instance is shared by both flows and by the morph so the correction is
    defined in a single, reproducible frame.
    """
    feat_mean: list
    feat_std: list
    ctx_mean: list
    ctx_std: list

    @classmethod
    def fit(cls, X, C, w=None):
        X = np.asarray(X, float)
        C = np.asarray(C, float)
        if w is None:
            fm, fs = X.mean(0), X.std(0)
            cm, cs = C.mean(0), C.std(0)
        else:
            w = np.asarray(w, float)
            wpos = np.clip(w, 0.0, None)
            sw = wpos.sum()
            fm = (wpos[:, None] * X).sum(0) / sw
            fs = np.sqrt((wpos[:, None] * (X - fm) ** 2).sum(0) / sw)
            cm = (wpos[:, None] * C).sum(0) / sw
            cs = np.sqrt((wpos[:, None] * (C - cm) ** 2).sum(0) / sw)
        fs = np.where(fs < 1e-12, 1.0, fs)
        cs = np.where(cs < 1e-12, 1.0, cs)
        return cls(fm.tolist(), fs.tolist(), cm.tolist(), cs.tolist())

    def x(self, X):
        return (np.asarray(X, float) - self.feat_mean) / self.feat_std

    def x_inv(self, Xs):
        return np.asarray(Xs, float) * self.feat_std + self.feat_mean

    def c(self, C):
        return (np.asarray(C, float) - self.ctx_mean) / self.ctx_std

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(**json.load(fh))


# ---------------------------------------------------------------------------
# sample container
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    X: np.ndarray          # (N,15) packed covariance
    C: np.ndarray          # (N,k)  context
    w: np.ndarray          # (N,)   weight
    context_names: list
    neg_weight_fraction: float = 0.0

    def __len__(self):
        return len(self.X)


# ---------------------------------------------------------------------------
# ROOT loading
# ---------------------------------------------------------------------------

def load_root(paths,
              tree: str,
              cov_prefix: str,
              context_branches: Sequence[str],
              weight_branch: str | None = None,
              log_pt_branch: str | None = None,
              clip_negative_weights: bool = True,
              max_events: int | None = None) -> Sample:
    """
    Load a Sample from one or more ROOT files.

    cov_prefix        : e.g. "trk_cov_" or "mu3_cov_"; the 15 branches are
                        cov_prefix + name for name in features.PACK_NAMES.
    context_branches  : list of branch names used as conditioning variables.
    weight_branch     : sWeight (data) or gen weight (MC); None -> unit weights.
    log_pt_branch     : if given, that context branch is replaced by log(pt);
                        keep the name in context_branches, pass it here too.
    """
    import uproot
    import awkward as ak

    cov_branches = [cov_prefix + n for n in F.PACK_NAMES]
    want = list(cov_branches) + list(context_branches)
    if weight_branch:
        want.append(weight_branch)

    parts = []
    files = [paths] if isinstance(paths, str) else list(paths)
    for fp in files:
        with uproot.open(fp) as fh:
            t = fh[tree]
            arr = t.arrays(want, library="ak", how=dict)
            missing = [b for b in want if b not in t]
            if missing:
                near = [k for k in t.keys() if k.startswith(cov_prefix.rstrip("_"))]
                raise KeyError(
                    f"{len(missing)} branch(es) not in {fp}:{tree}, first: {missing[:3]}\n"
                    f"branches starting with {cov_prefix.rstrip('_')!r}: {near[:20]}"
                )
            arr = t.arrays(want, library="ak", how=dict)
            parts.append(arr)
    data = {k: ak.concatenate([p[k] for p in parts]) for k in want}

    # broadcast context (and weight) to the covariance's structure, then flatten
    ref = data[cov_branches[0]]
    is_jagged = ref.ndim > 1

    def flat(name):
        a = data[name]
        if is_jagged:
            a, _ = ak.broadcast_arrays(a, ref)
            a = ak.flatten(a)
        return ak.to_numpy(a).astype(np.float64)

    X = np.stack([flat(b) for b in cov_branches], axis=1)

    C_cols, cnames = [], []
    for b in context_branches:
        col = flat(b)
        if log_pt_branch is not None and b == log_pt_branch:
            col = np.log(np.clip(col, 1e-6, None))
            cnames.append("log_" + b)
        else:
            cnames.append(b)
        C_cols.append(col)
    C = np.stack(C_cols, axis=1)

    if weight_branch:
        w = flat(weight_branch)
    else:
        w = np.ones(len(X))

    neg = float((w < 0).mean())
    if clip_negative_weights:
        w = np.clip(w, 0.0, None)

    if max_events is not None and len(X) > max_events:
        sel = np.random.default_rng(0).choice(len(X), max_events, replace=False)
        X, C, w = X[sel], C[sel], w[sel]

    # keep only rows whose covariance is finite & PD (drops fit failures)
    good = np.isfinite(X).all(1) & np.isfinite(C).all(1) & np.isfinite(w)
    M = F.packed_to_matrix(X)
    good &= F.is_positive_definite(M)
    if not good.all():
        X, C, w = X[good], C[good], w[good]

    return Sample(X, C, w, cnames, neg)


# ---------------------------------------------------------------------------
# synthetic toy (no ROOT) -- used by the smoke test and by --synthetic
# ---------------------------------------------------------------------------

def make_synthetic(n=200_000, seed=0,
                   sigma_dxy_data_scale=1.10,
                   dpcorr_phi_lambda=0.15,
                   eta_slope=0.20):
    """
    Build a toy (mc, data) pair differing by a *known* amount so the whole
    chain can be validated end to end:

      * sigma_dxy is `sigma_dxy_data_scale` x larger in data,
      * the phi-lambda partial correlation is shifted by `dpcorr_phi_lambda`
        in data,
      * plus an eta-dependent difference so the flow has to use the context.

    Context is [pt, eta, nValidHits, nPV]. Returns two Samples.
    """
    rng = np.random.default_rng(seed)

    def draw(n_, is_data):
        pt = np.exp(rng.normal(1.0, 0.5, n_))              # ~ few GeV
        eta = rng.uniform(-2.4, 2.4, n_)
        nhit = rng.integers(8, 25, n_).astype(float)
        npv = rng.integers(10, 60, n_).astype(float)

        # base log-sigmas depend on kinematics/hits (more hits -> tighter)
        abseta = np.abs(eta)
        ls = np.empty((n_, 5))
        ls[:, 0] = -6.5 + 0.15 * abseta - 0.02 * nhit + 0.05 * rng.standard_normal(n_)  # qoverp
        ls[:, 1] = -4.0 + 0.10 * abseta - 0.02 * nhit + 0.05 * rng.standard_normal(n_)  # lambda
        ls[:, 2] = -4.2 + 0.10 * abseta - 0.02 * nhit + 0.05 * rng.standard_normal(n_)  # phi
        ls[:, 3] = -3.0 + 0.30 * abseta - 0.03 * nhit + 0.08 * rng.standard_normal(n_)  # dxy
        ls[:, 4] = -3.0 + 0.30 * abseta - 0.03 * nhit + 0.08 * rng.standard_normal(n_)  # dsz

        # canonical partial correlations (atanh space), mild, context-dependent
        z = 0.15 * rng.standard_normal((n_, 10))
        # index of pcorr_phi_lambda in LOWER_PAIRS: (i=2 phi, j=1 lambda)
        k_pl = F.LOWER_PAIRS.index((2, 1))
        z[:, k_pl] += 0.30 + 0.10 * eta            # a real correlation with eta structure

        if is_data:
            ls[:, 3] += np.log(sigma_dxy_data_scale)          # sigma_dxy bigger in data
            ls[:, 3] += eta_slope * (abseta > 1.5)            # eta-dependent extra
            z[:, k_pl] += np.arctanh(np.clip(np.tanh(z[:, k_pl]) + dpcorr_phi_lambda, -0.99, 0.99)) \
                          - z[:, k_pl]                         # shift the correlation by ~0.15

        y = np.concatenate([ls, np.arctanh(np.tanh(z))], axis=1)
        M = F.features_to_matrix(y)
        X = F.matrix_to_packed(M)
        C = np.stack([pt, eta, nhit, npv], axis=1)
        w = np.ones(n_)
        return Sample(X, C, w, ["pt", "eta", "nValidHits", "nPV"], 0.0)

    return draw(n, False), draw(n, True)
