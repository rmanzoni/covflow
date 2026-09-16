"""
covflow.features
================

The bijection between a track's 5x5 curvilinear covariance matrix and an
*unconstrained* vector in R^15 that the normalising flow actually sees.

Parameter order is fixed by CMSSW (reco::TrackBase::TrackParameters) and MUST
match TrackCovUtils.h / TrackCovFeatures.h exactly:

    0 = qoverp   1 = lambda   2 = phi   3 = dxy   4 = dsz

Packed 15-vector order (i <= j, row-major upper triangle), identical to
trkcov::covIndexPairs():

    (0,0)(0,1)(0,2)(0,3)(0,4)(1,1)(1,2)(1,3)(1,4)(2,2)(2,3)(2,4)(3,3)(3,4)(4,4)

Feature vector y in R^15 (what the flow transports):

    y[0:5]  = log sigma_i                 sigma_i = sqrt(Sigma_ii)
    y[5:15] = atanh(canonical partial correlations)   ["logsigma_corr", default]

The 10 off-diagonal features are the *canonical partial correlations* (CPCs) of
the correlation matrix, in the C-vine / Lewandowski-Kurowicka-Joe sense, read
off the Cholesky factor of R. Each CPC lives in (-1, 1); atanh sends it to R.
The map is a bijection R^15 -> {PD 5x5 covariances}, so NO output of the network
can ever be an invalid covariance -- positive-definiteness is structural, not a
constraint the flow has to learn.

CPC order matches the strictly-lower-triangular (i > j) row-major order of L,
which we name pcorr_<param_i>_<param_j>:

    (1,0)(2,0)(2,1)(3,0)(3,1)(3,2)(4,0)(4,1)(4,2)(4,3)

An alternative "log_cholesky" parameterisation is provided for cross-checks; it
is simpler but its off-diagonals are unbounded relative to the scale, so a flow
that wanders can produce matrices that are PD on paper yet numerically singular.
Prefer the default.

Everything here is plain numpy and vectorised over a leading batch axis, so the
same code path handles one track or ten million. A C++ mirror lives in
cpp/TrackCovFeatures.h and is checked for parity.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Fixed conventions
# ---------------------------------------------------------------------------

PARAM_NAMES = ("qoverp", "lambda", "phi", "dxy", "dsz")
DIM = 5

# Packed 15-vector order (i <= j), == trkcov::covIndexPairs().
PACK_PAIRS = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 1), (1, 2), (1, 3), (1, 4),
    (2, 2), (2, 3), (2, 4),
    (3, 3), (3, 4),
    (4, 4),
]
PACK_NAMES = [f"{PARAM_NAMES[i]}_{PARAM_NAMES[j]}" for (i, j) in PACK_PAIRS]

# Strictly-lower-triangular (i > j) order for the CPCs / Cholesky off-diagonals.
LOWER_PAIRS = [(i, j) for i in range(DIM) for j in range(i)]  # 10 pairs
PCORR_NAMES = [f"pcorr_{PARAM_NAMES[i]}_{PARAM_NAMES[j]}" for (i, j) in LOWER_PAIRS]

# Human-readable names of the 15 flow features, in y order.
FEATURE_NAMES = [f"log_sigma_{p}" for p in PARAM_NAMES] + PCORR_NAMES

N_FEATURES = 15
_EPS = 1.0e-12
# Clamp for tanh-derived partial correlations. A CPC of exactly +-1 makes the
# correlation matrix singular; capping at 1 - 1e-6 keeps 1 - z^2 >= 2e-6 so the
# reconstructed matrix stays comfortably numerically PD, while still allowing
# correlations up to 0.999999 -- far beyond anything physical for track params.
_CPC_CLAMP = 1.0 - 1.0e-6


# ---------------------------------------------------------------------------
# packed 15 <-> full symmetric 5x5
# ---------------------------------------------------------------------------

def packed_to_matrix(packed: np.ndarray) -> np.ndarray:
    """(..., 15) packed upper-triangle -> (..., 5, 5) symmetric matrix."""
    packed = np.asarray(packed, dtype=np.float64)
    batch = packed.shape[:-1]
    M = np.zeros(batch + (DIM, DIM), dtype=np.float64)
    for k, (i, j) in enumerate(PACK_PAIRS):
        M[..., i, j] = packed[..., k]
        M[..., j, i] = packed[..., k]
    return M


def matrix_to_packed(M: np.ndarray) -> np.ndarray:
    """(..., 5, 5) symmetric -> (..., 15) packed upper-triangle."""
    M = np.asarray(M, dtype=np.float64)
    batch = M.shape[:-2]
    out = np.empty(batch + (N_FEATURES,), dtype=np.float64)
    for k, (i, j) in enumerate(PACK_PAIRS):
        out[..., k] = M[..., i, j]
    return out


# ---------------------------------------------------------------------------
# default parameterisation: log-sigma + canonical partial correlations
# ---------------------------------------------------------------------------

def matrix_to_features(M: np.ndarray) -> np.ndarray:
    """
    (..., 5, 5) PD covariance -> (..., 15) unconstrained features.

        y[0:5]  = log sqrt(diag)
        y[5:15] = atanh(CPC), read from the Cholesky factor of the correlation.
    """
    M = np.asarray(M, dtype=np.float64)
    var = np.diagonal(M, axis1=-2, axis2=-1)
    if np.any(var <= 0):
        raise ValueError("non-positive variance on the diagonal")
    sigma = np.sqrt(var)
    log_sigma = np.log(sigma)

    # correlation matrix R = D^-1 M D^-1
    inv_sigma = 1.0 / sigma
    R = M * inv_sigma[..., :, None] * inv_sigma[..., None, :]
    # symmetrise against round-off, unit diagonal
    R = 0.5 * (R + np.swapaxes(R, -1, -2))
    idx = np.arange(DIM)
    R[..., idx, idx] = 1.0

    L = np.linalg.cholesky(R)  # lower-triangular, positive diagonal

    # Recover CPCs row by row from L. For row i with running product
    # P_j = prod_{k<j} sqrt(1 - z_ik^2), P_0 = 1:
    #   L[i,j] = z_ij * P_j  (j < i),   L[i,i] = P_i
    # so z_ij = L[i,j] / P_j, then P_{j+1} = P_j * sqrt(1 - z_ij^2).
    batch = M.shape[:-2]
    z = np.empty(batch + (len(LOWER_PAIRS),), dtype=np.float64)
    for i in range(1, DIM):
        P = np.ones(batch, dtype=np.float64)
        for j in range(i):
            zij = L[..., i, j] / np.clip(P, _EPS, None)
            zij = np.clip(zij, -_CPC_CLAMP, _CPC_CLAMP)
            k = LOWER_PAIRS.index((i, j))
            z[..., k] = zij
            P = P * np.sqrt(np.clip(1.0 - zij * zij, _EPS, None))

    atanh_z = np.arctanh(z)
    return np.concatenate([log_sigma, atanh_z], axis=-1)


def features_to_matrix(y: np.ndarray) -> np.ndarray:
    """
    (..., 15) unconstrained features -> (..., 5, 5) PD covariance.
    Exact inverse of matrix_to_features up to the CPC clamp.
    """
    y = np.asarray(y, dtype=np.float64)
    log_sigma = y[..., :DIM]
    atanh_z = y[..., DIM:]
    sigma = np.exp(log_sigma)
    z = np.tanh(atanh_z)
    z = np.clip(z, -_CPC_CLAMP, _CPC_CLAMP)

    batch = y.shape[:-1]
    L = np.zeros(batch + (DIM, DIM), dtype=np.float64)
    L[..., 0, 0] = 1.0
    for i in range(1, DIM):
        P = np.ones(batch, dtype=np.float64)
        for j in range(i):
            k = LOWER_PAIRS.index((i, j))
            zij = z[..., k]
            L[..., i, j] = zij * P
            P = P * np.sqrt(np.clip(1.0 - zij * zij, _EPS, None))
        L[..., i, i] = P

    R = L @ np.swapaxes(L, -1, -2)
    D = sigma[..., :, None] * sigma[..., None, :]
    return R * D


# ---------------------------------------------------------------------------
# alternative parameterisation: log-Cholesky of the full covariance
# ---------------------------------------------------------------------------

def matrix_to_features_logchol(M: np.ndarray) -> np.ndarray:
    """
    (..., 5, 5) PD covariance -> (..., 15) features, log-Cholesky variant.

        y[0:5]  = log(diag(L))           (Cholesky of the *covariance*)
        y[5:15] = strictly-lower entries of L, raw (LOWER_PAIRS order)
    """
    M = np.asarray(M, dtype=np.float64)
    L = np.linalg.cholesky(M)
    idx = np.arange(DIM)
    log_diag = np.log(np.clip(L[..., idx, idx], _EPS, None))
    batch = M.shape[:-2]
    off = np.empty(batch + (len(LOWER_PAIRS),), dtype=np.float64)
    for k, (i, j) in enumerate(LOWER_PAIRS):
        off[..., k] = L[..., i, j]
    return np.concatenate([log_diag, off], axis=-1)


def features_to_matrix_logchol(y: np.ndarray) -> np.ndarray:
    """Inverse of matrix_to_features_logchol."""
    y = np.asarray(y, dtype=np.float64)
    log_diag = y[..., :DIM]
    off = y[..., DIM:]
    batch = y.shape[:-1]
    L = np.zeros(batch + (DIM, DIM), dtype=np.float64)
    idx = np.arange(DIM)
    L[..., idx, idx] = np.exp(log_diag)
    for k, (i, j) in enumerate(LOWER_PAIRS):
        L[..., i, j] = off[..., k]
    return L @ np.swapaxes(L, -1, -2)


# ---------------------------------------------------------------------------
# dispatch by name so the rest of the package is parameterisation-agnostic
# ---------------------------------------------------------------------------

_PARAM = {
    "logsigma_corr": (matrix_to_features, features_to_matrix),
    "log_cholesky": (matrix_to_features_logchol, features_to_matrix_logchol),
}


def get_transforms(name: str = "logsigma_corr"):
    """Return (matrix_to_features, features_to_matrix) for a named scheme."""
    if name not in _PARAM:
        raise KeyError(f"unknown parameterisation {name!r}; "
                       f"choose from {list(_PARAM)}")
    return _PARAM[name]


def packed_to_features(packed, name: str = "logsigma_corr",
                       out_dtype=np.float64, chunk: int = 1_000_000):
    """(N,15) packed covariance -> (N,15) features, in chunks.

    Identical to `matrix_to_features(packed_to_matrix(packed))`, except that
    the (N,5,5) float64 intermediate -- 200 bytes per track, the largest
    transient in the whole pipeline -- is capped at `chunk` rows instead of
    being materialised for the full sample.

    The arithmetic stays float64 regardless of `out_dtype`: matrix_to_features
    takes a Cholesky factor of the correlation matrix, and these covariances
    span several orders of magnitude on the diagonal. Only the RESULT is cast,
    and the features are O(1) quantities (log sigma, atanh of a partial
    correlation), so float32 there costs ~1e-7 -- four orders of magnitude
    below the smallest discrepancy the flows are asked to resolve.
    """
    to_feat, _ = get_transforms(name)
    packed = np.asarray(packed)
    out = np.empty((len(packed), N_FEATURES), dtype=np.dtype(out_dtype))
    for s in range(0, len(packed), int(chunk)):
        e = min(s + int(chunk), len(packed))
        out[s:e] = to_feat(packed_to_matrix(packed[s:e]))
    return out


def feature_names(name: str = "logsigma_corr"):
    if name == "logsigma_corr":
        return list(FEATURE_NAMES)
    if name == "log_cholesky":
        return [f"logL_{p}_{p}" for p in PARAM_NAMES] + \
               [f"L_{PARAM_NAMES[i]}_{PARAM_NAMES[j]}" for (i, j) in LOWER_PAIRS]
    raise KeyError(name)


# ---------------------------------------------------------------------------
# feature subsets: which dimensions the flows are actually trained in
# ---------------------------------------------------------------------------

# NOTE the difference from `active_features` in correct.morph():
#   * a SUBSET here changes the dimensionality of the flows themselves -- they
#     are built, trained and inverted in that subspace only. Features outside
#     the subset are copied from MC untouched.
#   * `active_features` trains a full 15-D flow and then masks which components
#     of the *output* are kept.
# For "correct only the diagonal" you almost always want the subset: a 5-D flow
# has far more capacity per dimension than a 15-D one at equal size, and the
# sharply-peaked partial-correlation features -- which are the hardest for a
# spline flow to resolve -- are removed from the problem entirely.

# ---------------------------------------------------------------------------
# r-phi / r-z block structure
# ---------------------------------------------------------------------------
#
# A CMS helix fit is driven by two nearly independent measurements:
#   r-phi (bending plane) constrains qoverp, phi, dxy
#   r-z   (longitudinal)  constrains lambda, dsz
# Measured on 2022C data and Bc MC (400k tracks each): within-block |partial
# correlation| >= 0.288, cross-block <= 0.0018 -- a factor ~160 separation,
# stable across every pt, |eta| and nPV bin. So of the 10 partial correlations
# only 4 carry information; the other 6 are zero to within a part in 500.

PARAM_BLOCKS = {"r-phi": (0, 2, 3), "r-z": (1, 4)}


def param_block(i):
    for name, members in PARAM_BLOCKS.items():
        if i in members:
            return name
    raise KeyError(i)


def pair_within_block(i, j):
    return param_block(i) == param_block(j)


# feature indices of the 4 within-block partial correlations
WITHIN_BLOCK_PCORR = [DIM + k for k, (i, j) in enumerate(LOWER_PAIRS)
                      if pair_within_block(i, j)]
CROSS_BLOCK_PCORR = [DIM + k for k, (i, j) in enumerate(LOWER_PAIRS)
                     if not pair_within_block(i, j)]


def _block_subset(block):
    """log-sigmas of a block's parameters plus its internal correlations."""
    members = PARAM_BLOCKS[block]
    idx = list(members)
    for k, (i, j) in enumerate(LOWER_PAIRS):
        if i in members and j in members:
            idx.append(DIM + k)
    return sorted(idx)


SUBSETS = {
    "all":  list(range(N_FEATURES)),      # 15: scales + all correlations
    "diag": list(range(DIM)),             #  5: log sigma only
    "corr": list(range(DIM, N_FEATURES)), # 10: partial correlations only
    # 9: everything that carries information. Drops the 6 cross-block partial
    # correlations, which are ~0 in data AND MC, so correcting them is pure
    # noise-fitting -- and they are the sharply peaked features a spline flow
    # resolves worst.
    "block": sorted(list(range(DIM)) + WITHIN_BLOCK_PCORR),
    "rphi": _block_subset("r-phi"),       # 6: qoverp/phi/dxy scales + 3 corr
    "rz":   _block_subset("r-z"),         # 3: lambda/dsz scales + 1 corr
}


def subset_indices(spec):
    """
    Resolve a subset spec to a list of feature indices.

    `spec` is either a key of SUBSETS ('all', 'diag', 'corr') or an explicit
    iterable of integer indices.
    """
    if spec is None:
        return list(range(N_FEATURES))
    if isinstance(spec, str):
        if spec not in SUBSETS:
            raise KeyError(f"unknown feature subset {spec!r}; "
                           f"choose from {list(SUBSETS)} or give explicit indices")
        return list(SUBSETS[spec])
    idx = [int(i) for i in spec]
    if not idx:
        raise ValueError("empty feature subset")
    bad = [i for i in idx if not 0 <= i < N_FEATURES]
    if bad:
        raise ValueError(f"feature indices out of range [0,{N_FEATURES}): {bad}")
    if len(set(idx)) != len(idx):
        raise ValueError(f"duplicate feature indices: {idx}")
    return idx


def subset_names(spec, param="logsigma_corr"):
    names = feature_names(param)
    return [names[i] for i in subset_indices(spec)]


# ---------------------------------------------------------------------------
# validity helpers
# ---------------------------------------------------------------------------

def is_positive_definite(M: np.ndarray) -> np.ndarray:
    """Boolean (...,) mask: True where the (...,5,5) matrix is PD."""
    M = np.asarray(M, dtype=np.float64)
    Msym = 0.5 * (M + np.swapaxes(M, -1, -2))
    try:
        w = np.linalg.eigvalsh(Msym)
    except np.linalg.LinAlgError:
        flat = Msym.reshape(-1, DIM, DIM)
        ok = np.zeros(flat.shape[0], dtype=bool)
        for n in range(flat.shape[0]):
            try:
                ok[n] = np.all(np.linalg.eigvalsh(flat[n]) > 0)
            except np.linalg.LinAlgError:
                ok[n] = False
        return ok.reshape(M.shape[:-2])
    return np.all(w > 0, axis=-1)


def roundtrip_error(packed: np.ndarray, name: str = "logsigma_corr") -> float:
    """
    Max relative error of packed -> matrix -> features -> matrix -> packed.
    A number this small (<~1e-8) means the parameterisation is invertible on
    *these* matrices; it is the cheapest of all the validation checks.
    """
    to_feat, to_mat = get_transforms(name)
    M0 = packed_to_matrix(packed)
    y = to_feat(M0)
    M1 = to_mat(y)
    p0 = matrix_to_packed(M0)
    p1 = matrix_to_packed(M1)
    denom = np.maximum(np.abs(p0), _EPS)
    return float(np.max(np.abs(p1 - p0) / denom))
