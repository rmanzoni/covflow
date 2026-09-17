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
from dataclasses import dataclass, asdict, field
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

    @staticmethod
    def _wd(A):
        """Array plus the dtype to work in: float32 in, float32 out.

        `fit` stays float64 -- the moments are a reduction over the whole
        sample and cost nothing. But x/x_inv/c run on (N,15) arrays several
        times per job, and forcing float64 there turned every one of them into
        a full-size float64 allocation even when the caller held float32.
        Standardising is (x-mu)/sd with both O(1); float32 costs ~1e-7 and
        keeps the array zero-copy on the way into torch, which takes float32
        anyway.
        """
        A = np.asarray(A)
        return A, (A.dtype if A.dtype == np.float32 else np.dtype(np.float64))

    def x(self, X):
        X, dt = self._wd(X)
        return (X - np.asarray(self.feat_mean, dt)) / np.asarray(self.feat_std, dt)

    def x_inv(self, Xs):
        Xs, dt = self._wd(Xs)
        return Xs * np.asarray(self.feat_std, dt) + np.asarray(self.feat_mean, dt)

    def c(self, C):
        C, dt = self._wd(C)
        return (C - np.asarray(self.ctx_mean, dt)) / np.asarray(self.ctx_std, dt)

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
    n_read: int = 0         # rows read from file, before any cut
    n_after_selection: int = 0
    selection: str | None = None
    extra: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.X)

    def cutflow(self):
        return (f"read {self.n_read:,} -> selection {self.n_after_selection:,}"
                f" -> clean/PD {len(self):,}")


# ---------------------------------------------------------------------------
# ROOT loading
# ---------------------------------------------------------------------------

def selection_branches(expr: str) -> list:
    """
    Branch names referenced by a selection expression, found by walking the AST.

    Auto-extracting rather than asking for the list separately means the two can
    never drift apart. Attribute access (`a.b`) and function names are ignored,
    so `abs(mu1_dxy) < 0.05 & (mu1_pt > 3)` yields ['mu1_dxy', 'mu1_pt'].
    """
    import ast
    tree = ast.parse(expr, mode="eval")

    # '&' binds TIGHTER than '==' in Python, so
    #     a == True & b == True
    # parses as the chained comparison
    #     (a == (True & b)) and ((True & b) == True)
    # which explodes inside numpy with an unhelpful bitwise_and TypeError.
    # Catch the shape here instead.
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare) and len(n.ops) > 1:
            raise ValueError(
                f"selection {expr!r} contains a chained comparison "
                f"(e.g. 'a == b == c'), which numpy cannot evaluate.\n"
                f"  This is almost always '&' precedence: '&' binds tighter "
                f"than '==', so 'a == True & b == True' does NOT mean what it "
                f"looks like.\n"
                f"  Parenthesise every comparison separately:\n"
                f"    (a == True) & (b == True)\n"
                f"  or, for boolean branches, drop the '== True' entirely:\n"
                f"    a & b")

    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return sorted(names - called - set(_SEL_GLOBALS))


# names available inside a selection expression besides the branches
_SEL_GLOBALS = {
    "abs": np.abs, "sqrt": np.sqrt, "log": np.log, "log10": np.log10,
    "exp": np.exp, "cos": np.cos, "sin": np.sin, "tan": np.tan,
    "arctan2": np.arctan2, "minimum": np.minimum, "maximum": np.maximum,
    "isfinite": np.isfinite, "where": np.where, "np": np, "pi": np.pi,
}


def split_and_terms(expr):
    """Top-level '&' terms of a selection expression, as source strings."""
    import ast
    tree = ast.parse(expr, mode="eval").body
    terms, stack = [], [tree]
    while stack:
        n = stack.pop()
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitAnd):
            stack += [n.left, n.right]
        else:
            terms.append(ast.unparse(n))
    return list(reversed(terms))


def drop_terms_using(expr, branches):
    """
    Remove the top-level '&' terms that reference any of `branches`.

    Used to relax a user selection for the mass fit: if they already cut a
    narrow mass window, fitting inside it would leave no sideband to constrain
    the background. Only top-level terms are removed -- a mass cut buried inside
    an '|' is left alone, and reported, because dropping it would change the
    meaning of the selection.
    """
    keep, dropped = [], []
    for t in split_and_terms(expr):
        (dropped if set(selection_branches(t)) & set(branches) else keep).append(t)
    new = " & ".join(f"({t})" for t in keep) if keep else None
    return new, dropped


class _Reservoir:
    """Uniform random subsample of a stream whose length is not known in advance.

    Algorithm R. Needed because `max_events` has to mean the same thing it
    always did -- a uniform random subsample of the SELECTED sample -- and the
    old implementation got that by materialising the whole sample first, which
    is exactly what this rewrite exists to avoid.

    The per-chunk update is vectorised. Two rows in the same chunk can draw the
    same slot; numpy fancy assignment keeps the last, which is what sequential
    Algorithm R would have done anyway.
    """

    def __init__(self, k, rng):
        self.k = int(k)
        self.rng = rng
        self.n = 0                  # items seen
        self.filled = 0             # items held
        self.buf = None             # dict name -> (k, ...) array

    def add(self, block):
        m = len(next(iter(block.values())))
        if m == 0:
            return
        if self.buf is None:
            self.buf = {name: np.empty((self.k,) + v.shape[1:], v.dtype)
                        for name, v in block.items()}

        take = min(self.k - self.filled, m)
        if take:
            for name, buf in self.buf.items():
                buf[self.filled:self.filled + take] = block[name][:take]
            self.filled += take
            self.n += take

        rest = m - take
        if rest:
            # item t (1-based) replaces a uniformly chosen slot with prob k/t
            t = self.n + 1 + np.arange(rest)
            slot = (self.rng.random(rest) * t).astype(np.int64)
            hit = slot < self.k
            if hit.any():
                rows = np.nonzero(hit)[0] + take
                for name, buf in self.buf.items():
                    buf[slot[hit]] = block[name][rows]
            self.n += rest

    def result(self):
        if self.buf is None:
            return None
        return {name: v[:self.filled] for name, v in self.buf.items()}


def _flatten_chunk(arrays, cov_branches, context_branches, log_pt_branch,
                   weight_branch, sel_branches, extra_branches, selection):
    """One chunk of awkward arrays -> (X, C, w, extra, cnames, n_before_sel).

    All arithmetic here is float64, whatever the caller stores afterwards: the
    PD test takes eigenvalues of a matrix whose diagonal spans several orders of
    magnitude, and that is not a float32 computation.
    """
    import awkward as ak

    ref = arrays[cov_branches[0]]
    is_jagged = ref.ndim > 1

    def flat(name, cast=True):
        """
        Flatten a branch onto the same per-track footing as the covariance.

        cast=True  -> float64, for covariance/context/weight arithmetic.
        cast=False -> keep the native dtype. Selection branches MUST use this:
                      casting a bool ID branch to float64 makes '&' fail with
                      "ufunc 'bitwise_and' not supported for the input types".
        """
        a = arrays[name]
        if is_jagged:
            a, _ = ak.broadcast_arrays(a, ref)
            a = ak.flatten(a)
        out = ak.to_numpy(a)
        return out.astype(np.float64) if cast else out

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

    w = flat(weight_branch) if weight_branch else np.ones(len(X))

    extra = {b: flat(b) for b in extra_branches}
    n_before_sel = len(X)

    if selection:
        env = dict(_SEL_GLOBALS)
        env.update({b: flat(b, cast=False) for b in sel_branches})
        try:
            mask = eval(selection, {"__builtins__": {}}, env)   # noqa: S307
        except ValueError as e:
            if "truth value of an array" in str(e):
                raise ValueError(
                    f"selection {selection!r} uses Python's and/or/not, which "
                    f"cannot act on arrays. Use & | ~ with each comparison "
                    f"parenthesised, e.g. '(mu1_pt > 3) & (abs(mu1_eta) < 2.4)'"
                ) from e
            raise
        mask = np.asarray(mask)
        if mask.dtype != bool:
            raise TypeError(f"selection {selection!r} evaluated to "
                            f"{mask.dtype}, not bool -- remember numpy needs "
                            f"'&'/'|' with parenthesised comparisons, not "
                            f"'and'/'or'")
        if mask.shape != (n_before_sel,):
            raise ValueError(f"selection produced shape {mask.shape}, "
                             f"expected ({n_before_sel},)")
        X, C, w = X[mask], C[mask], w[mask]
        extra = {k: v[mask] for k, v in extra.items()}

    return X, C, w, extra, cnames, n_before_sel


def load_root(paths,
              tree: str,
              cov_prefix: str,
              context_branches: Sequence[str],
              weight_branch: str | None = None,
              log_pt_branch: str | None = None,
              selection: str | None = None,
              extra_branches: Sequence[str] = (),
              clip_negative_weights: bool = True,
              max_events: int | None = None,
              dtype=np.float32,
              chunk_size="512 MB",
              verbose: bool = True) -> Sample:
    """
    Load a Sample from one or more ROOT files, one chunk at a time.

    cov_prefix        : e.g. "trk_cov_" or "mu3_cov_"; the 15 branches are
                        cov_prefix + name for name in features.PACK_NAMES.
    context_branches  : list of branch names used as conditioning variables.
    weight_branch     : sWeight (data) or gen weight (MC); None -> unit weights.
    log_pt_branch     : if given, that context branch is replaced by log(pt);
                        keep the name in context_branches, pass it here too.
    selection         : numpy expression over branch names, e.g.
                        "(mu1_pt > 3) & (abs(mu1_eta) < 2.4)". Branches needed
                        by it are read automatically.
    dtype             : storage dtype of X, C and w (`extra` branches are
                        always float64). float32 halves the
                        resident size and costs nothing downstream -- the flows
                        train in float32 regardless (torch.as_tensor(...,
                        float32)), and features.py upcasts to float64 inside
                        every routine that needs the precision. float64 is
                        there for an A/B check, not for normal running.
    chunk_size        : passed to uproot.iterate. An int is a number of
                        entries, a string like "512 MB" an uncompressed size.

    WHY THIS IS CHUNKED

    The previous implementation called `t.arrays(want)` per file, held every
    result in a list, and concatenated -- so peak memory was several times the
    FULL uncut payload of every branch in every file, and neither `selection`
    nor `max_events` reduced it, because both were applied afterwards. On a
    partial Run 3 sample that is an 8 GB OOM before the first epoch.

    Here the selection and the finite/PD cleaning happen per chunk, so only
    surviving rows are ever accumulated, and `max_events` is honoured by a
    reservoir rather than by subsampling a materialised array.

    Note that `max_events` still requires reading every file end to end: a
    uniform subsample cannot be drawn from a prefix. It caps memory and
    training time, not I/O.

    ONE DELIBERATE SEMANTIC CHANGE: the finite/PD cleaning now runs BEFORE the
    subsample, so `--max-events N` yields N usable tracks rather than N minus
    however many had a failed fit. The subsample is still uniform, but it is a
    different draw from the old code's -- runs are not bit-comparable across
    this change.
    """
    import uproot

    dtype = np.dtype(dtype)
    cov_branches = [cov_prefix + n for n in F.PACK_NAMES]
    sel_branches = selection_branches(selection) if selection else []
    extra_branches = list(extra_branches or [])
    want = list(dict.fromkeys(list(cov_branches) + list(context_branches)
                              + list(sel_branches) + list(extra_branches)
                              + ([weight_branch] if weight_branch else [])))

    files = [paths] if isinstance(paths, str) else list(paths)

    # Branch check up front, on metadata only: a typo should cost one open,
    # not one full pass over the first file.
    for fp in files:
        with uproot.open(fp) as fh:
            have = set(fh[tree].keys())
        missing = [b for b in want if b not in have]
        if missing:
            stem = cov_prefix.rstrip("_")
            near = sorted(k for k in have if k.startswith(stem))[:20]
            raise KeyError(
                f"{len(missing)} branch(es) missing from {fp}:{tree}\n"
                f"  missing (first 5): {missing[:5]}\n"
                f"  branches starting with {stem!r}: {near or 'NONE'}\n"
                f"  (a common cause is a --cov-prefix without its trailing "
                f"underscore)")

    rng = np.random.default_rng(0)
    reservoir = _Reservoir(max_events, rng) if max_events else None
    blocks = []
    n_read = n_after_sel = n_kept = n_neg = 0
    n_bad_cov = 0
    cnames = None

    for fp in files:
        for arrays in uproot.iterate({fp: tree}, expressions=want,
                                     step_size=chunk_size, library="ak"):
            X, C, w, extra, cnames, n_b = _flatten_chunk(
                arrays, cov_branches, context_branches, log_pt_branch,
                weight_branch, sel_branches, extra_branches, selection)
            n_read += n_b
            n_after_sel += len(X)
            if len(X) == 0:
                continue

            n_neg += int((w < 0).sum())

            # finite + PD, in float64, before anything is stored or sampled
            good = np.isfinite(X).all(1) & np.isfinite(C).all(1) & np.isfinite(w)
            good &= F.is_positive_definite(F.packed_to_matrix(X))
            if not good.all():
                n_bad_cov += int((~good).sum())
                X, C, w = X[good], C[good], w[good]
                extra = {k: v[good] for k, v in extra.items()}
            if len(X) == 0:
                continue
            n_kept += len(X)

            block = {"X": X.astype(dtype, copy=False),
                     "C": C.astype(dtype, copy=False),
                     "w": w.astype(dtype, copy=False)}
            # `extra` stays float64 on purpose. It is a passthrough for
            # arbitrary branches -- the sWeight mass today, an event number
            # tomorrow -- and float32 silently mangles integer identifiers
            # above 2^24. It is a handful of columns, so the memory is noise.
            block.update({"extra:" + k: v for k, v in extra.items()})

            if reservoir is not None:
                reservoir.add(block)
            else:
                blocks.append(block)

        if verbose:
            print(f"[load]   {fp}: {n_read:,} rows read, {n_after_sel:,} pass "
                  f"selection, {n_kept:,} usable")

    if reservoir is not None:
        merged = reservoir.result()
        if merged is None:
            raise ValueError("zero rows survived selection and cleaning")
    else:
        if not blocks:
            raise ValueError("zero rows survived selection and cleaning")
        keys = blocks[0].keys()
        merged = {k: np.concatenate([b[k] for b in blocks]) for k in keys}
        blocks.clear()

    if selection and n_after_sel == 0:
        raise ValueError(f"selection {selection!r} kept 0 of {n_read} rows")

    X, C, w = merged["X"], merged["C"], merged["w"]
    extra = {k[len("extra:"):]: v for k, v in merged.items()
             if k.startswith("extra:")}

    neg = float(n_neg / n_after_sel) if n_after_sel else 0.0
    if clip_negative_weights:
        w = np.clip(w, 0.0, None)

    if verbose:
        mb = sum(v.nbytes for v in (X, C, w)) / 2**20
        print(f"[load] {len(X):,} tracks stored as {dtype.name} "
              f"({mb:.0f} MB for X, C, w); {n_bad_cov:,} dropped as "
              f"non-finite or not positive definite")

    return Sample(X, C, w, cnames, neg,
                  n_read=n_read, n_after_selection=n_after_sel,
                  selection=selection, extra=extra)


# ---------------------------------------------------------------------------
# synthetic toy (no ROOT) -- used by the smoke test and by --synthetic
# ---------------------------------------------------------------------------

def make_synthetic(n=200_000, seed=0,
                   sigma_dxy_data_scale=1.10,
                   dpcorr_phi_lambda=0.15,
                   eta_slope=0.20,
                   ctx_shift=0.0,
                   conditional_identical=False):
    """
    Build a toy (mc, data) pair differing by a *known* amount so the whole
    chain can be validated end to end:

      * sigma_dxy is `sigma_dxy_data_scale` x larger in data,
      * the phi-lambda partial correlation is shifted by `dpcorr_phi_lambda`
        in data,
      * plus an eta-dependent difference so the flow has to use the context.

    ctx_shift > 0 additionally gives DATA a different pt/eta spectrum from MC
    (harder pt, more forward), mimicking signal MC vs an inclusive trigger
    stream. Because the covariance depends on kinematics, this alone makes the
    *marginal* distributions differ even when the conditionals are identical.

    conditional_identical=True removes every conditional difference, leaving
    only the spectrum difference. That combination is the null test for the
    context machinery: global metrics should show separation, binned and
    context-reweighted metrics should not.

    Context is [pt, eta, nValidHits, nPV]. Returns two Samples.
    """
    rng = np.random.default_rng(seed)

    def draw(n_, is_data):
        shift = ctx_shift if is_data else 0.0
        pt = np.exp(rng.normal(1.0 + shift, 0.5, n_))       # ~ few GeV
        eta = rng.uniform(-2.4, 2.4, n_)
        if shift:                                           # skew toward forward
            eta = np.sign(eta) * np.abs(eta) ** (1.0 / (1.0 + shift))
        nhit = rng.integers(8, 25, n_).astype(float)
        npv = rng.integers(10, 60, n_).astype(float)

        # base log-sigmas depend on kinematics/hits: sigma grows at low pt
        # (multiple scattering) and forward, and shrinks with more hits.
        abseta = np.abs(eta)
        lpt = np.log(pt)
        ls = np.empty((n_, 5))
        ls[:, 0] = -6.5 + 0.15*abseta - 0.02*nhit - 0.35*lpt + 0.05*rng.standard_normal(n_)
        ls[:, 1] = -4.0 + 0.10*abseta - 0.02*nhit - 0.30*lpt + 0.05*rng.standard_normal(n_)
        ls[:, 2] = -4.2 + 0.10*abseta - 0.02*nhit - 0.30*lpt + 0.05*rng.standard_normal(n_)
        ls[:, 3] = -3.0 + 0.30*abseta - 0.03*nhit - 0.45*lpt + 0.08*rng.standard_normal(n_)
        ls[:, 4] = -3.0 + 0.30*abseta - 0.03*nhit - 0.45*lpt + 0.08*rng.standard_normal(n_)

        # canonical partial correlations (atanh space), mild, context-dependent
        z = 0.15 * rng.standard_normal((n_, 10))
        # index of pcorr_phi_lambda in LOWER_PAIRS: (i=2 phi, j=1 lambda)
        k_pl = F.LOWER_PAIRS.index((2, 1))
        z[:, k_pl] += 0.30 + 0.10 * eta            # a real correlation with eta structure

        if is_data and not conditional_identical:
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
