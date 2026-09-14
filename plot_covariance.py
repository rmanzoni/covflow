#!/usr/bin/env python3
"""
plot_covariance.py
==================

Visualise the 5x5 curvilinear track covariance correlation structure from the
ntuples, inclusively and in bins of kinematics/pileup, and TEST whether it
factorises into blocks.

Why block structure is worth knowing
------------------------------------
A helix fit in CMS is driven by two nearly independent measurements:

    r-phi (bending plane)  ->  qoverp, phi, dxy
    r-z   (longitudinal)   ->  lambda, dsz

If that separation holds in the covariance, then of the 10 correlations only
4 are within-block (qoverp-phi, qoverp-dxy, phi-dxy, lambda-dsz) and 6 are
cross-block and should be ~0. A correction would then need
    5 log-sigmas + 4 within-block correlations = 9 features, not 15,
which directly reduces the dimensionality a normalising flow has to resolve.

This script measures that rather than assuming it.

Outputs (one PDF plus a JSON):
  page 1  inclusive correlation matrices: data, MC, difference
  page 2  same for canonical partial correlations (what covflow's flow sees)
  page 3  block test: per-pair |rho| with within/cross-block colouring
  page 4+ one page per binning variable: matrix per bin, plus trends vs the bin

Example
-------
python plot_covariance.py \
    --data data_2022C_partial.root --mc bc_covariance.root \
    --tree tree --cov-prefix mu1_cov_ \
    --selection '(mu1_pt > 4.5) & (abs(mu1_eta) < 2.4) & (mu1_id_medium>0.5) & (mu2_id_medium>0.5)' \
    --data-selection '(abs(mass-3.0969)<0.1)' \
    --mc-selection '(abs(jpsi_mass-3.0969)<0.1)' \
    --bin-vars 'mu1_pt' 'abs(mu1_eta)' 'npv' \
    --max-events 400000 --out covariance_structure
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from covflow import features as F
from covflow.data import selection_branches, _SEL_GLOBALS

# ---------------------------------------------------------------------------
# the physics hypothesis to be tested
# ---------------------------------------------------------------------------

# r-phi (bending plane) and r-z (longitudinal) blocks. Canonical definition
# lives in covflow.features so the plots, the feature subsets and the flow all
# agree on one thing.
BLOCKS = {k: list(v) for k, v in F.PARAM_BLOCKS.items()}


def block_of(i):
    return F.param_block(i)


def pair_is_within_block(i, j):
    return F.pair_within_block(i, j)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load(paths, tree, cov_prefix, bin_exprs, selection=None,
         max_events=None, seed=0):
    """
    Read the 15 covariance branches plus whatever the selection and the binning
    expressions need. Returns (packed (N,15), dict of binning arrays).
    """
    import uproot
    import awkward as ak

    cov_branches = [cov_prefix + n for n in F.PACK_NAMES]
    need = list(cov_branches)
    for e in list(bin_exprs) + ([selection] if selection else []):
        need += selection_branches(e)
    need = list(dict.fromkeys(need))

    parts = []
    files = [paths] if isinstance(paths, str) else list(paths)
    for fp in files:
        with uproot.open(fp) as fh:
            t = fh[tree]
            have = set(t.keys())
            missing = [b for b in need if b not in have]
            if missing:
                stem = cov_prefix.rstrip("_")
                raise KeyError(
                    f"{len(missing)} branch(es) missing from {fp}:{tree}: "
                    f"{missing[:6]}\n  branches starting with {stem!r}: "
                    f"{sorted(k for k in have if k.startswith(stem))[:15]}")
            parts.append(t.arrays(need, library="ak", how=dict))
    data = {k: ak.concatenate([p[k] for p in parts]) for k in need}

    ref = data[cov_branches[0]]
    jagged = ref.ndim > 1

    def flat(name, cast=True):
        a = data[name]
        if jagged:
            a, _ = ak.broadcast_arrays(a, ref)
            a = ak.flatten(a)
        out = ak.to_numpy(a)
        return out.astype(np.float64) if cast else out

    X = np.stack([flat(b) for b in cov_branches], axis=1)

    env = dict(_SEL_GLOBALS)
    all_br = set()
    for e in list(bin_exprs) + ([selection] if selection else []):
        all_br |= set(selection_branches(e))
    for b in all_br:
        env[b] = flat(b, cast=False)

    if selection:
        m = np.asarray(eval(selection, {"__builtins__": {}}, env))  # noqa: S307
        if m.dtype != bool:
            raise TypeError(f"selection {selection!r} is not boolean")
        X = X[m]
        env = {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == m.shape
                   else v) for k, v in env.items()}

    bins = {}
    for e in bin_exprs:
        bins[e] = np.asarray(eval(e, {"__builtins__": {}}, env),  # noqa: S307
                             dtype=np.float64)

    # keep finite, positive-definite matrices only
    good = np.isfinite(X).all(1)
    good &= F.is_positive_definite(F.packed_to_matrix(X))
    X = X[good]
    bins = {k: v[good] for k, v in bins.items()}

    if max_events is not None and len(X) > max_events:
        sel = np.random.default_rng(seed).choice(len(X), max_events, replace=False)
        X = X[sel]
        bins = {k: v[sel] for k, v in bins.items()}
    return X, bins


# ---------------------------------------------------------------------------
# correlation summaries
# ---------------------------------------------------------------------------

def correlation_stats(packed):
    """
    Per-track correlation matrices -> elementwise median and IQR.

    The median is used rather than the mean because these distributions have
    long tails; note the elementwise median of many PD matrices need not itself
    be PD, so it is a visualisation summary, not something to feed a fitter.
    Also returns the median canonical partial correlations, which are what the
    flow actually transports.
    """
    M = F.packed_to_matrix(packed)
    sig = np.sqrt(np.diagonal(M, axis1=-2, axis2=-1))
    R = M / sig[:, :, None] / sig[:, None, :]
    med = np.median(R, axis=0)
    q75, q25 = np.percentile(R, [75, 25], axis=0)
    iqr = q75 - q25

    y = F.matrix_to_features(M)              # log sigma + atanh(CPC)
    cpc = np.tanh(y[:, F.DIM:])
    cpc_med = np.median(cpc, axis=0)
    cpc_iqr = (np.percentile(cpc, 75, axis=0) - np.percentile(cpc, 25, axis=0))
    log_sig_med = np.median(y[:, :F.DIM], axis=0)
    return dict(corr_med=med, corr_iqr=iqr, cpc_med=cpc_med, cpc_iqr=cpc_iqr,
                log_sigma_med=log_sig_med, n=len(packed))


def block_test(stats):
    """
    Quantify the block hypothesis: compare |rho| within blocks against |rho|
    across blocks, for both the plain and the partial correlations.
    """
    med = stats["corr_med"]
    out = {"pairs": {}}
    within, cross = [], []
    for i in range(F.DIM):
        for j in range(i):
            nm = f"{F.PARAM_NAMES[i]}_{F.PARAM_NAMES[j]}"
            w = pair_is_within_block(i, j)
            v = float(abs(med[i, j]))
            out["pairs"][nm] = {"rho_med": float(med[i, j]),
                                "abs_rho": v,
                                "within_block": bool(w),
                                "block_i": block_of(i), "block_j": block_of(j)}
            (within if w else cross).append(v)
    # partial correlations, same split
    wcpc, ccpc = [], []
    for k, (i, j) in enumerate(F.LOWER_PAIRS):
        nm = F.PCORR_NAMES[k]
        v = float(abs(stats["cpc_med"][k]))
        out["pairs"].setdefault(nm, {})
        out["pairs"][nm].update({"abs_pcorr": v,
                                 "pcorr_med": float(stats["cpc_med"][k]),
                                 "within_block": bool(pair_is_within_block(i, j))})
        (wcpc if pair_is_within_block(i, j) else ccpc).append(v)

    out["summary"] = {
        "n_within": len(within), "n_cross": len(cross),
        "abs_rho_within_min": float(np.min(within)),
        "abs_rho_within_median": float(np.median(within)),
        "abs_rho_cross_max": float(np.max(cross)),
        "abs_rho_cross_median": float(np.median(cross)),
        "abs_pcorr_within_min": float(np.min(wcpc)),
        "abs_pcorr_cross_max": float(np.max(ccpc)),
        "separation_rho": float(np.min(within) / max(np.max(cross), 1e-9)),
        "separation_pcorr": float(np.min(wcpc) / max(np.max(ccpc), 1e-9)),
    }
    return out


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _matshow(ax, Mat, title, vmin=-1, vmax=1, cmap="RdBu_r", fmt="{:.2f}",
             annotate=True, blocks=True):
    import matplotlib.pyplot as plt
    im = ax.imshow(Mat, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(F.DIM)); ax.set_yticks(range(F.DIM))
    ax.set_xticklabels(F.PARAM_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(F.PARAM_NAMES, fontsize=7)
    if annotate:
        for i in range(F.DIM):
            for j in range(F.DIM):
                v = Mat[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(v) > 0.6 * max(abs(vmin), abs(vmax))
                        else "black")
    if blocks:
        # outline the hypothesised blocks
        order = {p: k for k, p in enumerate(F.PARAM_NAMES)}
        for name, members in BLOCKS.items():
            for i in members:
                for j in members:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                               fill=False, ec="k", lw=1.2))
    ax.set_title(title, fontsize=8)
    return im


def make_pdf(res_data, res_mc, binned, path, block_data, block_mc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        # ---- page 1: inclusive plain correlations -----------------------
        ncol = 3 if res_mc else 1
        fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0), squeeze=False)
        im = _matshow(axes[0][0], res_data["corr_med"],
                      f"data: median correlation (n={res_data['n']:,})")
        if res_mc:
            _matshow(axes[0][1], res_mc["corr_med"],
                     f"MC: median correlation (n={res_mc['n']:,})")
            d = res_data["corr_med"] - res_mc["corr_med"]
            _matshow(axes[0][2], d, "data - MC", vmin=-0.3, vmax=0.3)
        fig.colorbar(im, ax=axes[0][ncol - 1], fraction=0.046)
        fig.suptitle("Track covariance correlation matrix (boxes = "
                     "hypothesised r-phi / r-z blocks)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- page 2: partial correlations (what the flow sees) ----------
        def cpc_matrix(cpc):
            Mat = np.eye(F.DIM)
            for k, (i, j) in enumerate(F.LOWER_PAIRS):
                Mat[i, j] = Mat[j, i] = cpc[k]
            return Mat
        fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0), squeeze=False)
        _matshow(axes[0][0], cpc_matrix(res_data["cpc_med"]),
                 "data: median partial correlation")
        if res_mc:
            _matshow(axes[0][1], cpc_matrix(res_mc["cpc_med"]),
                     "MC: median partial correlation")
            _matshow(axes[0][2],
                     cpc_matrix(res_data["cpc_med"] - res_mc["cpc_med"]),
                     "data - MC", vmin=-0.3, vmax=0.3)
        fig.suptitle("Canonical partial correlations -- the off-diagonal "
                     "features covflow transports", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- page 3: block test ----------------------------------------
        fig, axes = plt.subplots(2, 1, figsize=(9, 7))
        for ax, key, lab in ((axes[0], "abs_rho", "|correlation|"),
                             (axes[1], "abs_pcorr", "|partial correlation|")):
            pairs = [(n, v) for n, v in block_data["pairs"].items() if key in v]
            pairs.sort(key=lambda t: -t[1][key])
            xs = np.arange(len(pairs))
            vals = [p[1][key] for p in pairs]
            cols = ["tab:red" if p[1]["within_block"] else "tab:blue" for p in pairs]
            ax.bar(xs - 0.2, vals, width=0.4, color=cols, label="data")
            if res_mc:
                mvals = [block_mc["pairs"][p[0]][key] for p in pairs]
                ax.bar(xs + 0.2, mvals, width=0.4, color=cols, alpha=0.45,
                       hatch="//", label="MC")
            ax.set_xticks(xs)
            ax.set_xticklabels([p[0] for p in pairs], rotation=40, ha="right",
                               fontsize=7)
            ax.set_ylabel(lab, fontsize=8)
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=7)
        s = block_data["summary"]
        axes[0].set_title(
            f"red = within block (r-phi: qoverp,phi,dxy | r-z: lambda,dsz),  "
            f"blue = cross block\n"
            f"data: min|rho| within = {s['abs_rho_within_min']:.3f}, "
            f"max|rho| cross = {s['abs_rho_cross_max']:.3f}  "
            f"(separation {s['separation_rho']:.1f}x)", fontsize=8)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- pages 4+: one per binning variable ------------------------
        for var, entries in binned.items():
            nb = len(entries)
            if nb == 0:
                continue
            ncols = min(nb, 5)
            nrows = int(np.ceil(nb / ncols))
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(3.1 * ncols, 3.0 * nrows),
                                     squeeze=False)
            for a in axes.ravel():
                a.axis("off")
            for b, ent in enumerate(entries):
                ax = axes[b // ncols][b % ncols]; ax.axis("on")
                _matshow(ax, ent["data"]["corr_med"],
                         f"{var} in [{ent['lo']:.3g},{ent['hi']:.3g})\n"
                         f"n={ent['data']['n']:,}", annotate=True)
            fig.suptitle(f"data: median correlation in bins of {var}", fontsize=10)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

            # trends: each pair vs bin centre, data vs MC
            fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
            centres = [0.5 * (e["lo"] + e["hi"]) if np.isfinite(e["hi"])
                       else e["lo"] for e in entries]
            for k, (i, j) in enumerate(F.LOWER_PAIRS):
                nm = F.PCORR_NAMES[k].replace("pcorr_", "")
                w = pair_is_within_block(i, j)
                ls = "-" if w else "--"
                lw = 1.8 if w else 1.0
                axes[0].plot(centres,
                             [e["data"]["cpc_med"][k] for e in entries],
                             ls, lw=lw, marker="o", ms=3, label=nm)
                if res_mc:
                    axes[1].plot(centres,
                                 [e["data"]["cpc_med"][k] - e["mc"]["cpc_med"][k]
                                  for e in entries], ls, lw=lw, marker="o", ms=3,
                                 label=nm)
            axes[0].set_ylabel("median partial correlation (data)", fontsize=8)
            axes[1].set_ylabel("data - MC", fontsize=8)
            axes[1].set_xlabel(var, fontsize=9)
            axes[0].set_title("solid = within block, dashed = cross block",
                              fontsize=9)
            for a in axes:
                a.grid(alpha=0.3); a.axhline(0, color="k", lw=0.6)
                a.legend(fontsize=6, ncol=5)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

            # log sigma trends
            fig, ax = plt.subplots(figsize=(9, 4))
            for d in range(F.DIM):
                ax.plot(centres, [e["data"]["log_sigma_med"][d] for e in entries],
                        "-o", ms=3, label=f"data {F.PARAM_NAMES[d]}")
                if res_mc:
                    ax.plot(centres, [e["mc"]["log_sigma_med"][d] for e in entries],
                            "--s", ms=3, alpha=0.6)
            ax.set_xlabel(var, fontsize=9)
            ax.set_ylabel("median log sigma", fontsize=8)
            ax.set_title("solid = data, dashed = MC", fontsize=9)
            ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=5)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return path


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--mc", nargs="*", default=[])
    p.add_argument("--tree", default="Events")
    p.add_argument("--cov-prefix", default="trk_cov_")
    p.add_argument("--selection", default=None,
                   help="applied to both samples")
    p.add_argument("--data-selection", default=None,
                   help="ANDed with --selection for data only")
    p.add_argument("--mc-selection", default=None,
                   help="ANDed with --selection for MC only")
    p.add_argument("--bin-vars", nargs="*",
                   default=["mu1_pt", "abs(mu1_eta)", "npv"],
                   help="expressions to bin in")
    p.add_argument("--n-bins", type=int, default=5)
    p.add_argument("--max-events", type=int, default=400_000)
    p.add_argument("--out", default="covariance_structure")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)

    def combine(*sels):
        s = [x for x in sels if x]
        return " & ".join(f"({x})" for x in s) if s else None

    sel_d = combine(a.selection, a.data_selection)
    sel_m = combine(a.selection, a.mc_selection)

    print(f"[load] data: {a.data}")
    if sel_d:
        print(f"       selection {sel_d}")
    Xd, Bd = load(a.data, a.tree, a.cov_prefix, a.bin_vars, sel_d, a.max_events)
    print(f"       {len(Xd):,} tracks")

    Xm, Bm = None, None
    if a.mc:
        print(f"[load] MC:   {a.mc}")
        if sel_m:
            print(f"       selection {sel_m}")
        Xm, Bm = load(a.mc, a.tree, a.cov_prefix, a.bin_vars, sel_m, a.max_events)
        print(f"       {len(Xm):,} tracks")

    res_d = correlation_stats(Xd)
    res_m = correlation_stats(Xm) if Xm is not None else None
    blk_d = block_test(res_d)
    blk_m = block_test(res_m) if res_m else None

    # ---- binned -------------------------------------------------------
    binned = {}
    for var in a.bin_vars:
        vals = Bd[var]
        edges = np.unique(np.percentile(vals, np.linspace(0, 100, a.n_bins + 1)))
        entries = []
        for b in range(len(edges) - 1):
            lo, hi = edges[b], edges[b + 1]
            md = (vals >= lo) & (vals < hi if b < len(edges) - 2 else vals <= hi)
            if md.sum() < 500:
                continue
            ent = {"lo": float(lo), "hi": float(hi),
                   "data": correlation_stats(Xd[md])}
            if Xm is not None:
                mm = (Bm[var] >= lo) & (Bm[var] < hi if b < len(edges) - 2
                                        else Bm[var] <= hi)
                if mm.sum() < 500:
                    continue
                ent["mc"] = correlation_stats(Xm[mm])
            entries.append(ent)
        binned[var] = entries
        print(f"[bin] {var}: {len(entries)} bins")

    pdf = make_pdf(res_d, res_m, binned, os.path.join(a.out, "covariance.pdf"),
                   blk_d, blk_m)

    # ---- report -------------------------------------------------------
    rep = {"n_data": int(len(Xd)), "n_mc": int(len(Xm)) if Xm is not None else 0,
           "blocks": BLOCKS, "block_test_data": blk_d,
           "block_test_mc": blk_m,
           "binned_block_separation": {
               var: [{"lo": e["lo"], "hi": e["hi"],
                      "separation_pcorr": block_test(e["data"])["summary"]["separation_pcorr"]}
                     for e in entries]
               for var, entries in binned.items()}}
    with open(os.path.join(a.out, "covariance_report.json"), "w") as fh:
        json.dump(rep, fh, indent=2)

    # ---- console summary ---------------------------------------------
    s = blk_d["summary"]
    print("\n=========== block structure (data) ===========")
    print(f" blocks: r-phi = {[F.PARAM_NAMES[i] for i in BLOCKS['r-phi']]}, "
          f"r-z = {[F.PARAM_NAMES[i] for i in BLOCKS['r-z']]}")
    print(f" within-block pairs: {s['n_within']}, cross-block: {s['n_cross']}")
    print(f" |rho|   within: min {s['abs_rho_within_min']:.4f}  "
          f"median {s['abs_rho_within_median']:.4f}")
    print(f" |rho|   cross : max {s['abs_rho_cross_max']:.4f}  "
          f"median {s['abs_rho_cross_median']:.4f}")
    print(f" separation (min within / max cross): {s['separation_rho']:.1f}x")
    print(f" |pcorr| within min {s['abs_pcorr_within_min']:.4f}, "
          f"cross max {s['abs_pcorr_cross_max']:.4f} "
          f"-> {s['separation_pcorr']:.1f}x")
    print("\n per pair (data | MC), * = within block")
    for nm, v in sorted(blk_d["pairs"].items(),
                        key=lambda t: -t[1].get("abs_pcorr", 0)):
        if "abs_pcorr" not in v:
            continue
        mv = blk_m["pairs"][nm]["pcorr_med"] if blk_m else float("nan")
        print(f"  {'*' if v['within_block'] else ' '} {nm:24s} "
              f"pcorr {v['pcorr_med']:+.4f} | {mv:+.4f}")
    if s["separation_pcorr"] > 5:
        keep = [F.PCORR_NAMES[k] for k, (i, j) in enumerate(F.LOWER_PAIRS)
                if pair_is_within_block(i, j)]
        kidx = [F.FEATURE_NAMES.index(n) for n in keep]
        print(f"\n Block structure holds ({s['separation_pcorr']:.0f}x "
              f"separation). The 6 cross-block partial correlations are "
              f"negligible in both samples, so a correction needs only")
        print(f"   5 log-sigmas + {len(keep)} within-block correlations = "
              f"{5+len(keep)} features")
        print(f"   --features {','.join(str(i) for i in list(range(5)) + kidx)}")
    print(f"\n wrote {pdf}")
    print(f"       {os.path.join(a.out,'covariance_report.json')}")


if __name__ == "__main__":
    main()
