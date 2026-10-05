#!/usr/bin/env python3
"""
fit_splot.py -- standalone J/psi mass fit + sPlot, validated independently of
the covflow training.

It reads the same ntuples, with the same selection syntax, as
train_covflow.py, fits the dimuon mass, computes sWeights, and runs every
check that can be done WITHOUT training a flow. Only once these look right is
it worth putting the weights into a training job.

WHAT IT PRODUCES (in --out)
  mass_fit.pdf/.png      the fit, 2018 layout, linear + log-y page
  mass_scan.pdf          the mass spectrum far beyond the fit window after the
                         (mass-relaxed) selection: a trigger or skim mass cut
                         inside the window shows up here as an edge
  sweight_curve.pdf      signal/background weight versus mass
  controls.pdf           per control variable: sPlot vs sideband subtraction
                         (vs MC), and low vs high sideband
  variations.pdf         sPlot signal shape of each control variable under
                         alternative fit models / windows
  toys.pdf               pulls from toy fits (only with --toys)
  sweights.root          tree "sweights": file_index, entry, mass, sw_sig,
                         sw_bkg (+ --id-branches) for every fitted candidate
  report.json            everything, machine readable
  fit_splot.log          the console output

EXAMPLES
  # 1. check the code itself on synthetic data (no input needed, ~15 s)
  python3 fit_splot.py --selftest --out splot_selftest

  # 2. the covflow reference selection, mu2 covariance as control variables
  python3 fit_splot.py \\
      --data /pnfs/psi.ch/cms/trivcat/store/user/manzoni/rjpsi_run3/covflow_dimuon_ntuples_16sept26/dimuoun_run3_PARTIAL.root \\
      --selection '(mu1_pt > 4.5) & (abs(mu1_eta) < 2.4) & (mu1_id_medium>0.5) & (mu2_id_medium>0.5) & (abs(mass-3.0969)<0.1) & ((lxy * cos2d /pt * 3.0969)>0.008)' \\
      --cov-prefix mu2_cov_ --out splot_run3_v1

  # 3. same, plus MC overlay and 200 toys of 1M events each
  python3 fit_splot.py ... \\
      --mc /pnfs/.../hb_dimuon_2018_mc_PARTIAL.root \\
      --mc-selection '(abs(mu1_gen_pdgid)==13) & (abs(mu2_gen_pdgid)==13)' \\
      --toys 200 --toy-events 1000000

THE MASS CUT
Terms of --selection that use the mass branch are dropped before reading,
exactly as train_covflow does for its fit: the fit needs sidebands. The
weights are then valid ONLY for the whole fit window: summing them over a
narrower mass range does not subtract the background (report.json ->
"leakage" says by how much). The training must therefore use the fit window,
not the original mass cut.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
import time

import numpy as np

try:
    from covflow import splot as S
    from covflow import data as D
    from covflow import features as F
except ImportError:                       # run from inside the repo
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from covflow import splot as S
    from covflow import data as D
    from covflow import features as F


DEFAULT_VARIATIONS = ["tails=independent", "background=2exp",
                      "window=0.20", "window=0.30"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, path):
        self.f = open(path, "w")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def expand(paths):
    """Globs for local paths; xrootd URLs are passed through untouched
    (glob would silently drop them)."""
    out = []
    for p in paths:
        if "://" in p:
            out.append(p)
        else:
            hit = sorted(glob.glob(p))
            if not hit:
                raise SystemExit(f"no file matches {p!r}")
            out += hit
    return out


def _eval(expr, arrays):
    env = dict(D._SEL_GLOBALS)
    env.update(arrays)
    with np.errstate(all="ignore"):
        return eval(expr, {"__builtins__": {}}, env)   # noqa: S307


def _eval_selection(expr, arrays, n):
    m = np.asarray(_eval(expr, arrays))
    if m.dtype != bool:
        raise TypeError(f"selection {expr!r} evaluated to {m.dtype}, not bool: "
                        f"use & | ~ with parenthesised comparisons")
    if m.shape != (n,):
        raise ValueError(f"selection gives shape {m.shape}, expected ({n},)")
    return m


def parse_variation(spec, nominal):
    """'window=0.30,background=exp' -> config dict based on the nominal."""
    cfg = dict(nominal)
    for tok in spec.split(","):
        k, _, v = tok.partition("=")
        k, v = k.strip(), v.strip()
        if k not in cfg:
            raise SystemExit(f"variation {spec!r}: unknown key {k!r} "
                             f"(allowed: {sorted(cfg)})")
        cfg[k] = float(v) if k == "window" else v
    return cfg


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def read_sample(files, tree, mass_branch, selection, half_window, controls,
                id_branches, prescale, rng, step, scan_edges, label):
    """
    One pass over the files. Keeps, for candidates passing `selection` and
    within +-half_window of the J/psi mass: mass, file index, entry, the
    control variables and the id branches. Also histograms the mass over
    scan_edges for all selected candidates (no window, no prescale).
    """
    import uproot

    sel_br = D.selection_branches(selection) if selection else []
    ctrl_br = sorted({b for e in controls.values() for b in D.selection_branches(e)})
    want = list(dict.fromkeys([mass_branch] + sel_br + ctrl_br + list(id_branches)))

    totals = []
    for fp in files:              # metadata only: fail before the long read
        with uproot.open(fp) as fh:
            if tree not in fh:
                raise SystemExit(f"{fp}: no tree {tree!r}; keys {fh.keys()[:10]}")
            t = fh[tree]
            have = set(t.keys())
            miss = [b for b in want if b not in have]
            if miss:
                raise SystemExit(f"{fp}:{tree} lacks branch(es) {miss}")
            totals.append(t.num_entries)
    print(f"[read {label}] {len(files)} file(s), {sum(totals):,} entries, "
          f"{len(want)} branches")

    keep = {"mass": [], "file_index": [], "entry": []}
    keep.update({f"ctrl:{k}": [] for k in controls})
    keep.update({f"id:{b}": [] for b in id_branches})
    scan = np.zeros(len(scan_edges) - 1)
    n_read = n_sel = 0
    bar = S.Progress(sum(totals), f"[read {label}]", " entries")
    for fi, fp in enumerate(files):
        with uproot.open(fp) as fh:
            for arrs, rep in fh[tree].iterate(want, step_size=step,
                                              library="np", report=True):
                n = len(arrs[mass_branch])
                m = np.asarray(arrs[mass_branch], float)
                mask = np.isfinite(m)
                if selection:
                    mask &= _eval_selection(selection, arrs, n)
                n_read += n
                n_sel += int(mask.sum())
                scan += np.histogram(m[mask], bins=scan_edges)[0]
                sel = mask & (np.abs(m - S.JPSI_MASS) <= half_window)
                if prescale < 1.0:
                    sel &= rng.random(n) < prescale
                idx = np.nonzero(sel)[0]
                keep["mass"].append(m[idx])
                keep["file_index"].append(np.full(idx.size, fi, np.int32))
                keep["entry"].append(rep.tree_entry_start + idx.astype(np.int64))
                for k, e in controls.items():
                    v = np.broadcast_to(np.asarray(_eval(e, arrs), float), (n,))
                    keep[f"ctrl:{k}"].append(v[idx].astype(np.float32))
                for b in id_branches:
                    keep[f"id:{b}"].append(arrs[b][idx])
                bar.update(n)
    bar.close()
    out = {k: (np.concatenate(v) if v else np.array([])) for k, v in keep.items()}
    print(f"[read {label}] selected {n_sel:,} of {n_read:,} candidates; "
          f"{out['mass'].size:,} kept within +-{1e3*half_window:.0f} MeV"
          + (f" after prescale {prescale}" if prescale < 1 else ""))
    out["_counts"] = {"entries_read": n_read, "selected": n_sel,
                      "kept_in_read_window": int(out["mass"].size)}
    out["_scan"] = scan
    return out


# ---------------------------------------------------------------------------
# one fit configuration
# ---------------------------------------------------------------------------

def run_config(cfg, mass, args, fixed, label, verbose=True):
    lo, hi = S.JPSI_MASS - cfg["window"], S.JPSI_MASS + cfg["window"]
    model = S.MassModel(lo, hi, signal=cfg["signal"], tails=cfg["tails"],
                        background=cfg["background"])
    inwin = (mass >= lo) & (mass <= hi)
    m = mass[inwin]
    fx = {k: v for k, v in fixed.items() if k in model.names}
    if args.fit == "binned":
        nb = int(round((hi - lo) / args.bin_width))
        edges = np.linspace(lo, hi, nb + 1)
        cnt, _ = np.histogram(m, bins=edges)
        res = S.fit(model, counts=cnt, edges=edges, fixed=fx,
                    strategy=args.strategy, verbose=verbose, label=label)
    else:
        res = S.fit(model, mass=m, fixed=fx, strategy=args.strategy,
                    verbose=verbose, label=label)
    gof = S.goodness_of_fit(res, mass=m, nbins=args.plot_bins)
    sp = S.SPlot(res, m)
    return {"cfg": cfg, "model": model, "res": res, "gof": gof, "sp": sp,
            "inwin": inwin}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("input")
    g.add_argument("--data", nargs="+", help="data ROOT files (globs ok)")
    g.add_argument("--tree", default="tree")
    g.add_argument("--mass-branch", default="mass")
    g.add_argument("--selection", default=None,
                   help="covflow selection string; terms using the mass "
                        "branch are dropped for the fit")
    g.add_argument("--prescale", type=float, default=1.0,
                   help="keep this random fraction of the selected candidates "
                        "(for fast tuning; the weights file then covers only "
                        "the kept fraction)")
    g.add_argument("--step-size", default="200 MB")
    g.add_argument("--id-branches", nargs="*", default=[],
                   help="extra branches copied into sweights.root, e.g. "
                        "run luminosityBlock event")

    g = p.add_argument_group("model")
    g.add_argument("--window", type=float, default=0.25,
                   help="fit half-window around the J/psi mass [GeV] (2018: 0.25)")
    g.add_argument("--signal", default="2dscb", choices=S.SIGNAL_MODELS)
    g.add_argument("--tails", default="shared", choices=S.TAIL_MODES)
    g.add_argument("--background", default="exp", choices=S.BACKGROUND_MODELS,
                   help="exp (default) or 2exp (the 2018 model). On Run 3 data "
                        "(splot_run3_v2) 2exp put the second component at "
                        "fb=0 in 3 of 4 fits, with the same yield as exp to "
                        "1 event but meaningless HESSE errors")
    g.add_argument("--fit", default="binned", choices=["binned", "unbinned"],
                   help="binned (default) uses exact bin integrals; with 1 MeV "
                        "bins it matches the unbinned result and its cost does "
                        "not grow with the sample size")
    g.add_argument("--bin-width", type=float, default=0.001)
    g.add_argument("--strategy", type=int, default=2)
    g.add_argument("--fix", nargs="*", default=[], metavar="NAME=VALUE",
                   help="hold parameters constant, e.g. nL=3 nR=5")

    g = p.add_argument_group("MC (optional)")
    g.add_argument("--mc", nargs="*", default=[])
    g.add_argument("--mc-selection", default=None,
                   help="ANDed with the (mass-relaxed) --selection")
    g.add_argument("--mc-mass-branch", default=None)
    g.add_argument("--fix-tails-from-mc", action="store_true",
                   help="fit MC signal alone and fix the tail parameters in "
                        "data. Only sensible if the MC resolution and FSR "
                        "match the data (NOT the case for Run 2 2018 MC vs "
                        "Run 3 data without checking)")

    g = p.add_argument_group("validation")
    g.add_argument("--cov-prefix", default=None,
                   help="add log(sigma) of the 5 track parameters from "
                        "<prefix>X_X as control variables, e.g. mu2_cov_")
    g.add_argument("--control", nargs="*", default=[], metavar="EXPR",
                   help="further control variables (selection-style "
                        "expressions), e.g. 'sqrt(mu2_cov_dxy_dxy)' mu2_pt")
    g.add_argument("--control-bins", type=int, default=40)
    g.add_argument("--sr-nsigma", type=float, default=2.5)
    g.add_argument("--sb-nsigma", type=float, default=5.0)
    g.add_argument("--variations", nargs="*", default=DEFAULT_VARIATIONS,
                   help=f"alternative fits, each 'key=value[,key=value]' with "
                        f"key in window/signal/tails/background. Default: "
                        f"{DEFAULT_VARIATIONS}. Pass --variations with no "
                        f"value to skip.")
    g.add_argument("--leakage-windows", nargs="*", type=float,
                   default=[0.10, 0.15],
                   help="half-widths at which to report how much background "
                        "survives if the weights are used on that subrange only")
    g.add_argument("--toys", type=int, default=0)
    g.add_argument("--toy-events", type=float, default=0,
                   help="events per toy (0 = as fitted)")
    g.add_argument("--seed", type=int, default=1)

    g = p.add_argument_group("output")
    g.add_argument("--out", required=True)
    g.add_argument("--rlabel", default="Run 3 (13.6 TeV)")
    g.add_argument("--plot-bins", type=int, default=100)
    g.add_argument("--scan-range", nargs=2, type=float, default=[2.6, 3.6])
    g.add_argument("--scan-bin-width", type=float, default=0.002)
    g.add_argument("--no-weights-file", action="store_true")
    g.add_argument("--selftest", action="store_true",
                   help="run the synthetic-data self-test of the fitter and exit")
    a = p.parse_args(argv)
    if not a.selftest and not a.data:
        p.error("--data is required (or --selftest)")
    if a.signal == "dscb" and a.tails == "independent":
        p.error("--tails independent needs --signal 2dscb")
    return a


def main(argv=None):
    a = parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    sys.stdout = _Tee(os.path.join(a.out, "fit_splot.log"))
    t0 = time.time()
    print(f"[fit_splot] {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  "
          f"argv: {' '.join(sys.argv)}")

    if a.selftest:
        ok, checks = S.selftest(outdir=a.out)
        with open(os.path.join(a.out, "report.json"), "w") as fh:
            json.dump({"selftest": [{"check": c, "passed": o, "detail": d}
                                    for c, o, d in checks]}, fh, indent=1)
        return 0 if ok else 1

    rng = np.random.default_rng(a.seed)
    files = expand(a.data)
    fixed = {}
    for tok in a.fix:
        k, _, v = tok.partition("=")
        fixed[k.strip()] = float(v)

    # ---- selection: drop the mass terms, as train_covflow does -----------
    sel, dropped = (D.drop_terms_using(a.selection, [a.mass_branch])
                    if a.selection else (None, []))
    if dropped:
        print(f"[selection] dropped for the fit (need sidebands): {dropped}")
        print("[selection] NOTE: the sWeights are valid only over the WHOLE fit "
              "window. Training with these terms re-applied does NOT subtract "
              "the background -- see 'leakage' below.")
    print(f"[selection] used: {sel}")

    # ---- controls -----------------------------------------------------------
    controls = {}
    if a.cov_prefix:
        for par in F.PARAM_NAMES:
            controls[f"log_sigma_{par}"] = f"log(sqrt({a.cov_prefix}{par}_{par}))"
    for e in a.control:
        controls[e] = e

    # ---- configurations ----------------------------------------------------
    nominal = {"window": a.window, "signal": a.signal, "tails": a.tails,
               "background": a.background}
    variations = {}
    for spec in (a.variations or []):
        cfg = parse_variation(spec, nominal)
        if cfg["signal"] == "dscb" and cfg["tails"] == "independent":
            print(f"[variations] skipping {spec!r}: independent tails need 2dscb")
            continue
        if cfg == nominal:
            print(f"[variations] skipping {spec!r}: identical to nominal")
            continue
        variations[spec] = cfg
    read_hw = max([nominal["window"]] + [c["window"] for c in variations.values()])

    # ---- read ---------------------------------------------------------------
    scan_edges = np.arange(a.scan_range[0], a.scan_range[1] + 1e-9,
                           a.scan_bin_width)
    step = (int(a.step_size) if str(a.step_size).strip().isdigit()
            else a.step_size)             # entries or '200 MB', as in covflow
    dat = read_sample(files, a.tree, a.mass_branch, sel, read_hw, controls,
                      a.id_branches, a.prescale, rng, step, scan_edges,
                      "data")
    mass = dat["mass"]
    S.plot_scan(dat["_scan"], scan_edges,
                (S.JPSI_MASS - a.window, S.JPSI_MASS + a.window),
                os.path.join(a.out, "mass_scan.pdf"), rlabel=a.rlabel)

    mc = None
    if a.mc:
        mc_sel = " & ".join(f"({x})" for x in (sel, a.mc_selection) if x) or None
        print(f"[selection] MC: {mc_sel}")
        mc = read_sample(expand(a.mc), a.tree, a.mc_mass_branch or a.mass_branch,
                         mc_sel, read_hw, controls, [], 1.0, rng, step,
                         scan_edges, "MC")

    # ---- tails from MC (optional) ------------------------------------------
    report = {"argv": sys.argv, "files": files, "selection_used": sel,
              "selection_dropped": dropped, "controls": controls,
              "counts": dat["_counts"]}
    if a.fix_tails_from_mc:
        if mc is None:
            raise SystemExit("--fix-tails-from-mc needs --mc")
        lo, hi = S.JPSI_MASS - a.window, S.JPSI_MASS + a.window
        mm = S.MassModel(lo, hi, signal=a.signal, tails=a.tails,
                         background=a.background)
        edges = np.linspace(lo, hi, int(round((hi - lo) / a.bin_width)) + 1)
        cnt, _ = np.histogram(mc["mass"], bins=edges)
        bfix = {"Nb": 0.0, "lam1": 0.0, "lam2": 0.0, "fb": 0.5}
        rmc = S.fit(mm, counts=cnt, edges=edges,
                    fixed={k: v for k, v in bfix.items() if k in mm.names},
                    label="MC signal")
        S.plot_fit(rmc, os.path.join(a.out, "mass_fit_mc.pdf"), counts=cnt,
                   edges=edges, rlabel="MC signal")
        for k in mm.tail_names:
            fixed.setdefault(k, rmc.values[k])       # --fix wins
        report["mc_signal_fit"] = rmc.summary()
        print(f"[fit] tails fixed from MC: "
              + ", ".join(f"{k}={fixed[k]:.3f}" for k in mm.tail_names))

    # ---- nominal fit ----------------------------------------------------------
    print("[fit] ===== nominal =====")
    nom = run_config(nominal, mass, a, fixed, "nominal")
    res, sp, gof = nom["res"], nom["sp"], nom["gof"]
    m_in = mass[nom["inwin"]]
    try:
        reg = S.regions(res, a.sr_nsigma, a.sb_nsigma)
    except ValueError as e:
        print(f"[regions] {e}; sideband checks skipped")
        reg = None
    S.plot_fit(res, os.path.join(a.out, "mass_fit.pdf"), mass=m_in,
               nbins=a.plot_bins, gof=gof, rlabel=a.rlabel, sp_regions=reg)
    if reg is not None:
        S.plot_weights(sp, reg, os.path.join(a.out, "sweight_curve.pdf"))
    report["fit"] = res.summary()
    report["gof"] = gof
    report["splot"] = sp.info
    report["regions"] = reg
    report["leakage"] = [sp.subrange_leakage(S.JPSI_MASS - w, S.JPSI_MASS + w)
                         for w in a.leakage_windows if w < a.window]

    # ---- weights file ------------------------------------------------------
    sw_s, sw_b = sp.weights(m_in)
    if not a.no_weights_file:
        import uproot
        path = os.path.join(a.out, "sweights.root")
        cols = {"file_index": dat["file_index"][nom["inwin"]],
                "entry": dat["entry"][nom["inwin"]],
                "mass": m_in, "sw_sig": sw_s, "sw_bkg": sw_b}
        for b in a.id_branches:
            cols[b] = dat[f"id:{b}"][nom["inwin"]]
        with uproot.recreate(path) as fo:
            # explicit mktree: uproot >= 5.7 writes RNTuple on dict assignment
            t = fo.mktree("sweights", {k: v.dtype for k, v in cols.items()})
            t.extend(cols)
        print(f"[out] {path}: {m_in.size:,} candidates (file_index -> path in "
              f"report.json 'files')")

    # ---- controls ------------------------------------------------------------
    ctrl_results, mc_hists = {}, {}
    report["control_checks"] = {}
    if controls and reg is not None:
        sr = (m_in > reg["SR"][0]) & (m_in < reg["SR"][1])
        for k in controls:
            v = dat[f"ctrl:{k}"][nom["inwin"]].astype(float)
            fin = np.isfinite(v)
            if fin.sum() < 100:
                print(f"[controls] {k}: only {fin.sum()} finite values, skipped")
                continue
            vv = v[sr & fin]
            lo_q, hi_q = np.percentile(vv, [0.5, 99.5])
            edges = np.linspace(lo_q, hi_q, a.control_bins + 1)
            r = S.control_check(m_in, v, sp, reg, edges)
            ctrl_results[k] = (edges, r)
            mc_check = None
            if mc is not None:
                w = np.abs(mc["mass"] - S.JPSI_MASS) <= a.window
                hv = mc[f"ctrl:{k}"][w].astype(float)
                fm = np.isfinite(hv)
                h, _ = np.histogram(hv[fm], bins=edges)
                mc_hists[k] = S._norm(h.astype(float), np.sqrt(h))
                # sPlot assumes the control variable is independent of the
                # mass WITHIN the signal. For track covariances it is not
                # exactly (large sigma(q/p) -> mass further from the peak).
                # Weighting pure-signal MC with the data's weight function
                # shows the size of the effect: without a correlation the
                # weighted and unweighted MC shapes agree.
                wmc, _ = sp.weights(mc["mass"][w])
                mu_u = float(hv[fm].mean())
                mu_w = float(np.average(hv[fm], weights=wmc[fm]))
                mc_check = {"mc_mean_unweighted": mu_u,
                            "mc_mean_with_data_sweights": mu_w,
                            "shift": mu_w - mu_u,
                            "data_splot_minus_mc_mean":
                                float(np.average(v[fin], weights=sw_s[fin])) - mu_u}
            c1, n1 = r["chi2_splot_vs_sideband"]
            c2, n2 = r["chi2_sblow_vs_sbhigh"]
            report["control_checks"][k] = {
                "chi2_splot_vs_sideband": [c1, n1],
                "chi2_sblow_vs_sbhigh": [c2, n2],
                "chi2_sblow_vs_sbhigh_signal_removed":
                    list(r["chi2_sblow_vs_sbhigh_signal_removed"]),
                "fraction_outside_plot_range": r["clipped_fraction"],
                "signal_fraction_SB_low": r["signal_fraction_SB_low"],
                "signal_fraction_SB_high": r["signal_fraction_SB_high"],
                "splot_weighted_mean": float(np.average(v[fin], weights=sw_s[fin])),
                "signal_mass_correlation_mc": mc_check,
            }
        S.plot_controls(ctrl_results, os.path.join(a.out, "controls.pdf"),
                        mc=mc_hists or None)

    # ---- variations ----------------------------------------------------------
    var_rows = {"nominal": nom}
    for spec, cfg in variations.items():
        print(f"[fit] ===== variation: {spec} =====")
        try:
            var_rows[spec] = run_config(cfg, mass, a, fixed, spec)
        except Exception as e:                        # report, do not abort
            print(f"[fit] variation {spec} FAILED: {e}")
            report.setdefault("variation_failures", {})[spec] = str(e)
    report["variations"] = {}
    for spec, vr in var_rows.items():
        r_ = vr["res"]
        row = {"cfg": vr["cfg"], "valid": r_.valid,
               "Ns": r_.values["Ns"], "Ns_err": r_.errors["Ns"],
               "purity": r_.values["Ns"] / (r_.values["Ns"] + r_.values["Nb"]),
               "chi2": vr["gof"]["chi2"], "ndf": vr["gof"]["ndf"],
               "at_limit": r_.at_limit,
               "sw_sum_minus_1": vr["sp"].info["max_abs_sw_s_plus_sw_b_minus_1"],
               "controls": {}}
        m_v = mass[vr["inwin"]]
        wv, _ = vr["sp"].weights(m_v)
        for k, (edges, _) in ctrl_results.items():
            v = dat[f"ctrl:{k}"][vr["inwin"]].astype(float)
            fin = np.isfinite(v)
            h, _ = np.histogram(v[fin], bins=edges, weights=wv[fin])
            e2, _ = np.histogram(v[fin], bins=edges, weights=wv[fin] ** 2)
            row["controls"][k] = {
                "weighted_mean": float(np.average(v[fin], weights=wv[fin])),
                "shape": (h / h.sum()).tolist(),
                "shape_err": (np.sqrt(e2) / abs(h.sum())).tolist()}
        report["variations"][spec] = row
    if len(var_rows) > 1 and ctrl_results:
        _plot_variations(report["variations"], ctrl_results,
                         os.path.join(a.out, "variations.pdf"))

    # ---- toys ----------------------------------------------------------------
    if a.toys > 0:
        summ, pulls = S.run_toys(res, a.toys, n_events=a.toy_events or None,
                                 bin_width=a.bin_width, seed=a.seed + 1000)
        S.plot_toys(pulls, os.path.join(a.out, "toys.pdf"))
        report["toys"] = summ

    report["elapsed_s"] = time.time() - t0
    with open(os.path.join(a.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=_json_default)
    _verdict(report)
    print(f"[fit_splot] done in {time.time() - t0:.0f}s -> {a.out}/")
    return 0


def _plot_variations(rows, ctrl_results, path):
    plt, _ = S._plt()
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.gridspec import GridSpec
    cols = ["black", "#c1272d", "#3f7fbf", "#e76300", "#832db6", "#5a9e3a"]
    with PdfPages(path) as pdf:
        for k, (edges, _) in ctrl_results.items():
            ctr = 0.5 * (edges[1:] + edges[:-1])
            nom = np.asarray(rows["nominal"]["controls"][k]["shape"])
            fig = plt.figure(figsize=(9, 8))
            gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
            ax, axr = fig.add_subplot(gs[0]), None
            axr = fig.add_subplot(gs[1], sharex=ax)
            for (spec, row), c in zip(rows.items(), cols):
                h = np.asarray(row["controls"][k]["shape"])
                e = np.asarray(row["controls"][k]["shape_err"])
                lab = (f"{spec}: Ns={row['Ns']:.0f}, "
                       f"$\\chi^2$/ndf={row['chi2']:.0f}/{row['ndf']}")
                if spec == "nominal":
                    ax.errorbar(ctr, h, yerr=e, fmt="o", ms=3, color=c, label=lab)
                else:
                    ax.stairs(h, edges, color=c, lw=1.4, label=lab)
                rr = np.divide(h, nom, out=np.full_like(h, np.nan), where=nom > 0)
                axr.plot(ctr, rr, "o-" if spec == "nominal" else "-", color=c,
                         ms=2, lw=1)
            ax.set_ylabel("sPlot signal, normalised")
            ax.legend(fontsize=8)
            plt.setp(ax.get_xticklabels(), visible=False)
            axr.set_ylim(0.9, 1.1)
            axr.set_ylabel("/ nominal")
            axr.set_xlabel(k)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _verdict(r):
    """Plain-language summary. Each line says what was compared with what."""
    f, g, s = r["fit"], r["gof"], r["splot"]
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    ok = "yes" if f["migrad_valid"] else "NO"
    print(f"Fit converged (MIGRAD valid):                        {ok}")
    if f["parameters_at_limit"]:
        print(f"  parameters stuck at a limit: {f['parameters_at_limit']}")
    print(f"Fit describes the mass spectrum: chi2/ndf = {g['chi2']:.0f}/"
          f"{g['ndf']} = {g['chi2_ndf']:.2f}  (1 = perfect; 2018 fit: 3.2)")
    err = (f"+- {f['errors']['Ns']:,.0f}" if f["errors_reliable"] else
           "(HESSE error not meaningful: covariance inaccurate or a fraction "
           "pinned at 0/1)")
    print(f"Signal yield in window: {f['values']['Ns']:,.0f} {err}, purity "
          f"{100*f['purity_in_window']:.1f}%")
    dev = s["max_abs_sw_s_plus_sw_b_minus_1"]
    print(f"Weights consistent (signal + background weight = 1 per candidate): "
          f"max deviation {dev:.1e}  {'OK' if dev < 1e-2 else 'NOT OK: fit not at maximum'}")
    print(f"Negative signal weights: "
          f"{100*s['negative_signal_weight_fraction']:.1f}% of candidates; "
          f"effective signal statistics {s['effective_n_signal']:,.0f}")
    for lk in r.get("leakage", []):
        lo, hi = lk["subrange"]
        print(f"If the weights are used only within +-{1e3*(hi-lo)/2:.0f} MeV: "
              f"{100*lk['background_fraction']:.1f}% of the weighted sample is "
              f"background (over the full window: 0)")
    for k, c in r.get("control_checks", {}).items():
        a1, n1 = c["chi2_splot_vs_sideband"]
        a2, n2 = c["chi2_sblow_vs_sbhigh"]
        print(f"{k}:")
        mcc = c.get("signal_mass_correlation_mc")
        if mcc:
            print(f"   pure-signal MC weighted with the data sWeights vs "
                  f"unweighted: mean shift {mcc['shift']:+.4f}, against a "
                  f"data-MC difference of {mcc['data_splot_minus_mc_mean']:+.4f} "
                  f"[a shift that is not small compared with the difference "
                  f"means the variable depends on mass within the signal, and "
                  f"the data target is biased by about that much]")
        print(f"   sPlot signal shape vs sideband-subtracted shape: chi2/ndf "
              f"{a1:.0f}/{n1} (same events partly: indicative)")
        a3, _ = c["chi2_sblow_vs_sbhigh_signal_removed"]
        print(f"   low vs high sideband (mostly background):        chi2/ndf "
              f"{a2:.0f}/{n2}  [independent samples: a real test of whether "
              f"the background shape depends on mass; the fit puts "
              f"{100*c['signal_fraction_SB_low']:.0f}%/"
              f"{100*c['signal_fraction_SB_high']:.0f}% signal in them; with "
              f"that removed: {a3:.0f}]")
    if len(r.get("variations", {})) > 1:
        nom = r["variations"]["nominal"]
        print("Fit-model variations (signal yield; weighted mean of each "
              "control relative to nominal):")
        for spec, row in r["variations"].items():
            if spec == "nominal":
                continue
            shifts = []
            for k, c in row["controls"].items():
                m0 = nom["controls"][k]["weighted_mean"]
                shifts.append(f"{k} {c['weighted_mean'] - m0:+.4f}")
            print(f"   {spec:22s} Ns {row['Ns']/nom['Ns']-1:+.2%}  "
                  f"valid={row['valid']}  " + ", ".join(shifts))
    if "toys" in r:
        t = r["toys"]
        if "Ns" in t:
            n = t["Ns"]
            print(f"Toys ({t['n_toys']} generated from the nominal fit, "
                  f"{t['failed']} fits failed, {t['events_per_toy']:,.0f} "
                  f"events each):")
            print(f"   signal-yield bias: {n['bias_percent']:+.2f}% +- "
                  f"{n['bias_percent_err']:.2f}%")
            print(f"   actual scatter of the fitted yield / reported error: "
                  f"{n['spread_over_error']:.2f}  (1 = errors are right)")
            print(f"   pulls: median {n['pull_median']:+.2f}, central-68% "
                  f"half-width {n['pull_68_halfwidth']:.2f}, "
                  f"{n['n_abs_pull_above_5']} beyond |5|")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
