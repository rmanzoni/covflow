#!/usr/bin/env python3
"""
train_covflow.py
================

Driver: load data + MC, train the MC and data flows on a shared standardiser,
run the validation suite, distil the morph to a deployable MLP, export ONNX,
and write report.json + marginals.pdf + correction_size.pdf.

Fail-loud: wrong branch, zero matched rows, non-finite covariance, or a data
flow trained on all-negative weights all raise rather than silently degrade.

Examples
--------
Real ntuples:
    python train_covflow.py \
        --data '/path/charmonium2018*.root' --mc '/path/hb_mc*.root' \
        --tree Events --cov-prefix trk_cov_ \
        --data-weight-branch sweight_jpsi \
        --context trk_pt trk_eta trk_nValidHits nPV --log-pt trk_pt \
        --out covflow_run1

Dry run on the synthetic toy (no ROOT, no branches needed):
    python train_covflow.py --synthetic --out covflow_toy --epochs 40
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from covflow import data as D
from covflow import features as F


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="*", default=[], help="data ROOT glob(s)")
    p.add_argument("--mc", nargs="*", default=[], help="MC ROOT glob(s)")
    p.add_argument("--tree", default="Events")
    p.add_argument("--cov-prefix", default="trk_cov_")
    p.add_argument("--context", nargs="+",
                   default=["trk_pt", "trk_eta", "trk_nValidHits", "nPV"])
    p.add_argument("--log-pt", default=None,
                   help="context branch to replace by log(pt), e.g. trk_pt")
    p.add_argument("--data-weight-branch", default=None)
    p.add_argument("--mc-weight-branch", default=None)
    p.add_argument("--data-selection", default=None,
                   help="numpy expression applied to data, e.g. "
                        "'(mu1_pt > 3) & (abs(mu1_eta) < 2.4)'. Branches it "
                        "needs are read automatically. Use & | ~ with "
                        "parenthesised comparisons, not and/or/not.")
    p.add_argument("--mc-selection", default=None,
                   help="same, for MC; may differ from --data-selection")
    p.add_argument("--selection", default=None,
                   help="shorthand: apply the same selection to data and MC")
    p.add_argument("--features", default="all",
                   help="feature subspace the flows are TRAINED in: "
                        "'all' (15), 'diag' (5 log-sigma), 'corr' (10 partial "
                        "correlations), or explicit indices like '0,3,4'. "
                        "Features outside it are copied from MC unchanged.")
    p.add_argument("--param", default="logsigma_corr",
                   choices=["logsigma_corr", "log_cholesky"])
    p.add_argument("--active-features", nargs="*", type=int, default=None,
                   help="further mask on the OUTPUT of a full-dimensional "
                        "flow. Different from --features, which changes the "
                        "flow's dimensionality. e.g. 3 = sigma_dxy only")
    p.add_argument("--closure-bins", default="4",
                   help="quantile bins per context dimension for the binned "
                        "closure test: an int for all dims, or a comma list "
                        "like '4,3,1'. '0' disables the binned closure.")
    p.add_argument("--closure-min-count", type=int, default=400,
                   help="skip context cells with fewer entries than this")
    p.add_argument("--plot-bins", default="3,3",
                   help="context binning used for marginals_binned.pdf. Keep "
                        "it coarse (default 3,3) -- the panels must stay "
                        "readable and each cell needs enough entries to see a "
                        "shape. Independent of --closure-bins.")
    p.add_argument("--no-binned-plots", action="store_true")
    p.add_argument("--no-context-reweight", action="store_true",
                   help="skip the context-reweighted global closure")
    p.add_argument("--inject", nargs="*", default=None, metavar="IDX:DELTA",
                   help="Inject a KNOWN distortion before training, to test "
                        "whether the flow can recover it. 'IDX:DELTA' adds "
                        "DELTA to feature IDX (for a log-sigma this multiplies "
                        "that sigma by exp(DELTA)). e.g. '0:0.15' inflates "
                        "sigma_qoverp by 16%%. Several may be given. The "
                        "validation then reports whether the injected shift "
                        "was removed -- if a large injected shift is NOT "
                        "recovered, the failure is structural, not a question "
                        "of the discrepancy being too small.")
    p.add_argument("--inject-into", default="mc", choices=["mc", "data"],
                   help="which sample receives the injected distortion")
    p.add_argument("--null-test", default=None, choices=["data", "mc"],
                   help="Split this sample in half and morph half 1 -> half 2. "
                        "The two halves come from the SAME distribution, so the "
                        "correct map is the identity and EVERY residual the "
                        "report shows is pure noise: the floor of what this "
                        "setup can achieve. A feature whose real data/MC "
                        "discrepancy is below its null-test residual cannot be "
                        "corrected, only broadened.")
    p.add_argument("--sweights", default=None, metavar="MASS_BRANCH",
                   help="Derive sWeights in data by fitting MASS_BRANCH with a "
                        "double-sided Crystal Ball signal plus two "
                        "exponentials, and train the data flow on the "
                        "background-subtracted sample. Without this the data "
                        "flow learns the covariance of signal PLUS background, "
                        "and the correction target is wrong by roughly the "
                        "background fraction.")
    p.add_argument("--sweight-window", type=float, default=0.200,
                   help="half-width of the mass fit range in GeV (default "
                        "0.200). A mass cut in --selection tighter than this is "
                        "relaxed FOR THE FIT ONLY, so there is sideband left to "
                        "constrain the background.")
    p.add_argument("--sweight-mc-mass-branch", default=None,
                   help="fit simulated signal with this branch first and FIX "
                        "the Crystal Ball tail parameters in the data fit. "
                        "Strongly recommended: the two exponentials can absorb "
                        "the signal's radiative tail, which biases the yield.")
    p.add_argument("--reweight-bins", default=None,
                   help="context binning used for the reweighted closure only "
                        "(default: same as --closure-bins). Reweighting wants "
                        "FINE bins -- residual within-bin spectrum differences "
                        "leak straight into the features with the steepest "
                        "kinematic dependence.")
    p.add_argument("--standardiser-on", default="pooled",
                   choices=["pooled", "mc"], help="fit the shared scaler on")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--schedule", default="cosine", choices=["cosine","none"],
                   help="cosine LR annealing (default) stabilises the tail of "
                        "training; the noisy val-NLL swings seen at constant "
                        "LR make early stopping fire on fluctuations")
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--transforms", type=int, default=4)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--keep-negative-weights", action="store_true")
    p.add_argument("--no-distill", action="store_true")
    p.add_argument("--no-onnx", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true",
                   help="use the built-in toy instead of ROOT files")
    p.add_argument("--synthetic-n", type=int, default=200_000,
                   help="tracks per sample for --synthetic")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _expand(globs):
    files = []
    for g in globs:
        files += sorted(glob.glob(g))
    return files


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    # torch-dependent modules imported here so --help works without torch
    from covflow import flows as FL
    from covflow import correct as C
    from covflow import validate as V

    # ---- load ----------------------------------------------------------
    data_sel = a.data_selection or a.selection
    mc_sel = a.mc_selection or a.selection

    sweight_report = None
    if a.synthetic:
        print("[load] synthetic toy")
        mc, dat = D.make_synthetic(n=a.synthetic_n, seed=a.seed)
        if data_sel or mc_sel:
            raise SystemExit("--selection is not supported with --synthetic")
    else:
        df, mf = _expand(a.data), _expand(a.mc)
        if not df or not mf:
            raise SystemExit("no data or MC files matched; check --data/--mc globs")
        print(f"[load] {len(df)} data files, {len(mf)} MC files")
        if data_sel:
            print(f"[load] data selection: {data_sel}")
        if mc_sel:
            print(f"[load] MC   selection: {mc_sel}")
        if data_sel != mc_sel:
            print("[warn] data and MC selections differ -- any resulting "
                  "difference in the covariance distributions will be "
                  "absorbed into the correction as if it were detector "
                  "mismodelling")
        if a.sweights and a.data_weight_branch:
            raise SystemExit("--sweights and --data-weight-branch both given; "
                             "choose one source of data weights")
        dat = D.load_root(df, a.tree, a.cov_prefix, a.context,
                          weight_branch=a.data_weight_branch, log_pt_branch=a.log_pt,
                          selection=data_sel,
                          extra_branches=([a.sweights] if a.sweights else ()),
                          clip_negative_weights=not a.keep_negative_weights,
                          max_events=a.max_events)
        mc = D.load_root(mf, a.tree, a.cov_prefix, a.context,
                         weight_branch=a.mc_weight_branch, log_pt_branch=a.log_pt,
                         selection=mc_sel,
                         clip_negative_weights=not a.keep_negative_weights,
                         max_events=a.max_events)
        # ---- sWeights ------------------------------------------------
        if a.sweights:
            from covflow import sweights as SW
            lo = SW.JPSI_MASS - a.sweight_window
            hi = SW.JPSI_MASS + a.sweight_window
            fit_sel, dropped = (D.drop_terms_using(data_sel, [a.sweights])
                                if data_sel else (None, []))
            if dropped:
                print(f"[sweights] relaxing for the fit only: dropped "
                      f"{dropped} from the data selection so the fit keeps "
                      f"sideband on both sides of the peak")
            fit_sample = D.load_root(
                df, a.tree, a.cov_prefix, a.context,
                log_pt_branch=a.log_pt, selection=fit_sel,
                extra_branches=[a.sweights])
            mfit = fit_sample.extra[a.sweights]
            print(f"[sweights] fit sample: {len(mfit):,} candidates "
                  f"(vs {len(dat):,} passing the full selection)")

            tails = None
            if a.sweight_mc_mass_branch:
                mc_fit = D.load_root(mf, a.tree, a.cov_prefix, a.context,
                                     log_pt_branch=a.log_pt, selection=mc_sel,
                                     extra_branches=[a.sweight_mc_mass_branch])
                tails, _ = SW.fit_signal_tails(
                    mc_fit.extra[a.sweight_mc_mass_branch], lo, hi)

            sw_par, sw_info = SW.fit_mass(mfit, lo, hi, fixed=tails)
            SW.plot_fit(mfit, sw_par, lo, hi,
                        os.path.join(a.out, "sweight_fit.pdf"))

            sw, _, swi = SW.compute_sweights(dat.extra[a.sweights],
                                             sw_par, lo, hi)
            dat.w = sw
            print(f"[sweights] applied to {swi['n_weighted']:,} data tracks: "
                  f"sum {swi['sum_signal_weights']:,.0f}, "
                  f"{100*swi['negative_signal_weight_fraction']:.1f}% negative, "
                  f"effective N {swi['effective_n']:,.0f}")
            if swi["negative_signal_weight_fraction"] > 0.35:
                print("[warn] a large fraction of sWeights are negative -- the "
                      "sample is background dominated and the subtraction will "
                      "be noisy")
            sweight_report = {"fit": sw_info, "weights": swi,
                              "parameters": {k: float(v) for k, v in sw_par.items()},
                              "relaxed_terms": dropped,
                              "fit_selection": fit_sel}
        print(f"[load] data cutflow: {dat.cutflow()}")
        print(f"[load] MC   cutflow: {mc.cutflow()}")

    if len(mc) == 0 or len(dat) == 0:
        raise SystemExit("zero rows after loading/cleaning; aborting")
    if dat.neg_weight_fraction > 0.02:
        print(f"[warn] data negative-weight fraction "
              f"{dat.neg_weight_fraction:.3f} > 2% -- clipping biases the target")
    print(f"[load] MC {len(mc):,} tracks, data {len(dat):,} tracks, "
          f"context = {mc.context_names}")

    # ---- null test: replace (mc, data) by two halves of one sample --------
    if a.null_test:
        src = dat if a.null_test == "data" else mc
        rng = np.random.default_rng(a.seed)
        order = rng.permutation(len(src))
        h1, h2 = order[:len(order) // 2], order[len(order) // 2:]

        def half(s, sel):
            return D.Sample(s.X[sel], s.C[sel], s.w[sel], s.context_names,
                            s.neg_weight_fraction)
        mc, dat = half(src, h1), half(src, h2)
        print(f"[null-test] splitting {a.null_test} into halves of "
              f"{len(mc):,} and {len(dat):,}")
        print("[null-test] the two halves are drawn from the SAME distribution, "
              "so the correct\n            correction is the identity -- every "
              "residual below is pure noise")

    to_feat, to_mat = F.get_transforms(a.param)
    y_mc_raw = to_feat(F.packed_to_matrix(mc.X))
    y_dat_raw = to_feat(F.packed_to_matrix(dat.X))

    # ---- inject a known distortion (closure test) ----------------------
    injected = {}
    if a.inject:
        for spec in a.inject:
            try:
                k_s, d_s = spec.split(":")
                k, delta = int(k_s), float(d_s)
            except ValueError:
                raise SystemExit(f"--inject expects IDX:DELTA, got {spec!r}")
            if not 0 <= k < F.N_FEATURES:
                raise SystemExit(f"--inject index {k} out of range")
            injected[k] = injected.get(k, 0.0) + delta
        names_all = F.feature_names(a.param)
        tgt = "MC" if a.inject_into == "mc" else "data"
        for k, delta in injected.items():
            print(f"[inject] {tgt}: {names_all[k]} += {delta:+.4f}"
                  + (f"  (sigma x {np.exp(delta):.4f})" if k < F.DIM else ""))
        if a.inject_into == "mc":
            y_mc_raw = y_mc_raw.copy()
            for k, d in injected.items():
                y_mc_raw[:, k] += d
            mc.X = F.matrix_to_packed(to_mat(y_mc_raw))
        else:
            y_dat_raw = y_dat_raw.copy()
            for k, d in injected.items():
                y_dat_raw[:, k] += d
            dat.X = F.matrix_to_packed(to_mat(y_dat_raw))
        # the transform is a bijection, but assert the matrices survive
        bad = (~F.is_positive_definite(F.packed_to_matrix(
            mc.X if a.inject_into == "mc" else dat.X))).sum()
        if bad:
            raise SystemExit(f"injection produced {bad} non-PD matrices")

    # ---- shared standardiser ------------------------------------------
    if a.standardiser_on == "pooled":
        Xpool = np.concatenate([y_mc_raw, y_dat_raw], 0)
        Cpool = np.concatenate([mc.C, dat.C], 0)
        wpool = np.concatenate([mc.w, dat.w], 0)
        scaler = D.Standardiser.fit(Xpool, Cpool, wpool)
    else:
        scaler = D.Standardiser.fit(y_mc_raw, mc.C, mc.w)
    scaler.save(os.path.join(a.out, "scalers.json"))

    # ---- feature subspace ----------------------------------------------
    spec = a.features
    if "," in spec or spec.strip().lstrip("-").isdigit():
        spec = [int(v) for v in spec.replace(",", " ").split()]
    idx = F.subset_indices(spec)
    sub_names = [F.feature_names(a.param)[i] for i in idx]
    print(f"[setup] training flows in {len(idx)}-D subspace "
          f"({a.features}): {sub_names}")
    if len(idx) < F.N_FEATURES:
        print(f"[setup] the other {F.N_FEATURES - len(idx)} feature(s) are "
              f"copied from MC unchanged")

    # ---- train the two flows ------------------------------------------
    fcfg = FL.FlowConfig(n_features=len(idx), n_context=mc.C.shape[1],
                         transforms=a.transforms, hidden=tuple(a.hidden),
                         bins=a.bins, seed=a.seed)
    tcfg = FL.TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                          schedule=a.schedule, patience=a.patience,
                          device=a.device)

    print("[train] MC flow")
    flow_mc = FL.build_flow(fcfg)
    flow_mc, nll_mc = FL.train_flow(flow_mc, scaler.x(y_mc_raw)[:, idx],
                                    scaler.c(mc.C), mc.w, tcfg, tag="mc")
    print("[train] data flow")
    # A DIFFERENT initialisation from the MC flow. With the same seed the two
    # flows start identical and their errors partly cancel in the composition,
    # which flatters the null test: it would measure correlated error rather
    # than the error the correction actually carries.
    import dataclasses as _dc
    fcfg_data = _dc.replace(fcfg, seed=fcfg.seed + 1)
    flow_data = FL.build_flow(fcfg_data)
    flow_data, nll_data = FL.train_flow(flow_data, scaler.x(y_dat_raw)[:, idx],
                                        scaler.c(dat.C), dat.w, tcfg, tag="data")
    FL.save_flow(flow_mc, os.path.join(a.out, "flow_mc.pt"))
    FL.save_flow(flow_data, os.path.join(a.out, "flow_data.pt"))

    # ---- training history: NLL and learning rate vs epoch ---------------
    # Kept because the shape of these curves is diagnostic in its own right:
    # a validation loss that swings by more than it improves means the
    # retained snapshot is a fluctuation, and the two flows then fluctuate
    # independently, which breaks the identity the morph relies on.
    hist = {"mc": [], "data": []}
    for rec in tcfg.history:
        tag, epoch, vloss = rec[0], rec[1], rec[2]
        lr = rec[3] if len(rec) > 3 else float("nan")
        if tag in hist:
            hist[tag].append({"epoch": int(epoch), "val_nll": float(vloss),
                              "lr": float(lr)})
    hist_csv = os.path.join(a.out, "training_history.csv")
    with open(hist_csv, "w") as fh:
        fh.write("flow,epoch,val_nll,lr\n")
        for tag in ("mc", "data"):
            for r in hist[tag]:
                fh.write(f"{tag},{r['epoch']},{r['val_nll']:.6f},{r['lr']:.6e}\n")

    try:
        _plot_training(hist, tcfg.diagnostics,
                       os.path.join(a.out, "training_curves.pdf"))
    except Exception as e:
        print(f"[warn] training curve plot failed: {e}")

    # ---- morph + validation -------------------------------------------
    print("[morph] correcting MC")
    packed_corr, y_mc, y_corr = C.morph(mc.X, mc.C, flow_mc, flow_data, scaler,
                                        param=a.param, active_features=a.active_features,
                                        feature_indices=idx, device=a.device)

    # latent diagnostics on MC
    z_mc = FL.data_to_latent(flow_mc, scaler.x(y_mc_raw)[:, idx],
                             scaler.c(mc.C), device=a.device)
    z_dat = FL.data_to_latent(flow_data, scaler.x(y_dat_raw)[:, idx],
                              scaler.c(dat.C), device=a.device)
    lat = {"latent_mean_mc": float(np.mean(z_mc)), "latent_std_mc": float(np.std(z_mc))}
    # MC latents landing outside the support the data flow actually saw.
    # Anchor on the OBSERVED min/max of the data latents per component: using
    # e.g. per-feature 0.5/99.5 percentiles would make .any(1) over 15 dims
    # report ~14% by pure combinatorics even for a perfect match.
    lo, hi = z_dat.min(0), z_dat.max(0)
    out = (z_mc < lo) | (z_mc > hi)
    lat["latent_clip_fraction"] = float(out.any(1).mean())          # per track
    lat["latent_clip_fraction_percomp"] = float(out.mean())         # per component

    print("[validate] running suite")
    rep = V.run(mc.X, mc.C, mc.w, dat.X, dat.C, dat.w,
                packed_corr, y_mc, y_corr, scaler, param=a.param, device=a.device)
    rep.update(lat)
    rep["nll_mc"], rep["nll_data"] = nll_mc, nll_data
    rep["training_diagnostics"] = tcfg.diagnostics
    rep["training_history"] = hist
    rep["correction_noise_ratio"] = V.correction_noise_ratio(
        y_mc, y_corr, mc.w, y_dat_raw, dat.w, feature_indices=idx,
        param=a.param)   # recomputed with conditional signal below if available
    rep["n_mc"], rep["n_data"] = len(mc), len(dat)
    rep["active_features"] = a.active_features
    rep["param"] = a.param
    rep["features_spec"] = a.features
    if sweight_report:
        rep["sweights"] = sweight_report
    rep["null_test"] = a.null_test
    rep["injected"] = {F.feature_names(a.param)[k]: v
                       for k, v in injected.items()} or None
    rep["inject_into"] = a.inject_into if injected else None
    if injected:
        # Did the flow remove the injected shift? Compare the median of the
        # corrected feature against data, in units of the injection.
        rec = {}
        for k, d in injected.items():
            nm = F.feature_names(a.param)[k]
            med_mc = float(np.median(y_mc[:, k]))
            med_cor = float(np.median(y_corr[:, k]))
            med_dat = float(np.median(y_dat_raw[:, k]))
            moved = med_cor - med_mc
            rec[nm] = {"injected_delta": d,
                       "median_shift_applied_by_flow": moved,
                       "fraction_of_injection_removed":
                           float(-moved / d) if d else float("nan"),
                       "median_mc_minus_data_before": med_mc - med_dat,
                       "median_mc_minus_data_after": med_cor - med_dat}
        rep["injection_recovery"] = rec
    rep["feature_indices"] = idx
    rep["feature_subspace_names"] = sub_names
    rep["data_selection"], rep["mc_selection"] = data_sel, mc_sel
    if not a.synthetic:
        rep["cutflow_data"] = dat.cutflow()
        rep["cutflow_mc"] = mc.cutflow()

    # When only part of the space is corrected, the 15-D AUC cannot reach 0.5 --
    # the untouched features still separate. The honest closure number is the
    # AUC restricted to the subspace the flows actually trained in.
    if len(idx) < F.N_FEATURES:
        rep["classifier_auc_subspace_before"] = V.classifier_auc(
            y_mc[:, idx], mc.w, y_dat_raw[:, idx], dat.w, device=a.device)
        rep["classifier_auc_subspace_after"] = V.classifier_auc(
            y_corr[:, idx], mc.w, y_dat_raw[:, idx], dat.w, device=a.device)

    # ---- closure at fixed context --------------------------------------
    # The morph corrects p(y|c), never p(c). With signal MC against an
    # inclusive data stream the spectra differ, so the global numbers above
    # mix flow error with kinematics. These two tests separate them.
    nb = [int(v) for v in str(a.closure_bins).replace(",", " ").split()]
    if len(nb) == 1:
        nb = nb * mc.C.shape[1]
    if len(nb) != mc.C.shape[1]:
        raise SystemExit(f"--closure-bins gave {len(nb)} values for "
                         f"{mc.C.shape[1]} context dimensions")

    if max(nb) > 1:
        Cpool = np.concatenate([mc.C, dat.C], 0)
        wpool = np.concatenate([mc.w, dat.w], 0)
        edges = V.context_bin_edges(Cpool, wpool, nb)
        rep["closure_bins"] = nb
        rep["context_names"] = mc.context_names

        # (a) context reweighting -> one confound-free global number
        if not a.no_context_reweight:
            rnb = ([int(v) for v in str(a.reweight_bins).replace(",", " ").split()]
                   if a.reweight_bins else list(nb))
            if len(rnb) == 1:
                rnb = rnb * mc.C.shape[1]
            if len(rnb) != mc.C.shape[1]:
                raise SystemExit(f"--reweight-bins gave {len(rnb)} values for "
                                 f"{mc.C.shape[1]} context dimensions")
            redges = (edges if rnb == list(nb)
                      else V.context_bin_edges(Cpool, wpool, rnb))
            print(f"[validate] context-reweighted closure (bins {rnb})")
            rep["reweight_bins"] = rnb
            w_rw, rwinfo = V.context_weights(mc.C, mc.w, dat.C, dat.w, redges)
            rep["context_reweight"] = rwinfo
            rep["classifier_auc_reweighted_before"] = V.classifier_auc(
                y_mc[:, idx], w_rw, y_dat_raw[:, idx], dat.w, device=a.device)
            rep["classifier_auc_reweighted_after"] = V.classifier_auc(
                y_corr[:, idx], w_rw, y_dat_raw[:, idx], dat.w, device=a.device)
            rw_b, rw_a = {}, {}
            for k in idx:
                nm = F.feature_names(a.param)[k]
                rw_b[nm] = V.weighted_w1(y_mc[:, k], w_rw, y_dat_raw[:, k], dat.w)
                rw_a[nm] = V.weighted_w1(y_corr[:, k], w_rw, y_dat_raw[:, k], dat.w)
            rep["w1_reweighted_before"] = rw_b
            rep["w1_reweighted_after"] = rw_a
            # features whose CONDITIONAL discrepancy got worse. The marginal
            # table cannot be used for this: it is dominated by the data/MC
            # kinematic spectrum mismatch, which the morph cannot and should
            # not fix, and it flags features that in fact improved at fixed
            # context.
            rep["worse_conditional"] = [k for k in rw_b
                                        if rw_a[k] > 1.1 * max(rw_b[k], 1e-9)]
            rep["correction_noise_ratio"] = V.correction_noise_ratio(
                y_mc, y_corr, mc.w, y_dat_raw, dat.w, feature_indices=idx,
                param=a.param, w_mc_reweighted=w_rw)

        # (b) cell-by-cell closure
        print("[validate] binned closure")
        rep["binned_closure"] = V.binned_closure(
            y_mc, mc.w, mc.C, y_corr, y_dat_raw, dat.w, dat.C,
            edges, mc.context_names, feature_indices=idx,
            min_count=a.closure_min_count, device=a.device, param=a.param)

        # (c) can the flow itself be split by block?
        try:
            bf = V.block_feature_correlation(y_dat_raw, dat.C, edges, dat.w,
                                             param=a.param)
            rep["block_feature_correlation_data"] = bf
            bfm = V.block_feature_correlation(y_mc, mc.C, edges, mc.w,
                                              param=a.param)
            rep["block_feature_correlation_mc"] = bfm
        except Exception as e:
            print(f"[warn] block feature correlation failed: {e}")

        # (d) the same thing, drawn
        if not a.no_binned_plots:
            pnb = [int(v) for v in str(a.plot_bins).replace(",", " ").split()]
            if len(pnb) == 1:
                pnb = pnb * mc.C.shape[1]
            pedges = V.context_bin_edges(Cpool, wpool, pnb)
            print(f"[validate] binned marginal plots ({int(np.prod([len(e)-1 for e in pedges]))} cells)")
            try:
                V.plot_binned_marginals(
                    y_mc, mc.w, mc.C, y_corr, y_dat_raw, dat.w, dat.C,
                    pedges, mc.context_names, feature_indices=idx,
                    path=os.path.join(a.out, "marginals_binned.pdf"),
                    min_count=max(a.closure_min_count // 2, 50), param=a.param)
            except Exception as e:
                print(f"[warn] binned marginal plots failed: {e}")

    V.plot_marginals(y_mc, mc.w, y_dat_raw, dat.w, y_corr, a.param,
                     os.path.join(a.out, "marginals.pdf"))
    V.plot_correction_size(y_mc, y_corr, a.param,
                           os.path.join(a.out, "correction_size.pdf"))

    # ---- distill + ONNX ------------------------------------------------
    if not a.no_distill:
        print("[distill] fitting residual MLP")
        model, size, dinfo = C.distill(y_mc, y_corr, mc.C, scaler, device=a.device)
        rep["distillation"] = dinfo
        # classifier AUC recomputed on the DISTILLED correction: the ship gate
        packed_d, y_d = C.distilled_apply(model, size, mc.X, mc.C, scaler,
                                          param=a.param, device=a.device)
        rep["classifier_auc_after_distilled"] = V.classifier_auc(
            y_d, mc.w, y_dat_raw, dat.w, scaler=scaler, device=a.device)
        np.save(os.path.join(a.out, "distill_size.npy"), size)
        import torch
        torch.save(model.state_dict(), os.path.join(a.out, "distill_mlp.pt"))
        if not a.no_onnx:
            try:
                n_in = F.N_FEATURES + mc.C.shape[1]
                onnx_path = os.path.join(a.out, "corrector.onnx")
                C.export_onnx(model, n_in, onnx_path)
                rep["onnx"] = "corrector.onnx"
                try:
                    rep["onnx_vs_torch_max_abs_diff"] = C.verify_onnx(
                        onnx_path, model, n_in, device=a.device)
                    rep["onnx_ops"] = C.onnx_ops(onnx_path)
                    rep["onnx_external_data"] = os.path.exists(onnx_path + ".data")
                except Exception as e:
                    rep["onnx_verify_error"] = str(e)
            except Exception as e:
                rep["onnx_error"] = str(e)
                print(f"[warn] ONNX export failed: {e}")

    with open(os.path.join(a.out, "report.json"), "w") as fh:
        json.dump(rep, fh, indent=2)

    _summary(rep)


def _plot_training(hist, diagnostics, path):
    """
    Two panels: validation NLL vs epoch (both flows, with the retained epoch
    marked and the noise band shaded) and learning rate vs epoch.

    The shaded band is +-1 sigma of the loss over the last evaluations. If the
    marked best epoch sits inside that band, model selection is being driven by
    noise rather than by convergence.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    colours = {"mc": "tab:orange", "data": "tab:blue"}
    for tag in ("mc", "data"):
        rows = hist.get(tag, [])
        if not rows:
            continue
        ep = [r["epoch"] for r in rows]
        nll = [r["val_nll"] for r in rows]
        axes[0].plot(ep, nll, "-o", ms=3, color=colours[tag], label=f"{tag} flow")
        axes[1].plot(ep, [r["lr"] for r in rows], "-", color=colours[tag],
                     label=f"{tag} flow")
        d = diagnostics.get(tag, {})
        be = d.get("best_epoch")
        if be is not None and be >= 0:
            bv = d.get("best_val_nll")
            axes[0].plot([be], [bv], "*", ms=16, color=colours[tag],
                         markeredgecolor="k", zorder=5)
            noise = d.get("val_noise_tail")
            if noise and np.isfinite(noise):
                tail = np.array(nll[-15:])
                med = float(np.median(tail))
                axes[0].axhspan(med - noise, med + noise, color=colours[tag],
                                alpha=0.12)
                axes[0].annotate(
                    f"{tag}: best {d.get('best_minus_median_sigma', float('nan')):.1f}"
                    f"$\\sigma$ below median, trend "
                    f"{d.get('tail_trend_per_epoch', float('nan')):+.3f}/epoch",
                    xy=(be, bv), xytext=(6, -14 if tag == "mc" else 10),
                    textcoords="offset points", fontsize=7, color=colours[tag])
    axes[0].set_ylabel("validation NLL", fontsize=9)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_title("star = retained snapshot; band = +-1 sigma of the last "
                      "15 evaluations", fontsize=9)
    axes[1].set_ylabel("learning rate", fontsize=9)
    axes[1].set_xlabel("epoch", fontsize=9)
    axes[1].set_yscale("log"); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def _summary(rep):
    def g(k, d="-"):
        return rep.get(k, d)
    if rep.get("null_test"):
        print("\n  *** NULL TEST on the %s sample: the true answer is the "
              "identity. ***" % rep["null_test"])
        print("  *** Everything below is the NOISE FLOOR of this setup, not a "
              "correction. ***")
    print("\n================ covflow report ================")
    print(f" roundtrip_error            {g('roundtrip_error'):.2e}")
    print(f" PD after correction        {100*g('pd_fraction_after_correction'):.4f}%")
    print(f" latent MC  mean/std        {g('latent_mean_mc'):+.3f} / {g('latent_std_mc'):.3f}")
    print(f" latent clip fraction       {g('latent_clip_fraction'):.4f} "
          f"(per component {g('latent_clip_fraction_percomp', float('nan')):.4f})")
    print(f" marginal W1  mean          {g('marginal_w1_mean_before'):.4f} -> {g('marginal_w1_mean_after'):.4f}")
    print(f" corr max|dRho|             {g('corr_max_dabs_before'):.4f} -> {g('corr_max_dabs_after'):.4f}")
    print(f" classifier AUC             {g('classifier_auc_before'):.4f} -> {g('classifier_auc_after'):.4f}")
    if "classifier_auc_after_distilled" in rep:
        print(f" classifier AUC (distilled) {g('classifier_auc_after_distilled'):.4f}")
    if "distillation" in rep:
        print(f" distill worst resid/width  {rep['distillation']['worst_residual_over_width']:.3f}")
    print(f" coverage worst             {g('coverage_worst'):.4f}")
    if "classifier_auc_subspace_before" in rep:
        print(f" classifier AUC (subspace)  {g('classifier_auc_subspace_before'):.4f}"
              f" -> {g('classifier_auc_subspace_after'):.4f}   <-- closure of the"
              f" trained subspace")
    if "onnx_vs_torch_max_abs_diff" in rep:
        print(f" ONNX vs torch max |diff|   {rep['onnx_vs_torch_max_abs_diff']:.2e}")

    # Per-feature table. A mean over 15 features can hide the correction making
    # some of them WORSE -- which is exactly what happened in run 2 -- so the
    # per-feature verdict is printed, not just the average.
    b = rep.get("marginal_w1_before", {})
    aft = rep.get("marginal_w1_after", {})
    trained = set(rep.get("feature_subspace_names", b.keys()))
    if b:
        print("\n MARGINAL comparison (contaminated by the kinematic spectrum --\n see the context-reweighted table below for the number that matters)")
        print(" feature                 W1 before    W1 after   change  trained")
        worse = []
        for k in b:
            r = aft[k] / b[k] if b[k] > 0 else float("nan")
            tag = ("  --" if r < 0.5 else "   -" if r < 0.9 else
                   "   =" if r < 1.1 else "   +" if r < 2 else "  ++")
            if r >= 1.1:
                worse.append(k)
            print(f"  {k:22s} {b[k]:9.4f}  {aft[k]:9.4f}   {tag}"
                  f"     {'y' if k in trained else '.'}")
        print("  (-- much better, - better, = unchanged, + worse, ++ much worse)")
        if worse:
            wc = rep.get("worse_conditional")
            if wc is None:
                print(f"\n [warn] correction made {len(worse)} feature(s) WORSE "
                      f"in the MARGINAL: {', '.join(worse)}")
            else:
                recovered = [k for k in worse if k not in wc]
                print(f"\n [note] {len(worse)} feature(s) look worse in the "
                      f"MARGINAL table above: {', '.join(worse)}")
                if recovered:
                    print(f"        ...but {len(recovered)} of them IMPROVE at "
                          f"fixed context: {', '.join(recovered)}")
                    print(f"        The marginal is contaminated by the data/MC "
                          f"kinematic spectrum, which the\n        morph cannot "
                          f"and should not fix. Read the conditional numbers.")
                if wc:
                    print(f"\n [warn] genuinely WORSE at fixed context: "
                          f"{', '.join(wc)}")
                else:
                    print(f"\n [ok] no feature is worse at fixed context")

    # ---- closure free of the p(c) confound ------------------------------
    if "classifier_auc_reweighted_before" in rep:
        rw = rep.get("context_reweight", {})
        print(f"\n CONTEXT-REWEIGHTED (MC reweighted to data's context spectrum)")
        print(f"  classifier AUC            {g('classifier_auc_reweighted_before'):.4f}"
              f" -> {g('classifier_auc_reweighted_after'):.4f}")
        print(f"  effective N of MC         {rw.get('effective_n',0):,.0f}"
              f"   uncovered data {100*rw.get('uncovered_data_fraction',0):.2f}%")
        b, aft = rep.get("w1_reweighted_before", {}), rep.get("w1_reweighted_after", {})
        for k in b:
            r = aft[k]/b[k] if b[k] > 0 else float('nan')
            print(f"   {k:22s} {b[k]:8.4f} -> {aft[k]:8.4f}   x{r:.2f}")

    cnr = rep.get("correction_noise_ratio")
    if cnr:
        used = next(iter(cnr.values())).get("signal_used", "marginal")
        print(f"\n SIZE OF THE CORRECTION vs the discrepancy it must fix"
              f"   (signal = {used})")
        print("  feature                 shift   scatter  scatter/signal")
        for k, v in cnr.items():
            print(f"  {k:22s} {v['median_shift']:+.4f}  {v['scatter_1sigma']:.4f}"
                  f"     {v['scatter_over_signal']:6.2f}")
        print("  DESCRIPTIVE ONLY -- do not use as a decision rule. A large "
              "spread is not\n  necessarily noise: tracks genuinely need "
              "different corrections. Measured on\n  real data, features with "
              "ratio > 2 have both improved threefold and got worse.")

    ir = rep.get("injection_recovery")
    if ir:
        print(f"\n INJECTION CLOSURE TEST (distortion added to "
              f"{rep.get('inject_into')})")
        for nm, v in ir.items():
            f = v["fraction_of_injection_removed"]
            verdict = ("RECOVERED" if 0.75 <= f <= 1.25 else
                       "PARTIAL" if 0.3 <= f < 0.75 else
                       "NOT RECOVERED" if f < 0.3 else "OVERSHOT")
            print(f"  {nm:22s} injected {v['injected_delta']:+.4f}, flow moved "
                  f"{v['median_shift_applied_by_flow']:+.4f}"
                  f"  -> {100*f:5.1f}% removed   {verdict}")
        print("  A large injected shift that is NOT recovered means the failure"
              " is structural,\n  not a matter of the native discrepancy being"
              " too small to see.")

    bf = rep.get("block_feature_correlation_data")
    if bf and bf.get("n_cells"):
        print(f"\n BLOCK SEPARABILITY OF THE FLOW (data, at fixed context)")
        print(f"  cross-block feature corr: max {bf['max_cross_block_feature_corr']:.3f}"
              f"  median {bf['median_cross_block_feature_corr']:.3f}")
        print(f"  within-block, for scale : median "
              f"{bf['median_within_block_feature_corr']:.3f}")
        wp = bf.get("worst_pair", {})
        print(f"  worst pair: {wp.get('a','?')} vs {wp.get('b','?')} "
              f"= {wp.get('abs_corr',float('nan')):.3f}")
        if bf["separable"]:
            print("  -> cross-block feature correlations are small: the 9-D flow"
                  " can be SPLIT into\n     independent 6-D (r-phi) and 3-D (r-z)"
                  " flows (--features rphi / rz), which\n     also lets the two"
                  " train in parallel.")
        else:
            print("  -> cross-block feature correlations are NOT negligible."
                  " The covariance MATRIX\n     factorises, but the features do"
                  " not, across tracks. Keep ONE joint flow\n     (--features"
                  " block); splitting it would throw this correlation away.")

    bc = rep.get("binned_closure")
    if bc and bc.get("cells"):
        used, tot = len(bc["cells"]), bc["n_cells_total"]
        print(f"\n BINNED CLOSURE  ({used} cells used, "
              f"{bc['n_cells_skipped']} skipped for <{bc['min_count']} entries)")
        if used < 0.25 * tot or used < 4:
            print(f"  [warn] only {used}/{tot} cells survived the min-count. "
                  f"These numbers are NOT a meaningful binned closure -- they "
                  f"come from a handful of small, possibly unrepresentative "
                  f"cells. Use coarser --closure-bins, a smaller "
                  f"--closure-min-count, or more events (--max-events).")
        print(f"  mean over cells: W1 {bc['w1_mean_before']:.4f} -> "
              f"{bc['w1_mean_after']:.4f}")
        if "auc_before_mean" in bc:
            print(f"                   AUC {bc['auc_before_mean']:.4f} -> "
                  f"{bc['auc_after_mean']:.4f}"
                  f"   (worst cell after: {bc['auc_after_worst']:.4f})")
        print("\n  per-feature, averaged over cells:")
        pb, pa = bc["w1_before_per_feature"], bc["w1_after_per_feature"]
        for k in pb:
            r = pa[k]/pb[k] if pb[k] > 0 else float('nan')
            flag = "  <-- worse" if r > 1.1 else ""
            print(f"   {k:22s} {pb[k]:8.4f} -> {pa[k]:8.4f}   x{r:.2f}{flag}")
        print("\n  worst cells by AUC after:")
        cs = sorted(bc["cells"], key=lambda c: -c.get("auc_after", 0))[:5]
        for c in cs:
            print(f"   AUC {c.get('auc_before',0):.3f} -> {c.get('auc_after',0):.3f}"
                  f"  n_mc={c['n_mc']:>7,} n_data={c['n_data']:>7,}  {c['label']}")
    print("================================================\n")


if __name__ == "__main__":
    main()
