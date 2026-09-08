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
    p.add_argument("--param", default="logsigma_corr",
                   choices=["logsigma_corr", "log_cholesky"])
    p.add_argument("--active-features", nargs="*", type=int, default=None,
                   help="feature indices to correct (default: all). "
                        "e.g. 3 = sigma_dxy only; 0 1 2 3 4 = scales only")
    p.add_argument("--standardiser-on", default="pooled",
                   choices=["pooled", "mc"], help="fit the shared scaler on")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-3)
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
    if a.synthetic:
        print("[load] synthetic toy")
        mc, dat = D.make_synthetic(n=a.synthetic_n, seed=a.seed)
    else:
        df, mf = _expand(a.data), _expand(a.mc)
        if not df or not mf:
            raise SystemExit("no data or MC files matched; check --data/--mc globs")
        print(f"[load] {len(df)} data files, {len(mf)} MC files")
        dat = D.load_root(df, a.tree, a.cov_prefix, a.context,
                          weight_branch=a.data_weight_branch, log_pt_branch=a.log_pt,
                          clip_negative_weights=not a.keep_negative_weights,
                          max_events=a.max_events)
        mc = D.load_root(mf, a.tree, a.cov_prefix, a.context,
                         weight_branch=a.mc_weight_branch, log_pt_branch=a.log_pt,
                         clip_negative_weights=not a.keep_negative_weights,
                         max_events=a.max_events)

    if len(mc) == 0 or len(dat) == 0:
        raise SystemExit("zero rows after loading/cleaning; aborting")
    if dat.neg_weight_fraction > 0.02:
        print(f"[warn] data negative-weight fraction "
              f"{dat.neg_weight_fraction:.3f} > 2% -- clipping biases the target")
    print(f"[load] MC {len(mc):,} tracks, data {len(dat):,} tracks, "
          f"context = {mc.context_names}")

    to_feat, _ = F.get_transforms(a.param)
    y_mc_raw = to_feat(F.packed_to_matrix(mc.X))
    y_dat_raw = to_feat(F.packed_to_matrix(dat.X))

    # ---- shared standardiser ------------------------------------------
    if a.standardiser_on == "pooled":
        Xpool = np.concatenate([y_mc_raw, y_dat_raw], 0)
        Cpool = np.concatenate([mc.C, dat.C], 0)
        wpool = np.concatenate([mc.w, dat.w], 0)
        scaler = D.Standardiser.fit(Xpool, Cpool, wpool)
    else:
        scaler = D.Standardiser.fit(y_mc_raw, mc.C, mc.w)
    scaler.save(os.path.join(a.out, "scalers.json"))

    # ---- train the two flows ------------------------------------------
    fcfg = FL.FlowConfig(n_features=F.N_FEATURES, n_context=mc.C.shape[1],
                         transforms=a.transforms, hidden=tuple(a.hidden),
                         bins=a.bins, seed=a.seed)
    tcfg = FL.TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                          device=a.device)

    print("[train] MC flow")
    flow_mc = FL.build_flow(fcfg)
    flow_mc, nll_mc = FL.train_flow(flow_mc, scaler.x(y_mc_raw), scaler.c(mc.C),
                                    mc.w, tcfg, tag="mc")
    print("[train] data flow")
    flow_data = FL.build_flow(fcfg)
    flow_data, nll_data = FL.train_flow(flow_data, scaler.x(y_dat_raw), scaler.c(dat.C),
                                        dat.w, tcfg, tag="data")
    FL.save_flow(flow_mc, os.path.join(a.out, "flow_mc.pt"))
    FL.save_flow(flow_data, os.path.join(a.out, "flow_data.pt"))

    # ---- morph + validation -------------------------------------------
    print("[morph] correcting MC")
    packed_corr, y_mc, y_corr = C.morph(mc.X, mc.C, flow_mc, flow_data, scaler,
                                        param=a.param, active_features=a.active_features,
                                        device=a.device)

    # latent diagnostics on MC
    z_mc = FL.data_to_latent(flow_mc, scaler.x(y_mc_raw), scaler.c(mc.C), device=a.device)
    z_dat = FL.data_to_latent(flow_data, scaler.x(y_dat_raw), scaler.c(dat.C), device=a.device)
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
    rep["n_mc"], rep["n_data"] = len(mc), len(dat)
    rep["active_features"] = a.active_features
    rep["param"] = a.param

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


def _summary(rep):
    def g(k, d="-"):
        return rep.get(k, d)
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
    if "onnx_vs_torch_max_abs_diff" in rep:
        print(f" ONNX vs torch max |diff|   {rep['onnx_vs_torch_max_abs_diff']:.2e}")
    print("================================================\n")


if __name__ == "__main__":
    main()
