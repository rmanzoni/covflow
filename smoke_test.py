#!/usr/bin/env python3
"""
tests/smoke_test.py
===================

End-to-end check of the whole chain on a synthetic (mc, data) pair that differ
by a KNOWN amount:

    * sigma_dxy 10% larger in data,
    * the phi-lambda partial correlation shifted by ~0.15,
    * plus an eta-dependent difference.

Needs torch + zuko, but NO ROOT. Trains small flows quickly and asserts that
the correction closes the gap: marginals improve, the injected correlation
improves, the median sigma_dxy ratio moves to ~1, and the MC-vs-data classifier
AUC collapses toward 0.5.

    python tests/smoke_test.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covflow import data as D          # noqa: E402
from covflow import features as F      # noqa: E402
from covflow import flows as FL        # noqa: E402
from covflow import correct as C       # noqa: E402
from covflow import validate as V      # noqa: E402


def main(n=60_000, epochs=40):
    # 0) feature round trip on real-ish matrices
    mc, dat = D.make_synthetic(n=n, seed=1)
    rt = F.roundtrip_error(mc.X)
    print(f"feature round trip           max rel error {rt:.1e}")
    assert rt < 1e-6, "parameterisation not invertible"

    to_feat, _ = F.get_transforms("logsigma_corr")
    y_mc_raw = to_feat(F.packed_to_matrix(mc.X))
    y_dat_raw = to_feat(F.packed_to_matrix(dat.X))

    scaler = D.Standardiser.fit(np.concatenate([y_mc_raw, y_dat_raw]),
                                np.concatenate([mc.C, dat.C]))

    fcfg = FL.FlowConfig(n_context=mc.C.shape[1], transforms=3, hidden=(96, 96), bins=8)
    tcfg = FL.TrainConfig(epochs=epochs, verbose=False, patience=6)

    flow_mc = FL.build_flow(fcfg)
    flow_mc, _ = FL.train_flow(flow_mc, scaler.x(y_mc_raw), scaler.c(mc.C), mc.w, tcfg, "mc")
    flow_data = FL.build_flow(fcfg)
    flow_data, _ = FL.train_flow(flow_data, scaler.x(y_dat_raw), scaler.c(dat.C), dat.w, tcfg, "data")

    # 1) latent convention on MC
    z = FL.data_to_latent(flow_mc, scaler.x(y_mc_raw), scaler.c(mc.C))
    y_back = scaler.x_inv(FL.latent_to_data(flow_mc, z, scaler.c(mc.C)))
    rterr = np.max(np.abs(y_back - y_mc_raw))
    print(f"latent convention            mean {z.mean():+.3f}, std {z.std():.3f}, "
          f"round trip {rterr:.1e}")

    # 2) morph + validation
    packed_corr, y_mc, y_corr = C.morph(mc.X, mc.C, flow_mc, flow_data, scaler)
    rep = V.run(mc.X, mc.C, mc.w, dat.X, dat.C, dat.w,
                packed_corr, y_mc, y_corr, scaler)

    pd = rep["pd_fraction_after_correction"]
    print(f"PD after correction          {100*pd:.4f}%")

    kdxy = F.FEATURE_NAMES.index("log_sigma_dxy")
    kpl = F.FEATURE_NAMES.index("pcorr_phi_lambda")
    w1b_dxy = rep["marginal_w1_before"]["log_sigma_dxy"]
    w1a_dxy = rep["marginal_w1_after"]["log_sigma_dxy"]
    w1b_pl = rep["marginal_w1_before"]["pcorr_phi_lambda"]
    w1a_pl = rep["marginal_w1_after"]["pcorr_phi_lambda"]
    print(f"log_sigma_dxy   W1           {w1b_dxy:.3f} -> {w1a_dxy:.3f}")
    print(f"pcorr_phi_lambda W1          {w1b_pl:.3f} -> {w1a_pl:.3f}")
    print(f"mean W1 over 15 features     {rep['marginal_w1_mean_before']:.3f} -> "
          f"{rep['marginal_w1_mean_after']:.3f}")

    r = rep["median_sigma_ratio"]["log_sigma_dxy"]
    print(f"median sigma_dxy  MC/data    {r['mc_over_data']:.3f} -> {r['corr_over_data']:.3f}")
    print(f"corr max|dRho|               {rep['corr_max_dabs_before']:.3f} -> "
          f"{rep['corr_max_dabs_after']:.3f}")
    print(f"classifier AUC               {rep['classifier_auc_before']:.3f} -> "
          f"{rep['classifier_auc_after']:.3f}")

    # 3) distillation gate
    model, size, dinfo = C.distill(y_mc, y_corr, mc.C, scaler, epochs=150, verbose=False)
    _, y_d = C.distilled_apply(model, size, mc.X, mc.C, scaler)
    auc_d = V.classifier_auc(y_d, mc.w, y_dat_raw, dat.w, scaler=scaler)
    print(f"distilled: worst residual    {100*dinfo['worst_residual_over_width']:.1f}%"
          f" of feature width, AUC {auc_d:.3f}")

    # ---- assertions ----
    ok = True
    ok &= pd > 0.999
    ok &= w1a_dxy < 0.5 * w1b_dxy
    ok &= w1a_pl < 0.5 * w1b_pl
    ok &= rep["marginal_w1_mean_after"] < rep["marginal_w1_mean_before"]
    ok &= rep["corr_max_dabs_after"] < rep["corr_max_dabs_before"]
    ok &= abs(r["corr_over_data"] - 1.0) < abs(r["mc_over_data"] - 1.0)
    ok &= rep["classifier_auc_after"] < 0.5 * (rep["classifier_auc_before"] - 0.5) + 0.5 + 0.02
    ok &= auc_d < rep["classifier_auc_before"]
    print("\nSMOKE TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
