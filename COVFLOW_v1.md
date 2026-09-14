# covflow v1 — NF correction of the MC track covariance matrix

First working version of the training script + validation suite. Verified end
to end in a container (torch 2.14, zuko 1.6) on the synthetic closure test.

## What it is

Two conditional flows trained on the *same* feature space and the *same*
context, one on MC and one on (sWeighted) data:

```
f_mc  : y -> z ~ N(0,I)      trained on MC
f_data: y -> z ~ N(0,I)      trained on data
y_corr = f_data^-1( f_mc(y | c) | c )
```

In 1D this is exactly CDF_mc then CDF_data^-1 — the quantile morphing you
already do for sigma_dxy. In 15D it transports the *joint* distribution, so
correlations between covariance elements move too, which per-variable 1D
morphing cannot reach.

## Feature space (why the output is always a valid covariance)

The flow never sees raw covariance elements. Each 5x5 curvilinear covariance is
mapped to an unconstrained vector in R^15:

```
y[0:5]  = log sigma_i                    (qoverp, lambda, phi, dxy, dsz)
y[5:15] = atanh(canonical partial correlations)
```

Every point of R^15 maps back to a PD matrix, so positive-definiteness is
structural, not something the network has to learn. It also isolates the
sigma_dxy mismodelling in a single component, `y[3]`, which makes
`--active-features 3` a direct cross-check against the 1D correction.

Ordering matches `TrackCovUtils.h` exactly (`covIndexPairs()` /
`covBranchNames()`), so `features.PACK_NAMES` and the ntuple branches are the
same objects in the same order. Treat that ordering as part of the file format.

## Conditioning

Context defaults to `[pt, eta, nValidHits, nPV]` (`--log-pt` puts pt in log).
The covariance depends on hit content more strongly than on anything else: if
you only condition on (pt, eta), the flow averages over tracks with different
hit patterns and the correction is a blend. Conditioning also means you correct
the covariance *given* the hit pattern — any data/MC difference in the
hit-pattern distribution itself survives untouched. That is usually what you
want (separate, tracker-level problem) but it has to be stated.

## Files

```
covflow/features.py   covariance <-> unconstrained R^15 (+ log-Cholesky variant)
covflow/data.py       uproot loading, context, sWeights, shared standardiser, toy
covflow/flows.py      conditional NSF (zuko) + weighted-NLL training
covflow/correct.py    the morph, distillation, ONNX export + verification
covflow/validate.py   marginals, correlations, classifier test, coverage, plots
train_covflow.py      driver: train -> validate -> distil -> export
tests/smoke_test.py   end-to-end closure on synthetic data, no ROOT needed
```

## Running

```bash
# closure test, no ROOT, no branches needed
python tests/smoke_test.py

# real ntuples
python train_covflow.py \
    --data '/path/data*.root' --mc '/path/mc*.root' \
    --tree Events --cov-prefix trk_cov_ \
    --context trk_pt trk_eta trk_nValidHits nPV --log-pt trk_pt \
    --data-weight-branch sweight_jpsi \
    --out covflow_run1
```

Writes `flow_mc.pt`, `flow_data.pt`, `scalers.json`, `corrector.onnx`,
`distill_mlp.pt`, `distill_size.npy`, `report.json`, `marginals.pdf`,
`correction_size.pdf`.

## Verification run (this container, 20k tracks/sample, 25 epochs)

Toy MC and data differ by a *known* amount: sigma_dxy 10% larger in data, the
phi-lambda partial correlation shifted by 0.15, plus an eta-dependent term.

```
feature round trip           max rel error 1.2e-11
latent convention            mean +0.021, std 0.994, round trip 7.3e-07
PD after correction          100.0000%
log_sigma_dxy    W1          0.177 -> 0.005   (35x)
pcorr_phi_lambda W1          0.186 -> 0.003   (62x)
mean W1 over 15 features     0.026 -> 0.006
median sigma_dxy  MC/data    0.863 -> 1.015
classifier AUC               0.858 -> 0.542
distilled: worst residual    5.0% of feature width, AUC 0.555
ONNX vs torch                9.5e-07, ops = [Gemm, Sigmoid], single file
```

Independently checked along the way:
* round trip 1e-12 on random PD matrices spanning 7 decades of scale, both
  parameterisations;
* PD = 100% for realistic feature ranges (failures appear only for
  deliberately near-singular inputs, |CPC| -> 1, which the suite flags);
* zuko convention confirmed: `flow(c).transform(y) == z` (data -> latent), base
  is `DiagNormal`, `log_prob` decomposition exact;
* weighted AUC matches `sklearn.roc_auc_score` to 3 decimals, ties give the
  analytic 0.75; weighted W1 recovers an injected shift of 2.0 as 1.993;
  weighted correlation recovers 0.6 as 0.599.

## What to check before believing it, on real ntuples

| check | want |
|---|---|
| `roundtrip_error` | <~1e-8 on *your* matrices |
| `invalid_matrix_fraction_*` | tiny — if not, something is wrong upstream, do not mask it |
| `latent_mean_mc` / `latent_std_mc` | ~0 / ~1, else the flow has not converged |
| `pd_fraction_after_correction` | 1.0 |
| `latent_clip_fraction` | small; MC tracks landing where the data flow saw nothing |
| marginal W1 before -> after | large gain per feature |
| `corr_max_dabs` | improved — the part 1D morphing cannot reach |
| **`classifier_auc` before >> 0.5, after ~0.5** | **the only test that can really fail** |
| `classifier_auc_after_distilled` | still ~0.5 — this is the ship gate |
| `distillation.worst_residual_over_width` | few % |
| `coverage_worst` | small |

## Physics caveats (unchanged, and they matter more than the code)

**Corrected uncertainty != corrected resolution.** This corrects the *reported*
covariance, not the actual spread of the reconstructed parameters. If the true
MC resolution is also off, inflating sigma_dxy alone narrows the pull
dxy/sigma_dxy instead of fixing it. Measure the pull width in data and MC (dxy
w.r.t. the refitted dimuon vertex) and decide which of the two is mismodelled;
you may need a parameter smearing alongside this, and the two are not
interchangeable.

**The per-event map is not unique.** Only the corrected *distribution* is
guaranteed. A correction that gets the marginals right by moving events a long
way is not trustworthy per event — that is what `correction_size.pdf` is for:
you want narrow and smooth.

**Where in the chain you apply it.** Correcting the covariance and refitting
changes vertex chi2, Lxy, Lxy_sig, IP3D and IP3D_sig — i.e. the selection
efficiency, and IP3D_sig is one of your fit categories. It must be applied
*before* the cuts that use those variables, not to the final ntuple.

**Background in data.** Train the data flow with sWeights from the J/psi mass
fit. The loader reports the negative-weight fraction and warns above 2%;
clipping biases the target if that fraction is not small.

## Systematics, free from the current interface

* train on statistically independent halves of the data, take the spread;
* apply at 0% and 100%, interpolating in *feature* space, not in the corrected
  observable;
* vary the context set;
* `--active-features` gives scale-only (`0 1 2 3 4`), correlation-only
  (`5 ... 14`) and sigma_dxy-only (`3`) variants for free.

## Open items

* `#! CONFIRM` the ntuple branch names: the loader assumes
  `<cov_prefix><name>` for `name` in `features.PACK_NAMES`, i.e.
  `trk_cov_qoverp_qoverp` ... `trk_cov_dsz_dsz`. If you use the fixed-role
  layout (`mu1_cov_*`, `mu3_cov_*`), run once per role with `--cov-prefix`.
* `#! CONFIRM` the sWeight branch name for the data flow, and whether MC needs
  a gen-weight branch (`--mc-weight-branch`).
* **Hit-content branches are not in the skim yet.** `numberOfValidHits()`,
  `numberOfValidPixelHits()` and PV multiplicity need to be added alongside the
  15 covariance branches, per the note in the design README — this is the
  single biggest lever on correction quality and it needs the reprocessing
  that `packedPFCandidates` is already waiting on.
* The C++ mirror (`cpp/TrackCovFeatures.h`) and the CMSSW ONNX producer are
  *not* in this drop. The Python side is deployment-ready (single-file ONNX,
  Gemm/Sigmoid only); the C++ transform must be written to mirror
  `features.py` exactly and checked for parity before anything ships.
* Distillation quality on the toy is 5-6.5% worst-feature residual with 200
  epochs on 20k tracks; on real statistics train it longer and re-check
  `classifier_auc_after_distilled`.
* Memory: the toy defaults (200k tracks, batch 8192) exceeded the 3 GB
  container used for verification. On the M1 Max use the defaults; on a small
  box pass `--synthetic-n`/`--batch-size`.
