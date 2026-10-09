# Standalone sPlot fitter (`splot.py`, `fit_splot.py`)

Purpose: tune and validate the J/psi mass fit and its sWeights **before** they
go into a covflow training job. Nothing here imports torch or touches
`sweights.py` / `train_covflow.py`; the training is unchanged by this commit.

Requires `iminuit` in the environment you run in (`pip install iminuit`).

## Quick start

```bash
# 1. is the code right? synthetic data with known truth, ~15 s, exit code 1 on failure
python3 fit_splot.py --selftest --out splot_selftest

# 2. real data, covflow reference selection, mu2 covariance as control variables
python3 fit_splot.py --data <data.root> --selection '<the training selection>' \
    --cov-prefix mu2_cov_ --out splot_v1

# fast iteration on the model: 10% of the candidates, no variations
python3 fit_splot.py ... --prescale 0.1 --variations --out splot_quick
```

## Model (from the 2018 sigma_dxy study)

Two double-sided Crystal Balls (common mean, shared tails, `s2 = rsig*s1`,
`rsig >= 1`) plus ONE exponential (default since the Run 3 test: the 2018
second exponential went to fb=0 with identical yields), window +-250 MeV,
tails free. Alternatives: `--signal dscb`, `--tails independent`,
`--background 2exp`,
`--window`, `--fix NAME=VALUE`, `--fix-tails-from-mc`.

Binned by default (1 MeV bins, exact bin integrals of the model): same yield
and error as unbinned on the self-test (505 vs 506), cost independent of N.

## Reading the summary

| line | what is compared | good |
|---|---|---|
| chi2/ndf | data vs fitted curve, 100 bins, sparse bins merged | ~1 (2018: 3.2) |
| weights consistent | signal + background weight per candidate vs 1 | < 1e-2 |
| leakage | background left if the weights are used on +-100/150 MeV only | the reason to train on the full window |
| sPlot vs sideband | two background subtractions of a control variable | agree (indicative only: shared events) |
| low vs high sideband | background shape of a control variable on each side | chi2/ndf ~ 1; if not, sPlot's key assumption fails |
| variations | yield and sWeighted mean of each control, alternative models/windows | spread small against the data/MC difference covflow corrects |
| toys | fitted vs generated yield; scatter vs reported error | bias ~0, ratio ~1 |

## Known properties of the 2018 model (measured on synthetic data)

- The two exponentials are usually more freedom than a 0.5 GeV window can
  constrain (lam1/lam2/fb correlations 0.97-0.99). MIGRAD can then be
  "invalid" and HESSE errors unstable while the curve is fine. Look at the
  variations and toys, not at MIGRAD's flag alone.
- The signal/background split in the tails is model-dependent at the
  few-% level; on a synthetic sample the free-tail fit moved Ns by 3% at a
  likelihood BETTER than the truth. The variation table measures what this
  does to the covariance variables.

## Using the weights in training

`train_covflow.py --sweights mass` now uses this module: it removes the mass
terms from all selections, applies the fit window to data AND MC, fits the
training sample itself, and keeps the negative weights. The `--sweight-*`
options mirror the model options here, so a configuration validated with
`fit_splot.py` can be passed over unchanged.

`sweights.root` (tree `sweights`) holds `file_index`, `entry`, `mass`,
`sw_sig`, `sw_bkg` for every candidate in the fit window;
`report.json["files"][file_index]` is the input path. The weights are valid
**only** summed over the whole fit window: the training must drop the
narrower mass cut, not re-apply it, and must not clip negative weights
(both done in train_covflow/validate since this patch).
