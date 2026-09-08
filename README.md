# covflow

**Correcting the simulated track covariance matrix to match data, using normalising flows.**

This document is written to be readable without prior context. It explains what
problem this solves, why the obvious approaches are not enough, how the method
works, how to run it, how to tell whether it worked, and — importantly — the
ways it can quietly mislead you.

> **Read §6.1 and §7 before using this on physics.** This package corrects the
> uncertainty a track *reports*, not the resolution it actually *has*. If the
> resolution is what is mismodelled, applying this correction makes the analysis
> worse while every internal validation check reports success. §7 gives the
> measurement, performable in data without truth, that tells the two cases
> apart.

---

## 1. The problem

### 1.1 What a track covariance matrix is

When CMS reconstructs a charged particle, it does not just report a trajectory.
It reports five parameters *and* an estimate of how uncertain each of them is,
including how the uncertainties are correlated with each other. Those five
parameters, in the "curvilinear" convention CMS stores them in, are:

| index | parameter | meaning |
|---|---|---|
| 0 | `qoverp` | charge divided by momentum, q/p |
| 1 | `lambda` | dip angle (π/2 − θ), i.e. how steeply the track goes forward |
| 2 | `phi` | azimuthal direction |
| 3 | `dxy` | transverse impact parameter — how far the track misses the reference point in the x–y plane |
| 4 | `dsz` | longitudinal impact parameter |

The uncertainties live in a 5×5 symmetric matrix Σ, the **covariance matrix**.
Its diagonal entries are the squared uncertainties, Σ_ii = σ_i². Its
off-diagonal entries say how the errors are correlated: for example, a track
whose momentum is overestimated will typically also have its impact parameter
pulled in a particular direction, and Σ_03 encodes that.

Because Σ is symmetric, it has 15 independent numbers, not 25: the 5 diagonal
entries plus the 10 above the diagonal.

Σ is not decoration. It is the *weight* that every downstream fit uses. When you
fit a vertex from three muons, the fitter asks each track "how confident are
you about where you are?", and Σ is the answer. Get Σ wrong and the vertex
position, its χ², and everything derived from it are wrong too.

### 1.2 Why it matters here

In the R(J/ψ) analysis, quantities derived from Σ are not incidental — they are
load-bearing:

- the flight distance significance **L_xy/σ(L_xy)**,
- the bachelor muon's impact parameter significance **IP3D/σ(IP3D)**, which is
  used to *define fit categories*,
- the vertex χ², used in the selection.

Every one of those is a ratio of a measured quantity to an uncertainty taken
from Σ. If the simulation reports uncertainties that are, say, 10% too small,
then every significance in MC is 10% too large, the MC migrates into the wrong
categories, and the template shapes the fit relies on are wrong — in a way that
does not show up as an obvious disagreement in any single plot.

### 1.3 The specific failure

Simulation does not reproduce the detector's reported uncertainties perfectly.
The tracker material, alignment, hit resolutions, and the number and quality of
hits on each track are all modelled approximately. The result is that the
distribution of Σ in MC differs from the distribution of Σ in data.

You have already seen and partially fixed one instance of this: **σ_dxy is
mismodelled**, and you correct it with a 1D quantile morphing (described in
§2.1). This project generalises that fix to the whole matrix.

---

## 2. Why the obvious approaches are not enough

### 2.1 One-dimensional quantile morphing

The standard fix for "MC distribution of a variable ≠ data distribution" is
**quantile morphing**, sometimes called histogram or CDF matching. Take an MC
value, ask what quantile it sits at in the MC distribution (say, the 30th
percentile), and replace it with the value sitting at that same quantile in
data:

```
x_corrected = CDF_data⁻¹( CDF_mc(x) )
```

This is exact, in the sense that if you apply it to every MC event, the
corrected MC distribution equals the data distribution by construction. It is
what you already do for σ_dxy.

Its limitation is that it works on **one variable at a time**. If you apply it
independently to σ_qoverp, σ_lambda, σ_phi, σ_dxy, σ_dsz and the ten
correlations, you will get all fifteen *marginal* distributions right and the
*joint* distribution wrong. Concretely: in data, tracks with large σ_dxy might
also tend to have large σ_dsz; morphing each variable separately reproduces both
distributions individually but destroys the relationship between them. Since the
vertex fit uses the whole matrix at once, that relationship matters.

### 2.2 Just rescaling one uncertainty

`TrackCovUtils.h` provides `scaleParameterUncertainty()`, which multiplies one
parameter's row and column by a factor — this inflates σ_dxy while keeping
correlations consistent. It is the right tool for a quick study, and it is what
the existing 1D correction feeds into.

But a single multiplicative factor cannot describe a correction that depends on
where in the distribution you are (the tails might need more correction than the
core), and it does not touch the correlation structure at all.

### 2.3 What we actually want

A map that takes the *whole* 15-number description of a track's covariance and
transports it, as a joint object, from the MC distribution to the data
distribution — while depending on the track's kinematics and hit content,
because the mismodelling is not the same for a 3 GeV forward track with 9 hits
as for a 20 GeV central track with 22 hits.

That is what a conditional normalising flow gives you.

---

## 3. The method

### 3.1 What a normalising flow is

A normalising flow is a neural network that learns an **invertible** map between
a complicated distribution you care about and a simple one you can write down —
here a standard Gaussian, N(0, I).

Write the map as f. It takes a data point y and returns a "latent" point z:

```
z = f(y),      y = f⁻¹(z)
```

The network is trained so that when you push your whole sample through f, the
resulting z values look like independent unit Gaussians. Because f is invertible
by construction (that is the architectural constraint that makes it a *flow*),
you can always go back.

The useful way to think about it: **f is a multi-dimensional generalisation of
the CDF.** In one dimension, the function that maps a distribution to a uniform
(or, after a further fixed transformation, to a Gaussian) *is* the CDF. In many
dimensions there is no unique CDF, but a flow gives you a workable stand-in that
handles all the correlations.

We use a *neural spline flow* (NSF), which builds the map out of monotonic
spline transformations — flexible enough to capture sharply peaked or skewed
distributions, and monotonic so invertibility is guaranteed.

### 3.2 The correction

Train **two** flows on the same 15-dimensional feature space:

```
f_mc   : trained on simulation, learns to map MC covariances    → N(0, I)
f_data : trained on real data,  learns to map data covariances  → N(0, I)
```

Then correct an MC track by going up through one and down through the other:

```
y_corrected = f_data⁻¹( f_mc(y) )
```

Read it out loud: *"take this MC covariance; ask where it sits in the MC
distribution; then take the covariance sitting at that same place in the data
distribution."*

That is exactly the logic of quantile morphing — `CDF_data⁻¹(CDF_mc(x))` — with
the flows playing the role of the CDFs. The difference is that it now happens in
15 dimensions simultaneously, so **correlations between covariance elements get
corrected too**, which is precisely what §2.1 could not do.

If both flows were perfect, corrected MC would follow the data distribution
exactly.

### 3.3 Conditioning: making the correction depend on the track

The mismodelling depends on the track. A correction averaged over all tracks
would be a blend, applying the forward-region fix to central tracks and vice
versa.

So both flows are **conditional**: they take extra inputs c (the "context") and
learn a *different* map for every value of c.

```
z = f_mc(y | c),      y_corrected = f_data⁻¹( z | c )
```

Note that c is the *same* on both sides — we correct a track by comparing it to
data tracks with the same kinematics and hit content.

Default context: **pt, η, number of valid hits, number of primary vertices.**

Why those:
- **pt and η** set the geometry and the amount of material traversed;
- **number of valid hits** is the strongest single driver. A track with 20 hits
  has genuinely smaller uncertainties than one with 9. If you do not condition
  on hit count, the flow averages over tracks with different hit patterns and
  the correction is a smear of several different corrections;
- **number of primary vertices** carries the pileup dependence.

**The consequence of conditioning, which must be stated explicitly:** you are
correcting the covariance *given* the hit pattern. If data and MC disagree about
the *distribution of hit patterns itself*, this method will not fix that — it
sails straight through untouched. That is usually what you want, because
hit-pattern mismodelling is a separate, tracker-level problem that should be
fixed at its source rather than absorbed here. But if hit content is badly
mismodelled, this correction inherits the problem.

### 3.4 The feature space: guaranteeing a valid matrix

Here is a subtlety that shapes the whole design.

A covariance matrix cannot be just any 15 numbers. It must be **positive
definite** (PD): every variance positive, and the correlations mutually
consistent. (A matrix that is not PD describes an impossible uncertainty — for
example, claiming two quantities are 99% correlated with a third but
uncorrelated with each other.) Feed a non-PD matrix to a vertex fitter and it
will fail, return nonsense, or crash.

A neural network outputs arbitrary real numbers. Nothing stops it emitting 15
numbers that do not form a valid matrix. You could try to detect and reject
those, but then your correction silently drops tracks — a selection bias
disguised as a technical fix.

**The solution is to change coordinates so that the constraint disappears.**
Instead of feeding the flow raw covariance elements, we map each matrix to an
unconstrained vector y ∈ ℝ¹⁵:

```
y[0:5]  = log σ_i                              (one per parameter)
y[5:15] = atanh(canonical partial correlations) (ten of them)
```

Why this works:

- **`log σ` instead of `σ`.** A variance must be positive. A logarithm can be
  any real number, and `exp` of any real number is positive. So the constraint
  "σ > 0" is now automatically satisfied whatever the network emits.
- **`atanh` of correlations.** A correlation must lie between −1 and +1. `tanh`
  maps any real number into (−1, 1), so again the constraint is free.
- **"Canonical partial correlations" (CPCs)** rather than the ordinary
  correlations. This is the part that does the real work. Ordinary correlations
  are not independently free: you cannot set ρ₁₂ = 0.9 and ρ₁₃ = 0.9 and then
  choose ρ₂₃ freely, because some combinations are geometrically impossible.
  CPCs are a reparameterisation (read off the Cholesky factor of the correlation
  matrix) in which all ten numbers *are* independently free in (−1, 1). Any
  combination gives a valid correlation matrix.

Put together: **every point of ℝ¹⁵ maps back to a valid, positive-definite
covariance matrix.** Positive definiteness is structural — a property of the
coordinate system, not something the network has to learn or something we check
and enforce afterwards. The flow cannot produce an invalid matrix even in
principle.

A pleasant side effect: this puts the σ_dxy mismodelling into a single
component, `y[3] = log σ_dxy`. That makes `--active-features 3` — correct only
that one feature and pass the other fourteen through untouched — a direct
apples-to-apples cross-check against the existing 1D correction.

### 3.5 Element ordering (this becomes part of your file format)

The 15 independent elements are stored in the fixed order defined by
`trkcov::covIndexPairs()` in `TrackCovUtils.h`:

```
(0,0)(0,1)(0,2)(0,3)(0,4) (1,1)(1,2)(1,3)(1,4) (2,2)(2,3)(2,4) (3,3)(3,4) (4,4)
```

with matching branch names `qoverp_qoverp`, `qoverp_lambda`, … , `dsz_dsz`.
`features.PACK_NAMES` in this package reproduces that list exactly, so the
Python and the C++ agree by construction.

Once you write ntuples using this ordering, **it is part of the file format**.
Changing it later silently reinterprets every existing file.

---

## 4. Installation and running

Needs `torch`, `zuko`, `uproot`, `awkward`, `numpy`, `matplotlib`.
`onnx`/`onnxruntime` only for the deployment export.

```bash
pip install torch zuko uproot awkward matplotlib onnx onnxruntime
```

### 4.1 Check it works, without any ROOT files

```bash
python tests/smoke_test.py
```

This builds a toy MC sample and a toy "data" sample that differ by a **known,
injected amount** (σ_dxy 10% larger in data, one correlation shifted by 0.15,
plus an η-dependent difference), runs the entire chain, and asserts that the
correction recovers the injected difference. If this fails, the installation or
the code is broken; nothing else is worth trying.

### 4.2 On real ntuples

```bash
python train_covflow.py \
    --data '/path/data*.root' --mc '/path/mc*.root' \
    --tree Events \
    --cov-prefix trk_cov_ \
    --context trk_pt trk_eta trk_nValidHits nPV --log-pt trk_pt \
    --data-weight-branch sweight_jpsi \
    --out covflow_run1
```

Argument by argument:

- `--cov-prefix` — the loader reads 15 branches named `<prefix><name>` for each
  name in `PACK_NAMES`, e.g. `trk_cov_qoverp_qoverp` … `trk_cov_dsz_dsz`. If
  your ntuple stores one candidate per event with fixed roles (`mu1_cov_*`,
  `mu3_cov_*`), run once per role.
- `--context` — the conditioning branches (§3.3).
- `--log-pt` — names the context branch to be replaced by its logarithm; pt
  spans orders of magnitude and behaves much better in log.
- `--data-weight-branch` — the sWeights (§6.4). Omit for unit weights.
- `--active-features` — restrict the correction to a subset of the 15 features.
  `3` = σ_dxy only; `0 1 2 3 4` = all scales, no correlations; `5 6 … 14` =
  correlations only. Used for cross-checks and systematics (§8).

Both flows are trained on the *same* standardisation (mean/σ computed once on
the pooled sample) so that their latent spaces are directly comparable — that
shared frame is what makes composing f_mc with f_data⁻¹ meaningful.

### 4.3 What it writes

| file | what it is |
|---|---|
| `flow_mc.pt`, `flow_data.pt` | the two trained flows |
| `scalers.json` | the shared standardisation; needed to reproduce the correction |
| `distill_mlp.pt`, `distill_size.npy` | the small deployable network (§9) |
| `corrector.onnx` | that network, exported for CMSSW |
| `report.json` | every validation number (§5) |
| `marginals.pdf` | data vs MC vs corrected MC, all 15 features |
| `correction_size.pdf` | how far the correction moves each track (§6.2) |

---

## 5. How to tell whether it worked

`report.json` and the driver's summary give the checks below, ordered roughly by
how easily each one fails. Read them in order — a failure early on makes the
later numbers meaningless.

### `roundtrip_error` — want ≲ 1e-8

Takes your actual matrices, converts them to features and back, and reports the
largest relative discrepancy. This tests only the coordinate change of §3.4, no
machine learning involved. If this is large, the parameterisation is breaking on
your matrices (typically near-singular ones) and nothing downstream can be
trusted.

### `invalid_matrix_fraction_mc` / `_data` — want tiny

The fraction of *input* matrices that are already not positive definite, usually
from failed fits. **Do not mask this.** If it is not tiny, something is wrong
upstream in the reconstruction or the ntuple, and silently dropping those tracks
hides a real problem.

### `latent_mean_mc`, `latent_std_mc` — want ≈ 0 and ≈ 1

Push MC through f_mc and look at the latent values. By construction they should
be standard Gaussian. If they are not, the flow has not converged, and the
correction is built on a map that does not do what it claims. Train longer, or
with more capacity.

### `pd_fraction_after_correction` — want exactly 1.0

The fraction of corrected matrices that are positive definite. §3.4 guarantees
this mathematically; this check confirms it numerically (catching, e.g.,
floating-point edge cases at extreme correlations).

### `latent_clip_fraction` — want small

The fraction of MC tracks whose latent coordinates land outside the region the
data flow ever saw. For those tracks, `f_data⁻¹` is extrapolating, and
extrapolation from a neural network is not trustworthy. A large value means MC
populates a region of covariance space that data does not — worth understanding
before correcting anything.

### `marginal_w1_before` → `marginal_w1_after` — want a large improvement

The Wasserstein-1 distance ("earth mover's distance") between MC and data for
each of the 15 features, before and after correction. This measures how much
work it takes to reshape one distribution into the other, so smaller is better.
These are the *marginals* — the part 1D morphing would also fix.

### `corr_max_dabs_before` → `_after` — want improvement

The largest disagreement between MC and data in the *correlations among the 15
features*. **This is the part 1D quantile morphing structurally cannot reach**
(§2.1), so it is the check that justifies doing any of this.

### `classifier_auc` — want ≫ 0.5 before, ≈ 0.5 after

**The only test here that can really fail, and the one to look at first if you
only look at one.**

Train a small neural network to distinguish MC covariances from data
covariances. AUC (area under the ROC curve) measures how well it succeeds: 1.0
is perfect separation, 0.5 is complete failure — the classifier cannot tell them
apart at all.

Before correction, the AUC should be well above 0.5: that is the mismodelling
you are trying to fix, and if it is already ≈ 0.5 there was nothing to correct.
After correction it should fall to ≈ 0.5.

The power of this test is that it looks at all 15 dimensions *jointly* and is
free to use any structure it can find. All the other checks look at things we
chose to look at; this one searches for whatever is left. If every marginal
looks perfect but the AUC is still 0.7, there is joint structure you have not
fixed.

### `coverage_worst` — want small

The fraction of MC tracks whose features fall outside the range spanned by data.
Same concern as `latent_clip_fraction`, expressed in physical features.

### `classifier_auc_after_distilled` — want ≈ 0.5

The classifier test rerun using the small deployable network (§9) rather than
the flows. **This is the gate that says it is safe to ship**, because this is
the correction that will actually run in CMSSW.

### `distillation.worst_residual_over_width` — want a few %

How faithfully the small network reproduces the flow-based correction, in units
of each feature's own width.

---

## 6. Ways this can mislead you

These matter more than the code does. Each is a real way to get a plausible
looking result that is wrong.

### 6.1 A corrected *uncertainty* is not a corrected *resolution*

This is the most important caveat in the document.

There are two different things that can be wrong in simulation:

1. the **reported uncertainty** σ — what the reconstruction claims;
2. the **actual resolution** — how much the reconstructed value really scatters
   around the true value.

This method corrects **only the first**. It changes what the track says about
its own uncertainty; it does not change how well the parameters were actually
measured.

If the real problem is the resolution, this correction makes things *worse in a
hidden way*. Suppose MC underestimates both σ_dxy and the true dxy scatter.
Inflating σ_dxy alone leaves the numerator of the pull dxy/σ_dxy unchanged while
growing the denominator — so the pull distribution gets *narrower*, moving
further from data even as the σ_dxy distribution looks perfect.

**Before applying this, measure the pull width in data and MC** and decide which
of the two is mismodelled. You may need a parameter smearing alongside this
correction. The two are not interchangeable and one does not substitute for the
other.

You have no truth in data, so how to measure this at all is a real question with
a real answer — **§7 is devoted to it**, including the case where every check in
§5 passes and the correction is nonetheless doing damage.

### 6.2 The per-track map is not unique

The flows guarantee that the corrected *distribution* matches data. They
guarantee nothing about which corrected track any individual MC track is paired
with. Many different pairings produce the same distribution; which one you get
depends on the architecture and the training.

So a correction that gets the marginals right by moving individual tracks a long
way is not trustworthy at the per-track level, even though every distributional
check passes. `correction_size.pdf` exists for exactly this: you want the
per-track corrections to be **narrow and smooth**. A broad or multi-peaked
correction-size distribution means the map is doing something violent, and any
per-track use of the result (which is what a vertex refit is) is suspect.

### 6.3 Where in the chain you apply it

Correcting the covariance and refitting changes the vertex χ², L_xy, L_xy
significance, IP3D and IP3D significance. Those are **selection variables and
category definitions**, not just plotted quantities. IP3D significance defines
fit categories in this analysis.

Therefore the correction must be applied **before** the cuts and the
categorisation that use those variables — i.e. inside the reconstruction chain,
not to the final ntuple. Applying it at the end means events were selected and
categorised on uncorrected quantities and only the plotted variable moved, which
is worse than not correcting at all, because it looks like it worked.

### 6.4 Background in the data sample

The data sample is not pure signal. Training the data flow on raw data would
teach it the covariance distribution of signal *plus background*.

Use **sWeights** from the J/ψ mass fit (`--data-weight-branch`), which
statistically subtract the background on a per-event basis. sWeights can be
negative, which most training procedures cannot handle; the loader clips them at
zero by default and **reports the negative-weight fraction**. Clipping biases
the target if that fraction is not small — the driver warns above 2%.

### 6.5 Conditioning hides what you condition on

Restated from §3.3 because it belongs in this list: any data/MC difference in
the *distribution of the context variables themselves* is invisible to this
method and survives it completely.

---

## 7. Measuring the pull: is it the uncertainty or the resolution?

§6.1 says you must find out whether the *reported uncertainty* or the *actual
resolution* is mismodelled before applying this correction. This section says
how, in data, where there is no truth.

### 7.1 Two simplifications that make it tractable

Write **s** for the actual scatter of a reconstructed parameter around its true
value (the resolution) and **σ** for the uncertainty the reconstruction reports
for it. The obvious plan — measure s in data — needs truth. Two observations
remove that need.

**You need the pull, not the resolution.** The quantity that controls every
significance in the analysis is the ratio

```
w = s / σ        ("pull width")
```

IP3D/σ(IP3D) is miscategorised if and only if w is wrong. You never need s and σ
separately, and w is directly measurable as the width of a pull distribution.

**You need agreement, not correctness.** The analysis requires w_MC = w_data. It
does *not* require w = 1. If data and MC both sit at w = 1.1, the templates are
fine. This matters more than it looks: any bias in the reference you construct —
an imperfect vertex, a residual real lifetime, non-Gaussian tails — **cancels in
the data/MC ratio**, provided the identical procedure is applied to both. That
cancellation is what makes a truth-free measurement possible.

So the measurable is: build a pull in data, build the same pull in MC the same
way, compare the widths.

### 7.2 What a pull distribution requires

```
pull = (measured − reference) / σ_total
```

Three conditions, all of which can silently break the measurement:

1. **An unbiased reference** — a configuration where the true value is known,
   almost always because it is zero by construction.
2. **Statistical independence** between the track and the reference, so their
   uncertainties add in quadrature. In practice this means **the track must be
   excluded from the vertex fit that defines its own reference.** Leave it in and
   the vertex is dragged toward the track, the two errors are correlated, and the
   pull width comes out biased low. This is the single most common way this
   measurement goes wrong.
3. **σ_total propagated consistently**, including the reference's own
   uncertainty: σ²_total = σ²_track + σ²_reference.

The trick that supplies condition (1) without truth: **particles produced
promptly at the primary vertex have zero true impact parameter and zero true
lifetime, exactly** — in data as in simulation. Prompt production is the truth
substitute.

### 7.3 Four probes

**(a) J/ψ mass pull → the momentum and angle block.** The cheapest test, and it
needs only branches you already have. Propagate the two muons' 5×5 covariances
through the Jacobian of m_μμ to get a per-event σ_m, then histogram

```
(m_μμ − m_PDG) / σ_m
```

The J/ψ natural width (93 keV) is negligible against a mass resolution of tens
of MeV, so the observed spread *is* resolution. This probes `qoverp`, `lambda`,
`phi` **and their correlations jointly**, which is exactly the block a
per-variable check cannot reach. Fit the core (double Gaussian, or a truncated
width) because final-state radiation produces a left tail. Repeating with Υ and
Z→μμ maps out the pt dependence.

**(b) Prompt-track impact-parameter pull → `dxy`, `dsz`.** Take tracks from
prompt production, refit the primary vertex *without* the track under study, and
form dxy/σ(dxy) with σ including the PV term. The true impact parameter is zero,
so the width of that distribution is the pull width directly. Bin in 1/pt and η:
σ_dxy behaves roughly as a ⊕ b/pt, and the data/MC ratio is strongly
pt-dependent — which is precisely why covflow conditions on kinematics (§3.3).

**(c) Prompt J/ψ decay-length significance → the analysis observable itself.**
The one to prioritise, because it tests the quantity the fit actually uses
rather than a proxy. Select prompt charmonium (no lifetime), compute
L_xy/σ(L_xy) exactly as the analysis does, and take the width of that
distribution as the pull width for the category variable. The same logic applies
to the bachelor muon: in signal it originates at the Bc vertex, so its true
impact parameter with respect to the three-muon vertex is zero. The control-sample
analogue is prompt J/ψ plus a prompt track, or the impact parameter of one J/ψ
muon with respect to a vertex built from the other muon and the beamspot.

**(d) Track splitting → absolute resolution, no reference at all.** Split each
track's hits (odd/even layers; upper/lower for cosmics), refit the two halves
independently, and take the width of the difference between them: s_half =
width/√2. This uses no vertex and no prompt assumption, making it the orthogonal
cross-check on (a)–(c). It measures half-track resolution, but the *data/MC
ratio* of half-track resolution transfers to full tracks well enough to confirm
the other three.

### 7.4 What truth is legitimately for

Simulation does have truth. Use it to **validate the truth-free estimator**, not
to measure the answer:

1. in MC, compute s directly from truth;
2. in MC, compute s with methods (a)–(d), pretending truth is unavailable;
3. if the two agree, the estimator is unbiased — now run it on data and believe
   the result.

Any residual bias found at step 3 is itself a systematic uncertainty on the
correction.

### 7.5 The decision table

Two ratios decide everything. R_σ = σ_data/σ_MC comes free from the covariance
branches — it is the mismodelling covflow is built to fix. R_w = w_data/w_MC
comes from the pull measurements above. The resolution ratio follows as
R_s = R_σ · R_w.

| observation | diagnosis | action |
|---|---|---|
| R_σ ≠ 1, R_w ≈ 1 | reported σ *and* real scatter are off by the same factor | **both**: covflow **and** a parameter smearing by the same factor |
| R_σ ≠ 1, R_w ≠ 1, with R_s ≈ 1 | only the reported σ is wrong; resolution is fine | covflow alone — the case it is designed for |
| R_σ ≈ 1, R_w ≠ 1 | only the resolution is wrong | smearing alone; covflow correctly does nothing (AUC already ≈ 0.5) |
| both ≠ 1, unrelated | both wrong, independently | smear to fix R_s, then covflow to fix R_σ, then verify pull closure |

**The first row is the trap, and it deserves to be understood in detail.** There,
the pull widths already agree between data and MC. Every check in §5 fires
exactly as it would on a genuine success: the σ distributions really do differ,
the classifier AUC is well above 0.5, the correction moves the distributions into
agreement, the AUC falls to 0.5, the marginals and correlations all improve. The
report looks like a clean win.

But the correction has changed the *denominator* of the pull without touching the
numerator. w_MC, which previously agreed with data, has just been broken — and
IP3D significance is a category variable, so events now migrate between fit
categories in the wrong direction. **No number in covflow's validation suite can
detect this.** Only the pull measurement of this section can. That is why it is a
prerequisite for using the package, not a cross-check to be run afterwards.

### 7.6 Applying both corrections, if it comes to that

Order matters: **smear first, correct second.**

Smearing changes the reconstructed parameter *values*, and therefore dxy, L_xy
and the vertex fit itself. Covflow changes the reported *uncertainties*. So apply
the parameter smearing before the vertex refit, then the covariance correction,
then the selection and categorisation — with both sitting upstream of every cut
that uses a significance (§6.3).

Then re-run the pull measurement on the corrected MC as a closure test. The
target is w_MC = w_data **and** agreement of the σ distributions. Achieving one
without the other means the problem has been moved, not fixed.

### 7.7 Pitfalls

- **Tails dominate an RMS.** Fit a core width or use a quantile-based width, and
  use the identical definition in data and MC.
- **Pileup contamination of the PV** biases methods (b) and (c). Use the same
  PV-association rule as the analysis.
- **Alignment weak modes** bias data only and can fake a resolution difference.
  Check that the pull width is flat in φ and in charge.
- **Residual non-prompt contamination** in a "prompt" sample inflates the width.
  Fit for it with a separate long-lived component rather than cutting it away —
  cutting on displacement sculpts the very distribution being measured.
- **The track must not be in its own reference vertex** (§7.2, condition 2).
## 8. Systematic uncertainties

The interface gives you the natural variations directly:

- **Statistical uncertainty of the correction** — train on two statistically
  independent halves of the data and take the spread of the results.
- **Applying it or not** — evaluate at 0% and 100%. If you want a continuous
  interpolation, interpolate in *feature* space (between y and y_corrected), not
  in the corrected observable, so that every intermediate point is still a valid
  covariance matrix.
- **Choice of context** — rerun with a different `--context` set.
- **Which part of the matrix you trust** — `--active-features` gives you
  scale-only (`0 1 2 3 4`), correlation-only (`5 … 14`), and σ_dxy-only (`3`)
  variants at no extra cost.

---

## 9. Deployment, and why there is a second, smaller network

The trained flows work, but shipping them into CMSSW is awkward: exporting the
full spline machinery to ONNX drags a large and fragile graph through the
converter, and it is slow to evaluate per track.

So `correct.distill()` fits a small plain MLP to *imitate* the flow-based
correction. Two design choices make this work well:

1. **It predicts the residual** (y_corrected − y), not y_corrected. The identity
   map — "change nothing" — is then the zero solution, which is the right
   default for a network to fall back on.
2. **The residual is expressed in units of each feature's own correction size.**
   Without this, features that need large corrections dominate the loss and
   features that need small ones are ignored; in these units all fifteen get
   comparable attention.

The result is a graph containing nothing but `Gemm` (matrix multiply) and
`Sigmoid` operations, microseconds per track, exported as a single
self-contained `.onnx` file. The driver verifies that the exported file
reproduces the PyTorch model numerically, and — crucially — **reruns the
classifier test using the distilled correction**. That number, not the flow's
own, is what decides whether this is safe to ship.

In CMSSW the corrected matrix is handed to `trkcov::trackWithCovariance()` from
`SetTrackCovarianceBeforeFit_snippet.cc`, which builds a new `reco::Track`
carrying the corrected covariance (`reco::Track` has no in-place covariance
setter by design). That track goes into the `TransientTrackBuilder` and then the
vertex fit exactly like an uncorrected one — nothing downstream needs to change.

---

## 10. Package layout

```
covflow/features.py   covariance ↔ unconstrained ℝ¹⁵ (§3.4), + log-Cholesky variant
covflow/data.py       uproot loading, context, sWeights, shared standardiser, toy generator
covflow/flows.py      conditional neural spline flow (zuko) + weighted-NLL training
covflow/correct.py    the morph (§3.2), distillation and ONNX export (§9)
covflow/validate.py   every check in §5, plus the plots
train_covflow.py      driver: load → train → validate → distil → export
tests/smoke_test.py   end-to-end closure on synthetic data, no ROOT needed
```

### Two parameterisations

`logsigma_corr` (the default, §3.4) bounds correlations to (−1, 1) by
construction. `log_cholesky` is simpler — it takes the logarithm of the diagonal
of the Cholesky factor and leaves the off-diagonals raw — but those
off-diagonals are unbounded relative to the scale, so a flow that wanders can
produce matrices that are positive definite on paper and numerically singular in
practice. Use the default; `log_cholesky` is there as a cross-check.

---

## 11. Verification status

The closure test of §4.1 injects a known difference and checks it is recovered
(20 000 tracks per sample, 25 epochs):

```
feature round trip           max rel error 1.2e-11
latent convention            mean +0.021, std 0.994, round trip 7.3e-07
PD after correction          100.0000%
log_sigma_dxy    W1          0.177 → 0.005     (35×)
pcorr_phi_lambda W1          0.186 → 0.003     (62×)
mean W1 over 15 features     0.026 → 0.006
median σ_dxy  MC/data        0.863 → 1.015
classifier AUC               0.858 → 0.542
distilled: worst residual    5.0% of feature width, AUC 0.555
ONNX vs torch                9.5e-07; ops = [Gemm, Sigmoid]; single file
```

Checked independently:

- the coordinate change round-trips to ~1e-12 on random positive-definite
  matrices spanning seven decades of scale, in both parameterisations;
- positive definiteness holds at 100% across realistic feature ranges (the only
  failures are for deliberately near-singular inputs, |CPC| → 1, which the
  validation flags);
- the flow library's direction convention was verified explicitly rather than
  assumed — `flow(c).transform(y)` maps data → latent, the base distribution is
  a diagonal Gaussian, and the log-probability decomposition is exact. Had this
  been backwards, the correction would have run and produced nonsense;
- the weighted statistics were checked against known answers: the weighted AUC
  reproduces `sklearn.roc_auc_score` to three decimals and gives the analytic
  0.75 on a tie-heavy case; weighted W1 recovers an injected shift of 2.0 as
  1.993; weighted correlation recovers 0.6 as 0.599.

---

## 12. Open items

- **Confirm the branch names.** The loader assumes `<cov-prefix><name>` for the
  names in `PACK_NAMES`. Confirm against the actual ntuple before the first real
  run.
- **Confirm the sWeight branch** for data, and whether MC needs a generator
  weight branch (`--mc-weight-branch`).
- **Hit-content branches are not in the skim yet.** `numberOfValidHits()`,
  `hitPattern().numberOfValidPixelHits()` and the PV multiplicity need to be
  written alongside the 15 covariance branches. Per §3.3 this is the single
  biggest lever on the quality of the correction, and it needs the same
  reprocessing that `packedPFCandidates` is already waiting on — worth bundling
  into one pass.
- **The C++ mirror is not in this drop.** A `TrackCovFeatures.h` reproducing
  `features.py` exactly, plus a Python/C++ parity test, is required before
  anything runs inside CMSSW. The Python side is deployment-ready.
- **Measure the pulls first.** §6.1 and §7 are a prerequisite for interpreting
  any result from this package, not a footnote to it. Concretely: the J/ψ mass
  pull (§7.3a) costs nothing beyond branches already in the ntuple, and the
  prompt decay-length significance (§7.3c) tests the category variable itself.
  Neither has been run yet.
- Distillation quality on the toy is a 5–6.5% worst-feature residual with 200
  epochs on 20 000 tracks; on real statistics, train it longer and re-check
  `classifier_auc_after_distilled`.
