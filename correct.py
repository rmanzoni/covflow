"""
covflow.correct
===============

The correction itself and its deployable distillation.

morph()      packed MC covariance -> packed corrected covariance, per track,
             via  y_corr = f_data^{-1}( f_mc(y | c) | c )  in standardised
             feature space, then back to a (guaranteed PD) covariance matrix.

distill()    fit a small MLP that reproduces the morph in one forward pass, so
             deployment does not drag the whole spline machinery through ONNX.
             It predicts the *residual* (y_corr - y) in units of the per-feature
             correction size, so the identity map is the zero solution and
             features that barely move are not drowned out in the loss.

export_onnx() write the distilled MLP to ONNX (plain Gemm/Sigmoid graph).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from . import features as F
from . import flows as FL
from .flows import to_numpy


# ---------------------------------------------------------------------------
# the morph
# ---------------------------------------------------------------------------

def morph(packed_mc, C_mc, flow_mc, flow_data, scaler,
          param="logsigma_corr", active_features=None, feature_indices=None,
          device="cpu", chunk=2_000_000, out_dtype=None):
    """
    Correct MC covariances to the data distribution.

    packed_mc : (N,15) MC packed covariance (features.PACK_NAMES order)
    C_mc      : (N,k)  raw MC context
    scaler    : shared Standardiser
    feature_indices : the subspace the flows were BUILT and TRAINED in (see
                      features.SUBSETS). Features outside it are copied from MC
                      unchanged. Must match what was passed to build_flow.
    active_features : optional further mask applied to the flow's output, for a
                      full-dimensional flow whose correction you want to apply
                      only partially. Indices refer to the full 15.
    chunk           : rows processed at a time. Everything below is per-row
                      arithmetic, so this changes nothing but the peak. Note
                      that it also caps the flow's internal evaluation batch
                      (`data_to_latent` defaults to 200k), and THAT is what
                      dominates: the spline knot tensors are tens of kB per row
                      in flight, so this one number sets both the host and the
                      device peak.
    out_dtype       : dtype of the three returned arrays. None follows
                      `packed_mc`: float32 in, float32 out.

    Returns (packed_corr, y_mc, y_corr) with y_* the *unstandardised* full
    15-component features.

    WHY THIS IS CHUNKED

    The unchunked version held, simultaneously and at full length: the (N,5,5)
    float64 matrices twice (once per direction of the transform, 200 B/row
    each), y_mc, ys_mc, ys_corr, y_corr, the np.where copy and packed_corr
    (120 B/row each in float64), plus the float32 torch tensors. Something like
    a kilobyte per track of transient, on top of the sample itself -- which is
    why the resident set kept climbing long after the loader had finished.

    Chunking caps all of that at `chunk` rows. It cannot change the result:
    every step is row-independent, and `data_to_latent` / `latent_to_data`
    already batch internally.
    """
    to_feat, to_mat = F.get_transforms(param)
    packed_mc = np.asarray(packed_mc)
    C_mc = np.asarray(C_mc)
    n = len(packed_mc)

    if out_dtype is None:
        dt = np.dtype(np.float32) if packed_mc.dtype == np.float32 \
            else np.dtype(np.float64)
    else:
        dt = np.dtype(out_dtype)

    idx = F.subset_indices(feature_indices)
    mask = None
    if active_features is not None:
        mask = np.zeros(F.N_FEATURES, dtype=bool)
        mask[list(active_features)] = True

    y_mc = np.empty((n, F.N_FEATURES), dtype=dt)
    y_corr = np.empty((n, F.N_FEATURES), dtype=dt)
    packed_corr = np.empty((n, F.N_FEATURES), dtype=dt)

    step = int(chunk) if chunk else max(n, 1)
    for s in range(0, n, step):
        e = min(s + step, n)
        y = F.packed_to_features(packed_mc[s:e], param, out_dtype=dt)
        ys = scaler.x(y)
        cs = scaler.c(C_mc[s:e])

        z = FL.data_to_latent(flow_mc, ys[:, idx], cs, device=device)
        ys[:, idx] = FL.latent_to_data(flow_data, z, cs, device=device)
        del z
        yc = scaler.x_inv(ys)

        if mask is not None:
            yc = np.where(mask[None, :], yc, y)

        y_mc[s:e] = y
        y_corr[s:e] = yc
        packed_corr[s:e] = F.matrix_to_packed(to_mat(yc))

    return packed_corr, y_mc, y_corr


def correction_size(y_mc, y_corr):
    """Per-feature robust correction size (IQR of y_corr - y), for weighting
    the distillation loss and for the correction_size.pdf diagnostic."""
    d = y_corr - y_mc
    q75, q25 = np.percentile(d, [75, 25], axis=0)
    return np.maximum(q75 - q25, 1e-6)


# ---------------------------------------------------------------------------
# distillation
# ---------------------------------------------------------------------------

class ResidualMLP(torch.nn.Module):
    """Predicts the standardised residual; deploy graph is Gemm/Sigmoid only."""
    def __init__(self, n_in, hidden=(64, 64), n_out=F.N_FEATURES):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.Sigmoid()]
            d = h
        layers += [torch.nn.Linear(d, n_out)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def distill(y_mc, y_corr, C_mc, scaler,
            hidden=(64, 64), epochs=200, lr=2e-3, batch=8192,
            device="cpu", verbose=True):
    """
    Fit ResidualMLP: input = [standardised features, standardised context],
    target = (y_corr - y_mc) / correction_size. Returns (model, size, info).
    """
    size = correction_size(y_mc, y_corr)
    ys = scaler.x(y_mc)
    cs = scaler.c(C_mc)
    inp = np.concatenate([ys, cs], axis=1).astype(np.float32, copy=False)
    tgt = ((y_corr - y_mc) / size).astype(np.float32, copy=False)

    dev = torch.device(device)
    model = ResidualMLP(inp.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr = torch.as_tensor(inp); Ytr = torch.as_tensor(tgt)
    g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        order = torch.randperm(len(Xtr), generator=g)
        tot = 0.0
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            opt.zero_grad()
            pred = model(Xtr[b].to(dev))
            loss = ((pred - Ytr[b].to(dev)) ** 2).mean()
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if verbose and (ep % 25 == 0 or ep == epochs - 1):
            print(f"[distill] epoch {ep:3d}  mse {tot/len(Xtr):.5f}")

    # residual quality: worst per-feature RMS residual as a fraction of width
    model.eval()
    # in batches: the whole training set in one forward pass is a GPU
    # allocation proportional to the sample, which is exactly the thing that
    # falls over first on a full-statistics run
    chunks = []
    with torch.no_grad():
        for s in range(0, len(Xtr), 200_000):
            chunks.append(to_numpy(model(Xtr[s:s + 200_000].to(dev))))
    pred = np.concatenate(chunks, 0) * size + y_mc
    del chunks
    resid = np.sqrt(((pred - y_corr) ** 2).mean(0))
    width = np.maximum(np.percentile(y_corr, 84, 0) - np.percentile(y_corr, 16, 0), 1e-9)
    info = {"residual_over_feature_width": (resid / width).tolist(),
            "worst_residual_over_width": float(np.max(resid / width))}
    return model, size, info


def distilled_apply(model, size, packed_mc, C_mc, scaler,
                    param="logsigma_corr", device="cpu"):
    """Apply a distilled model the same way ApplyCovFlowONNX would in CMSSW."""
    to_feat, to_mat = F.get_transforms(param)
    y_mc = to_feat(F.packed_to_matrix(packed_mc))
    inp = np.concatenate([scaler.x(y_mc), scaler.c(C_mc)], axis=1).astype(np.float32)
    with torch.no_grad():
        r = to_numpy(model(torch.as_tensor(inp).to(device)))
    y_corr = y_mc + r * size
    return F.matrix_to_packed(to_mat(y_corr)), y_corr


def export_onnx(model, n_in, path, opset=17):
    """
    Write the distilled MLP to a SINGLE self-contained .onnx file.

    torch>=2.x defaults to the dynamo exporter, which happily splits even a
    2 kB model into an external `.onnx.data` blob -- fragile to ship into
    CMSSW. Prefer the legacy exporter (single file, plain Gemm/Sigmoid graph);
    if that is unavailable, fall back to dynamo and then re-save the model
    with its weights inlined.
    """
    model = model.eval()
    dummy = torch.zeros(1, n_in, dtype=torch.float32)
    kw = dict(input_names=["features_context"],
              output_names=["residual_over_size"],
              dynamic_axes={"features_context": {0: "N"},
                            "residual_over_size": {0: "N"}},
              opset_version=opset)
    try:
        torch.onnx.export(model, dummy, path, dynamo=False, **kw)
    except TypeError:                       # older torch: no dynamo kwarg
        torch.onnx.export(model, dummy, path, **kw)
    except Exception:                       # legacy path unavailable
        torch.onnx.export(model, dummy, path, **kw)

    # inline any external data so the single file is self-contained
    try:
        import onnx
        m = onnx.load(path)                 # resolves external data if present
        onnx.save_model(m, path, save_as_external_data=False)
        ext = path + ".data"
        if os.path.exists(ext):
            os.remove(ext)
    except Exception:
        pass
    return path


def onnx_ops(path):
    """Sorted set of operator types in the exported graph (deployment sanity)."""
    import onnx
    m = onnx.load(path)
    return sorted({n.op_type for n in m.graph.node})


def verify_onnx(path, model, n_in, n=512, seed=0, device="cpu"):
    """
    Run random inputs through onnxruntime and through torch; return the max
    absolute disagreement. Exporting is not the same as exporting *correctly* --
    this is the check that the shipped graph is the model you validated.
    """
    import onnxruntime as ort
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n_in)).astype(np.float32)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    o = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    with torch.no_grad():
        t = to_numpy(model(torch.as_tensor(x).to(device)))
    return float(np.max(np.abs(o - t)))
