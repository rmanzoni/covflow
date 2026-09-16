"""
covflow.flows
=============

Conditional neural spline flow (zuko) over the 15 covariance features, trained
by weighted maximum likelihood, plus the two helpers the morph needs:

    data_to_latent(flow, y, c)   y -> z      (f in the README)
    latent_to_data(flow, z, c)   z -> y      (f^{-1})

zuko convention (used below): for `dist = flow(c)`, `dist.transform` maps
DATA -> LATENT, i.e. `dist.transform(y) == z` and `dist.transform.inv(z) == y`;
`dist.log_prob(y)` returns log p(y | c). We rely only on that public contract.

Two independent flows are trained on the SAME feature/context definition and
the SAME shared standardiser (see data.Standardiser):

    flow_mc   fit on MC     (unit or gen weights)
    flow_data fit on data   (sWeights)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

@dataclass
class FlowConfig:
    n_features: int = 15
    n_context: int = 4
    transforms: int = 4
    hidden: tuple = (128, 128)
    bins: int = 8
    seed: int = 0


def build_flow(cfg: FlowConfig):
    """A zuko NSF with `cfg.n_context` conditioning inputs."""
    import zuko
    torch.manual_seed(cfg.seed)
    flow = zuko.flows.NSF(
        features=cfg.n_features,
        context=cfg.n_context,
        transforms=cfg.transforms,
        hidden_features=cfg.hidden,
        bins=cfg.bins,
    )
    return flow


# ---------------------------------------------------------------------------
# weighted-NLL training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 4096
    lr: float = 2e-3
    lr_min_factor: float = 0.02    # cosine anneals lr -> lr*this
    warmup_epochs: int = 2
    schedule: str = "cosine"       # "cosine" or "none"
    weight_decay: float = 0.0
    val_frac: float = 0.15
    patience: int = 8
    clip_grad: float = 10.0
    device: str = "cpu"
    eval_every: int = 1
    verbose: bool = True
    history: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def _weighted_nll(flow, y, c, w):
    logp = flow(c).log_prob(y)             # (B,)
    return -(w * logp).sum() / w.sum().clamp_min(1e-12)


def train_flow(flow, ys, cs, w, tcfg: TrainConfig, tag=""):
    """
    Fit `flow` by weighted maximum likelihood, with a cosine learning-rate
    schedule.

    Why the schedule matters here: at a constant lr the validation NLL of these
    flows swings by O(0.4) nats between consecutive epochs -- more than the
    improvement over the last 15 epochs. Early stopping then fires on noise, and
    the "best epoch" snapshot is a lucky fluctuation rather than a converged
    model. Because the MC and data flows draw independent fluctuations, their
    composition f_data^-1 . f_mc stops being the identity where the two
    distributions agree, and that shows up as a random per-track correction that
    broadens marginals instead of shifting them.

    The returned diagnostics quantify this: `val_noise` is the scatter of the
    validation loss over the last evaluations, and `best_minus_median_sigma`
    says how far the retained snapshot is from a typical recent epoch. Below
    ~1.5 sigma the selection is dominated by noise.
    """
    dev = torch.device(tcfg.device)
    flow = flow.to(dev)
    ys = torch.as_tensor(np.asarray(ys), dtype=torch.float32).to(dev)
    cs = torch.as_tensor(np.asarray(cs), dtype=torch.float32).to(dev)
    w = torch.as_tensor(np.asarray(w), dtype=torch.float32).to(dev)

    n = len(ys)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = int(tcfg.val_frac * n)
    val_idx, tr_idx = perm[:n_val].to(dev), perm[n_val:].to(dev)

    # foreach=False is a CPU optimisation (~17% on Apple silicon, where the
    # multi-tensor path fights the unified-memory allocator). On CUDA it is the
    # wrong sign: it forces one kernel launch per parameter tensor, and an NSF
    # with several transforms is made of many small ones. So pick per device,
    # and degrade gracefully -- `fused` and `foreach` have both been added and
    # had their accepted dtypes changed across torch versions, and the training
    # environment here is not the application environment.
    if str(tcfg.device).startswith("cuda"):
        _opt_kwargs = ({"fused": True}, {"foreach": True}, {})
    else:
        _opt_kwargs = ({"foreach": False}, {})

    opt = None
    for _kw in _opt_kwargs:
        try:
            opt = torch.optim.Adam(flow.parameters(), lr=tcfg.lr,
                                   weight_decay=tcfg.weight_decay, **_kw)
            break
        except (TypeError, RuntimeError, ValueError):
            continue
    if opt is None:                     # cannot happen: the last entry is {}
        raise RuntimeError("could not construct torch.optim.Adam")
    if tcfg.verbose:
        print(f"[{tag}] Adam on {tcfg.device} with "
              f"{_kw if _kw else 'default kernels'}")

    def lr_at(epoch):
        if tcfg.schedule != "cosine":
            return tcfg.lr
        if epoch < tcfg.warmup_epochs:
            return tcfg.lr * (epoch + 1) / max(tcfg.warmup_epochs, 1)
        t = (epoch - tcfg.warmup_epochs) / max(tcfg.epochs - tcfg.warmup_epochs, 1)
        f = tcfg.lr_min_factor
        return tcfg.lr * (f + (1 - f) * 0.5 * (1 + np.cos(np.pi * min(t, 1.0))))

    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in flow.state_dict().items()}
    bad, best_epoch = 0, -1
    vlosses = []

    for epoch in range(tcfg.epochs):
        cur_lr = lr_at(epoch)
        for gparam in opt.param_groups:
            gparam["lr"] = cur_lr
        flow.train()
        order = tr_idx[torch.randperm(len(tr_idx), generator=g).to(dev)]
        for s in range(0, len(order), tcfg.batch_size):
            b = order[s:s + tcfg.batch_size]
            yb, cb, wb = ys[b], cs[b], w[b]
            if wb.sum() <= 0:
                continue
            opt.zero_grad(set_to_none=True)
            loss = _weighted_nll(flow, yb, cb, wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), tcfg.clip_grad)
            opt.step()

        if (epoch + 1) % max(tcfg.eval_every, 1) and epoch != tcfg.epochs - 1:
            continue

        flow.eval()
        with torch.no_grad():
            vloss = _weighted_nll(flow, ys[val_idx], cs[val_idx],
                                  w[val_idx]).item()
        vlosses.append(vloss)
        tcfg.history.append((tag, epoch, vloss, cur_lr))
        if tcfg.verbose:
            print(f"[{tag}] epoch {epoch:3d}  val NLL {vloss:.4f}  lr {cur_lr:.2e}"
                  + ("  *" if vloss < best - 1e-4 else ""))
        if vloss < best - 1e-4:
            best, bad, best_epoch = vloss, 0, epoch
            best_state = {k: v.detach().clone() for k, v in flow.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg.patience:
                if tcfg.verbose:
                    print(f"[{tag}] early stop at epoch {epoch}")
                break

    flow.load_state_dict(best_state)
    flow.eval()

    # convergence diagnostics
    tail = np.array(vlosses[-15:]) if len(vlosses) >= 5 else np.array(vlosses)
    noise = float(tail.std()) if len(tail) > 1 else float("nan")
    med = float(np.median(tail))
    sig = (med - best) / noise if noise > 0 else float("nan")
    slope = (float(np.polyfit(np.arange(len(tail)), tail, 1)[0])
             if len(tail) > 2 else float("nan"))
    diag = {"best_epoch": best_epoch, "best_val_nll": best,
            "val_noise_tail": noise, "best_minus_median_sigma": sig,
            "tail_trend_per_epoch": slope, "n_evals": len(vlosses),
            "stopped_early": bad >= tcfg.patience}
    tcfg.diagnostics[tag] = diag
    if tcfg.verbose:
        print(f"[{tag}] best epoch {best_epoch}, val NLL {best:.4f} | "
              f"tail noise {noise:.3f}, best is {sig:.2f} sigma below median, "
              f"trend {slope:+.4f}/epoch")
        if sig < 1.5:
            print(f"[{tag}] WARNING: the retained snapshot is only {sig:.2f} "
                  f"sigma better than a typical recent epoch -- model selection "
                  f"is dominated by noise, not convergence")
        if slope < -0.01:
            print(f"[{tag}] WARNING: still improving at {slope:+.4f} nats/epoch "
                  f"-- undertrained, increase --epochs")
    return flow, best


# ---------------------------------------------------------------------------
# tensor -> ndarray
# ---------------------------------------------------------------------------

def to_numpy(t):
    """Tensor -> ndarray, including on a torch built without NumPy support.

    CMSSW ships py3-torch built with USE_NUMPY off (2.6.0 on el9_amd64_gcc13),
    where Tensor.numpy() raises RuntimeError. Conversion INTO torch still works,
    so only this direction needs the fallback: .tolist() goes via Python floats,
    which is exact for float32 and costs nothing at the batch sizes an ntuplizer
    uses (tens of tracks per event).

    It is, however, roughly an order of magnitude slower than .numpy() on
    200k-row training batches -- so train where torch has NumPy, and treat this
    as what makes the trained model USABLE inside CMSSW, not as a way to train
    inside it.
    """
    t = t.detach().cpu()
    try:
        return t.numpy()
    except RuntimeError:
        return np.asarray(t.tolist(), dtype=np.float64)


# ---------------------------------------------------------------------------
# latent transforms (the two directions the morph composes)
# ---------------------------------------------------------------------------

@torch.no_grad()
def data_to_latent(flow, ys, cs, device="cpu", batch=200_000):
    """y (standardised features) -> z (latent), under `flow`."""
    dev = torch.device(device)
    flow = flow.to(dev).eval()
    ys = torch.as_tensor(np.asarray(ys), dtype=torch.float32)
    cs = torch.as_tensor(np.asarray(cs), dtype=torch.float32)
    out = []
    for s in range(0, len(ys), batch):
        t = flow(cs[s:s + batch].to(dev)).transform
        out.append(to_numpy(t(ys[s:s + batch].to(dev))))
    return np.concatenate(out, 0)


@torch.no_grad()
def latent_to_data(flow, zs, cs, device="cpu", batch=200_000):
    """z (latent) -> y (standardised features), under `flow`."""
    dev = torch.device(device)
    flow = flow.to(dev).eval()
    zs = torch.as_tensor(np.asarray(zs), dtype=torch.float32)
    cs = torch.as_tensor(np.asarray(cs), dtype=torch.float32)
    out = []
    for s in range(0, len(zs), batch):
        t = flow(cs[s:s + batch].to(dev)).transform
        out.append(to_numpy(t.inv(zs[s:s + batch].to(dev))))
    return np.concatenate(out, 0)


@torch.no_grad()
def log_prob(flow, ys, cs, device="cpu", batch=200_000):
    dev = torch.device(device)
    flow = flow.to(dev).eval()
    ys = torch.as_tensor(np.asarray(ys), dtype=torch.float32)
    cs = torch.as_tensor(np.asarray(cs), dtype=torch.float32)
    out = []
    for s in range(0, len(ys), batch):
        out.append(to_numpy(flow(cs[s:s + batch].to(dev)).log_prob(ys[s:s + batch].to(dev))))
    return np.concatenate(out, 0)


def save_flow(flow, path):
    torch.save(flow.state_dict(), path)


# The base distribution's two fixed buffers are registered under different names
# by different zuko versions: 1.5 passes them positionally to
# UnconditionalDistribution(DiagNormal, ...) so Partial names them _0/_1, while
# 1.6 passes them as keywords so they are named loc/scale. The VALUES are
# zeros(features) and ones(features) in both -- a standard normal, never
# trained -- so a checkpoint written by one version is usable by the other once
# the two keys are renamed. Nothing else in the state dict differs.
#
# This matters because zuko 1.6 requires Python >= 3.10 while CMSSW ships 3.9,
# so "use the same version everywhere" is not available: training happens on
# 1.6 and application, inside CMSSW, on 1.5.
_BASE_ALIASES = {"base.loc": "base._0", "base.scale": "base._1"}


def _align_base_keys(state, expected):
    """Rename the base buffers to whatever THIS zuko calls them.

    Renames only the keys in _BASE_ALIASES, only when the target name is the one
    the model actually wants, and only after checking the buffer still holds the
    standard normal that makes the rename an identity. Everything else must
    already match: a checkpoint differing anywhere but the base is a different
    model, and is left to fail on load_state_dict, loudly.
    """
    forward = dict(_BASE_ALIASES)
    backward = {v: k for k, v in _BASE_ALIASES.items()}

    out = {}
    for key, val in state.items():
        new_key = key
        if key not in expected:
            if key in forward and forward[key] in expected:
                new_key = forward[key]
            elif key in backward and backward[key] in expected:
                new_key = backward[key]
        if new_key != key:
            # _0 / loc is the mean, _1 / scale the width
            want = 0.0 if new_key.endswith("_0") or new_key.endswith("loc") else 1.0
            if not torch.allclose(val, torch.full_like(val, want)):
                raise RuntimeError(
                    f"refusing to rename {key!r} -> {new_key!r}: the buffer is "
                    f"not the constant {want} it must be for the two zuko "
                    f"conventions to be equivalent. This checkpoint's base "
                    f"distribution is not a standard normal, so the rename "
                    f"would silently change the model.")
        out[new_key] = val
    return out


def load_flow(cfg: FlowConfig, path):
    flow = build_flow(cfg)
    state = torch.load(path, map_location="cpu")
    state = _align_base_keys(state, flow.state_dict())
    flow.load_state_dict(state)   # strict: everything else must match exactly
    flow.eval()
    return flow
