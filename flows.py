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
    batch_size: int = 8192
    lr: float = 2e-3
    weight_decay: float = 0.0
    val_frac: float = 0.15
    patience: int = 8
    clip_grad: float = 10.0
    device: str = "cpu"
    verbose: bool = True
    history: list = field(default_factory=list)


def _weighted_nll(flow, y, c, w):
    logp = flow(c).log_prob(y)             # (B,)
    return -(w * logp).sum() / w.sum().clamp_min(1e-12)


def train_flow(flow, ys, cs, w, tcfg: TrainConfig, tag=""):
    """
    Fit `flow` on standardised features `ys` (N,15), context `cs` (N,k),
    weights `w` (N,). Early-stops on held-out weighted NLL; restores the best
    state. Arrays are numpy; converted here.
    """
    dev = torch.device(tcfg.device)
    flow = flow.to(dev)
    ys = torch.as_tensor(np.asarray(ys), dtype=torch.float32)
    cs = torch.as_tensor(np.asarray(cs), dtype=torch.float32)
    w = torch.as_tensor(np.asarray(w), dtype=torch.float32)

    n = len(ys)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = int(tcfg.val_frac * n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    opt = torch.optim.Adam(flow.parameters(), lr=tcfg.lr,
                           weight_decay=tcfg.weight_decay)

    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in flow.state_dict().items()}
    bad = 0

    for epoch in range(tcfg.epochs):
        flow.train()
        order = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        for s in range(0, len(order), tcfg.batch_size):
            b = order[s:s + tcfg.batch_size]
            yb = ys[b].to(dev); cb = cs[b].to(dev); wb = w[b].to(dev)
            if wb.sum() <= 0:
                continue
            opt.zero_grad()
            loss = _weighted_nll(flow, yb, cb, wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), tcfg.clip_grad)
            opt.step()

        flow.eval()
        with torch.no_grad():
            vloss = _weighted_nll(flow, ys[val_idx].to(dev),
                                  cs[val_idx].to(dev), w[val_idx].to(dev)).item()
        tcfg.history.append((tag, epoch, vloss))
        if tcfg.verbose:
            print(f"[{tag}] epoch {epoch:3d}  val NLL {vloss:.4f}"
                  + ("  *" if vloss < best - 1e-4 else ""))
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in flow.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg.patience:
                if tcfg.verbose:
                    print(f"[{tag}] early stop at epoch {epoch}")
                break

    flow.load_state_dict(best_state)
    flow.eval()
    return flow, best


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
        out.append(t(ys[s:s + batch].to(dev)).cpu().numpy())
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
        out.append(t.inv(zs[s:s + batch].to(dev)).cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def log_prob(flow, ys, cs, device="cpu", batch=200_000):
    dev = torch.device(device)
    flow = flow.to(dev).eval()
    ys = torch.as_tensor(np.asarray(ys), dtype=torch.float32)
    cs = torch.as_tensor(np.asarray(cs), dtype=torch.float32)
    out = []
    for s in range(0, len(ys), batch):
        out.append(flow(cs[s:s + batch].to(dev)).log_prob(ys[s:s + batch].to(dev)).cpu().numpy())
    return np.concatenate(out, 0)


def save_flow(flow, path):
    torch.save(flow.state_dict(), path)


def load_flow(cfg: FlowConfig, path):
    flow = build_flow(cfg)
    flow.load_state_dict(torch.load(path, map_location="cpu"))
    flow.eval()
    return flow
