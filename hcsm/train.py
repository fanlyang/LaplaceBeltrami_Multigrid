"""Energy-based training with full validation and restored best checkpoints.

The legacy ``epochs`` field counts optimizer updates. ``log_every`` counts
updates, while scheduler/early-stop patience counts validation checks. Both
trainers evaluate the complete fixed validation set in bounded batches.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .model import build_features, stage1_loss, to_torch_sparse
from .problem import Problem
from .samplers import Split


@dataclass
class TrainConfig:
    """Explicit update/check units; epochs is a compatibility field name."""

    epochs: int = 3000
    batch_size: int = 32
    val_batch: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    early_stop: int = 25  # consecutive validation checks, not optimizer steps
    lr_patience_checks: int = 8
    min_delta: float = 1e-6
    log_every: int = 100
    seed: int = 0


def validation_means(model, residuals, errors, features, batch_size, loss_fn):
    """Evaluate ALL validation samples, weighting partial batches correctly."""
    if len(errors) == 0 or batch_size < 1:
        raise ValueError("validation requires samples and a positive batch size")
    totals = None
    model.eval()
    with torch.no_grad():
        for start in range(0, len(errors), batch_size):
            end = min(start + batch_size, len(errors))
            values = loss_fn(model(residuals[start:end], features, None), errors[start:end])
            if not isinstance(values, tuple):
                values = (values,)
            values = np.array([float(v) for v in values])
            totals = values * (end - start) if totals is None else totals + values * (end - start)
    return tuple(float(v) for v in totals / len(errors))


def restore_best(model, path):
    """Ensure downstream evaluation uses the weights named by the metadata."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return saved


def _tensors(split: Split, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.asarray(split.E_F, dtype=np.float64)).to(dtype),
        torch.from_numpy(np.asarray(split.R_F, dtype=np.float64)).to(dtype),
    )


def train_smoother(
    model: torch.nn.Module,
    problem: Problem,
    tr: Split,
    va: Split,
    cfg: TrainConfig,
    features_np: Optional[np.ndarray] = None,
    out_dir: str = ".",
    log: Callable[[str], None] = print,
    dtype: torch.dtype = torch.float64,
    checkpoint_name: str = "best_model.pt",
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Train one model and return (history, best).

    ``best`` carries the epoch, the validation loss and the checkpoint path, so
    that every number the experiment later reports can be traced to a file.
    """
    if features_np is None:
        features_np = build_features(problem)

    A = to_torch_sparse(problem.A, dtype)
    AT = to_torch_sparse(problem.A.T.tocsr(), dtype)
    features = torch.from_numpy(np.asarray(features_np, dtype=np.float64)).to(dtype)

    Etr, Rtr = _tensors(tr, dtype)
    Eva, Rva = _tensors(va, dtype)

    # The loss denominator is exactly one by construction. Prove it here rather
    # than trusting the constructor: an assertion that costs microseconds and
    # removes a whole class of silent scaling error.
    den = np.asarray(tr.E_F, dtype=np.float64)
    den_energy = np.einsum("bi,bi->b", den, (problem.A @ den.T).T)
    worst = float(np.max(np.abs(den_energy - 1.0)))
    if worst > 1e-10:
        raise SystemExit(
            "training targets are not unit A-energy (worst deviation %.3e); the "
            "reported loss would not be the specified ratio" % worst
        )

    n_tr = Etr.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=cfg.lr_patience_checks,
        threshold=cfg.min_delta, threshold_mode="abs")
    g = torch.Generator().manual_seed(cfg.seed)

    history: List[Dict[str, object]] = []
    best, best_epoch, bad = float("inf"), -1, 0
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        idx = torch.randperm(n_tr, generator=g)[: cfg.batch_size]
        corr = model(Rtr[idx], features, None)
        loss = stage1_loss(corr, Etr[idx], A, AT)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if epoch == 1 or epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            v_loss, = validation_means(
                model, Rva, Eva, features, cfg.val_batch,
                lambda corr, errors: stage1_loss(corr, errors, A, AT))
            sched.step(v_loss)

            row: Dict[str, object] = {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_loss": v_loss,
                "lr": float(opt.param_groups[0]["lr"]),
                "seconds": time.time() - t0,
                "omega": float(model.omega.item()) if hasattr(model, "omega") else float("nan"),
            }
            history.append(row)
            log("      step  %6d  train %.6f  val %.6f  lr %.2e%s"
                % (epoch, row["train_loss"], v_loss, row["lr"],
                   ("  omega %.4f" % row["omega"]) if np.isfinite(row["omega"]) else ""))

            if v_loss < best - cfg.min_delta:
                best, best_epoch, bad = v_loss, epoch, 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": asdict(cfg),
                        "epsilon": problem.eps,
                        "epoch": epoch,
                        "val_loss": v_loss,
                    },
                    os.path.join(out_dir, checkpoint_name),
                )
            else:
                bad += 1
                if bad >= cfg.early_stop:
                    log("      early stop at epoch %d (best epoch %d, val %.6f)"
                        % (epoch, best_epoch, best))
                    break

    ckpt = os.path.join(out_dir, checkpoint_name)
    if not os.path.exists(ckpt):
        raise SystemExit("training produced no checkpoint -- the first log point "
                         "never improved on infinity, which should be impossible")

    restore_best(model, ckpt)
    return history, {
        "best_epoch": best_epoch,
        "best_val_loss": best,
        "checkpoint": ckpt,
        "epochs_run": history[-1]["epoch"] if history else 0,
        "seconds": time.time() - t0,
    }


def train_penalised(
    model: torch.nn.Module,
    problem: Problem,
    tr,
    va,
    cfg: TrainConfig,
    Q,
    lam: float,
    features_np: Optional[np.ndarray] = None,
    out_dir: str = ".",
    log: Callable[[str], None] = print,
    dtype: torch.dtype = torch.float64,
    checkpoint_name: str = "best_model.pt",
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Train on L = E||Q e^+||_A^2 + lambda E||e^+||_A^2.

    Same optimiser, schedule, clipping and model selection as the rest of the
    framework; the only difference from ``train_smoother`` is the objective.
    Both terms are logged separately every log point, because the total alone
    cannot tell "the smoother got better" from "the penalty took over".

    Selection follows the validation TOTAL, which is the objective actually
    being minimised. The test set is never consulted.
    """
    from .losses import penalised_loss

    if features_np is None:
        from .model import build_features
        features_np = build_features(problem)
    features = torch.from_numpy(np.asarray(features_np, dtype=np.float64)).to(dtype)

    # The tensors are built once, outside the loop: rebuilding them per epoch
    # would put a numpy->torch copy of the whole training set inside the hot
    # path and dominate the run.
    Etr = torch.from_numpy(np.asarray(tr.E, dtype=np.float64)).to(dtype)
    Eva = torch.from_numpy(np.asarray(va.E, dtype=np.float64)).to(dtype)
    Rtr = torch.from_numpy(np.asarray(tr.R, dtype=np.float64)).to(dtype)
    Rva = torch.from_numpy(np.asarray(va.R, dtype=np.float64)).to(dtype)

    den = np.asarray(tr.E, dtype=np.float64)
    worst = float(np.max(np.abs(np.einsum("bi,bi->b", den,
                                          (problem.A @ den.T).T) - 1.0)))
    if worst > 1e-10:
        raise SystemExit("training targets are not unit A-energy (worst %.3e); "
                         "the two loss terms would not be comparable" % worst)

    n_tr = Etr.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=cfg.lr_patience_checks,
        threshold=cfg.min_delta, threshold_mode="abs")
    g = torch.Generator().manual_seed(cfg.seed)

    history: List[Dict[str, object]] = []
    best, best_epoch, bad = float("inf"), -1, 0
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        idx = torch.randperm(n_tr, generator=g)[: cfg.batch_size]
        corr = model(Rtr[idx], features, None)
        loss, comp, full = penalised_loss(corr, Etr[idx], Q, lam)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if epoch == 1 or epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            v_loss, v_comp, v_full = validation_means(
                model, Rva, Eva, features, cfg.val_batch,
                lambda corr, errors: penalised_loss(corr, errors, Q, lam))
            sched.step(float(v_loss))
            row = {"epoch": epoch, "train_total": float(loss.item()),
                   "train_complement": float(comp.item()),
                   "train_full": float(full.item()),
                   "val_total": float(v_loss),
                   "val_complement": float(v_comp),
                   "val_full": float(v_full),
                   "lr": float(opt.param_groups[0]["lr"]),
                   "seconds": time.time() - t0}
            history.append(row)
            log("      step  %6d  train %.6f (Q %.6f + %.3g*full %.6f)  "
                "val %.6f (Q %.6f)" % (epoch, row["train_total"],
                                       row["train_complement"], lam,
                                       row["train_full"], row["val_total"],
                                       row["val_complement"]))
            if float(v_loss) < best - cfg.min_delta:
                best, best_epoch, bad = float(v_loss), epoch, 0
                torch.save({"state_dict": model.state_dict(),
                            "config": asdict(cfg), "epsilon": problem.eps,
                            "lambda": lam, "epoch": epoch,
                            "val_total": float(v_loss)},
                           os.path.join(out_dir, checkpoint_name))
            else:
                bad += 1
                if bad >= cfg.early_stop:
                    log("      early stop at epoch %d (best %d, val %.6f)"
                        % (epoch, best_epoch, best))
                    break

    restore_best(model, os.path.join(out_dir, checkpoint_name))
    return history, {"best_epoch": best_epoch, "best_val_loss": best,
                     "optimizer_steps": epoch, "validation_samples": len(Eva),
                     "lambda": lam, "epochs_run": history[-1]["epoch"],
                     "seconds": time.time() - t0,
                     "checkpoint": os.path.join(out_dir, checkpoint_name)}


def write_history(history: List[Dict[str, object]], path: str) -> None:
    if not history:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        for row in history:
            w.writerow(row)
