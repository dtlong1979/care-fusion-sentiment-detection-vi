"""Shared training machinery (Protocol Part D2): seeding, discriminative-LR
optimizer, warmup+linear schedule, AMP, early stopping on macro-F1.

Used by both the CARE-Fusion trainer and the baselines so every model sees the
same optimizer/seed protocol (Part E: same dataset, split, seed).
"""
from __future__ import annotations

import copy
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


def apply_profile(cfg: dict, profile: Optional[dict]) -> Tuple[dict, Optional[int]]:
    """Merge a run profile (smoke/pilot) into a copy of cfg. Returns (cfg, subset).

    Profile keys override train/preprocess/resources fields so the rest of the
    code reads a single, consistent config regardless of profile.
    """
    cfg = copy.deepcopy(cfg)
    if not profile:
        return cfg, None
    for k in ["seeds", "max_epochs", "patience", "batch_size", "fp16", "cf_detach",
              "lambda1", "lambda2", "lr_phobert", "lr_head"]:
        if k in profile:
            cfg["train"][k] = profile[k]
    if "max_length" in profile:
        cfg["preprocess"]["max_length"] = profile["max_length"]
    if "oof_folds" in profile:
        cfg["resources"]["oof_folds"] = profile["oof_folds"]
    return cfg, profile.get("subset")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, lr_phobert: float, lr_head: float, weight_decay: float):
    """Discriminative LR: PhoBERT encoder slow, new heads fast."""
    bert, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bert if ".bert." in n or n.startswith("bert.") else head).append(p)
    return torch.optim.AdamW(
        [{"params": bert, "lr": lr_phobert},
         {"params": head, "lr": lr_head}],
        weight_decay=weight_decay,
    )


def move(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader: DataLoader, device, logits_key="logits") -> Dict[str, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = move(batch, device)
        out = model(batch)
        logits = out[logits_key] if isinstance(out, dict) else out
        all_logits.append(logits.float().cpu())
        all_labels.append(batch["labels"].cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return {"logits": logits, "probs": _softmax(logits), "preds": logits.argmax(1), "labels": labels}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def macro_f1(labels, preds) -> float:
    return float(f1_score(labels, preds, average="macro", zero_division=0))


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    cfg: dict,
    device,
    loss_fn: Callable,
    max_epochs: int,
    patience: int,
    fp16: bool = False,
    logits_key: str = "logits",
    log: Callable[[str], None] = print,
) -> Dict:
    """Generic train loop. `loss_fn(model, batch) -> dict with 'total'`.

    Returns best state_dict (by val macro-F1) and the val history.
    """
    tcfg = cfg["train"]
    model.to(device)
    opt = build_optimizer(model, tcfg["lr_phobert"], tcfg["lr_head"], tcfg["weight_decay"])
    steps = max_epochs * max(1, len(train_loader))
    sched = get_linear_schedule_with_warmup(opt, int(tcfg["warmup_ratio"] * steps), steps)
    use_amp = fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, best_state, no_improve, history = -1.0, None, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = move(batch, device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                losses = loss_fn(model, batch)
                loss = losses["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()

        if val_loader is None:
            log(f"  epoch {epoch}: train_loss={running/len(train_loader):.4f}")
            continue
        ev = predict(model, val_loader, device, logits_key)
        f1 = macro_f1(ev["labels"], ev["preds"])
        history.append(f1)
        log(f"  epoch {epoch}: train_loss={running/len(train_loader):.4f} val_macroF1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log(f"  early stop at epoch {epoch} (best val macroF1={best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_f1": best_f1, "history": history, "state": best_state}
