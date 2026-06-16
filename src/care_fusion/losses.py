"""Loss functions (Protocol Part D1).

L = L_cls + lambda1 * L_route + lambda2 * L_cf

- L_cls  : class-balanced focal loss (Cui et al. 2019) for heavy imbalance.
- L_route: cross-entropy of router logits vs weak regime labels (ignore -100).
- L_cf   : counterfactual consistency — ||KL(y_hat || y_hat') - s||^2, where the
           target shift s is small for redundancy and large for complementarity
           / conflict, read off the router weights.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .data import IGNORE_INDEX

# expected counterfactual shift target per regime [redundancy, complementarity, conflict]
_REGIME_SHIFT = torch.tensor([0.0, 0.5, 0.8])


def class_balanced_weights(class_counts: torch.Tensor, beta: float) -> torch.Tensor:
    eff_num = 1.0 - torch.pow(beta, class_counts.clamp_min(1))
    w = (1.0 - beta) / eff_num
    return w / w.sum() * len(class_counts)          # normalize, mean ~ 1


def class_balanced_focal_loss(logits, targets, class_counts, beta=0.9999, gamma=2.0):
    weights = class_balanced_weights(class_counts.to(logits.device), beta)
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    pt = p.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(1e-8, 1.0)
    logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    w = weights[targets]
    loss = -w * (1 - pt).pow(gamma) * logpt
    return loss.mean()


def routing_loss(r_logits, regime_labels):
    """CE over markers with weak labels; -100 entries ignored. 0 if no supervision
    or if routing is ablated (r_logits is None)."""
    if r_logits is None:
        return regime_labels.new_zeros((), dtype=torch.float)
    B, M, K = r_logits.shape
    flat_logits = r_logits.reshape(B * M, K)
    flat_labels = regime_labels.reshape(B * M)
    if (flat_labels != IGNORE_INDEX).sum() == 0:
        return r_logits.new_zeros(())
    return F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_INDEX)


def _kl(p_logits, q_logits):
    """KL(softmax(p) || softmax(q)) per row -> [B]."""
    p = p_logits.softmax(-1)
    logp = p_logits.log_softmax(-1)
    logq = q_logits.log_softmax(-1)
    return (p * (logp - logq)).sum(-1)


def counterfactual_loss(out, out_cf):
    """Penalize when the prediction shift under marker removal deviates from the
    regime-implied expected shift s."""
    shift = _kl(out["logits"], out_cf["logits"])             # [B]
    r = out["r"]                                             # [B, M, 3]
    mask = out["marker_mask"]                                # [B, M]
    s_vec = _REGIME_SHIFT.to(r.device)
    s_marker = (r * s_vec).sum(-1)                           # [B, M]
    denom = mask.sum(1).clamp_min(1e-8)
    s = (s_marker * mask).sum(1) / denom                     # [B] expected shift
    has_marker = (mask.sum(1) > 0).float()
    return (has_marker * (shift - s).pow(2)).sum() / has_marker.sum().clamp_min(1.0)


def compute_loss(out, batch, class_counts, cfg, out_cf: Optional[dict] = None) -> Dict[str, torch.Tensor]:
    tcfg = cfg["train"]
    l_cls = class_balanced_focal_loss(
        out["logits"], batch["labels"], class_counts,
        beta=tcfg["cb_beta"], gamma=tcfg["focal_gamma"])
    l_route = routing_loss(out["r_logits"], batch["regime_labels"])
    l_cf = counterfactual_loss(out, out_cf) if out_cf is not None else out["logits"].new_zeros(())
    # Auxiliary supervision on the text head so p_text is a genuine text-only
    # prediction -> delta_j = JSD(p_text || q_j) carries the intended meaning and
    # the router's weak-label supervision stays consistent at inference.
    aux_w = tcfg.get("aux_text_weight", 0.0)
    if aux_w > 0 and "text_logits" in out:
        l_aux = class_balanced_focal_loss(out["text_logits"], batch["labels"], class_counts,
                                          beta=tcfg["cb_beta"], gamma=tcfg["focal_gamma"])
    else:
        l_aux = out["logits"].new_zeros(())
    total = l_cls + tcfg["lambda1"] * l_route + tcfg["lambda2"] * l_cf + aux_w * l_aux
    return {"total": total, "cls": l_cls, "route": l_route, "cf": l_cf, "aux": l_aux}
