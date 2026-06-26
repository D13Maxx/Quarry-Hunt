import torch
import numpy as np


def top1_hit(logits: torch.Tensor, target_r: torch.Tensor, target_c: torch.Tensor, K: int) -> float:
    pred = logits.argmax(dim=1)
    pred_r, pred_c = pred // K, pred % K
    hits = ((pred_r == target_r) & (pred_c == target_c)).float()
    return hits.mean().item()


def argmax_chebyshev(logits: torch.Tensor, target_r: torch.Tensor, target_c: torch.Tensor, K: int) -> float:
    pred = logits.argmax(dim=1)
    pred_r, pred_c = pred // K, pred % K
    dist = torch.max(torch.abs(pred_r - target_r), torch.abs(pred_c - target_c)).float()
    return dist.mean().item()


def expected_chebyshev(logits: torch.Tensor, target_r: torch.Tensor, target_c: torch.Tensor, K: int) -> float:
    probs = torch.softmax(logits, dim=1)
    B = probs.shape[0]
    rows = torch.arange(K, device=logits.device)
    cols = torch.arange(K, device=logits.device)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    rr = rr.flatten().unsqueeze(0).expand(B, -1)
    cc = cc.flatten().unsqueeze(0).expand(B, -1)
    tr = target_r.unsqueeze(1).expand_as(rr)
    tc = target_c.unsqueeze(1).expand_as(cc)
    dist = torch.max(torch.abs(rr - tr), torch.abs(cc - tc)).float()
    return (probs * dist).sum(dim=1).mean().item()


def true_cell_prob(logits: torch.Tensor, target_r: torch.Tensor, target_c: torch.Tensor, K: int) -> float:
    probs = torch.softmax(logits, dim=1)
    idx = target_r * K + target_c
    mass = probs.gather(1, idx.unsqueeze(1)).squeeze(1)
    return mass.mean().item()


def compute_all(logits: torch.Tensor, target_r: torch.Tensor, target_c: torch.Tensor, K: int) -> dict[str, float]:
    return {
        "top1_hit": top1_hit(logits, target_r, target_c, K),
        "argmax_cheby": argmax_chebyshev(logits, target_r, target_c, K),
        "expected_cheby": expected_chebyshev(logits, target_r, target_c, K),
        "true_cell_prob": true_cell_prob(logits, target_r, target_c, K),
    }


def uniform_baseline(K: int) -> dict[str, float]:
    """Uniform 1/(K*K) over the window — pure chance."""
    return {"name": "uniform", "top1_hit": 1.0 / (K * K), "expected_cheby": _uniform_expected_cheby(K)}


def stay_baseline(target_r: np.ndarray, target_c: np.ndarray, K: int) -> dict[str, float]:
    """Always predict center cell."""
    half = K // 2
    hits = ((target_r == half) & (target_c == half)).mean()
    dist = np.maximum(np.abs(target_r - half), np.abs(target_c - half)).mean()
    return {"name": "stay", "top1_hit": float(hits), "argmax_cheby": float(dist)}


def last_seen_baseline(all_obs: list, all_target_r: np.ndarray, all_target_c: np.ndarray, K: int) -> dict[str, float]:
    """Predict the prey's last-seen relative position. Falls back to center if never seen."""
    from quarry.world import CH_PREY
    half = K // 2
    hits = 0
    total_dist = 0.0
    n = 0
    for obs_window, tr, tc in zip(all_obs, all_target_r, all_target_c):
        # obs_window: (T, C, H, W) — scan backward for last prey sighting
        pred_r, pred_c = half, half
        for t in range(obs_window.shape[0] - 1, -1, -1):
            prey_locs = np.argwhere(obs_window[t, CH_PREY] > 0.5)
            if len(prey_locs) > 0:
                # prey visible at this frame — use its position relative to center
                pr, pc = prey_locs[0]
                vis_half = obs_window.shape[2] // 2
                pred_r = np.clip(pr - vis_half + half, 0, K - 1)
                pred_c = np.clip(pc - vis_half + half, 0, K - 1)
                break
        hits += (pred_r == tr and pred_c == tc)
        total_dist += max(abs(pred_r - tr), abs(pred_c - tc))
        n += 1
    return {"name": "last_seen", "top1_hit": hits / n, "argmax_cheby": total_dist / n}


def _uniform_expected_cheby(K: int) -> float:
    half = K // 2
    total = 0.0
    for r in range(K):
        for c in range(K):
            for tr in range(K):
                for tc in range(K):
                    total += max(abs(r - tr), abs(c - tc))
    return total / (K * K * K * K)
