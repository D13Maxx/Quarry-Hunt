import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quarry.agents import prey_act
from quarry.config import Config
from quarry.env import QuarryEnv
from quarry.metrics import compute_all
from quarry.relative_frame import to_relative
from quarry.world import RIM


@torch.no_grad()
def evaluate(
    actor, predictor, cfg: Config,
    n_episodes: int = 100,
    device: str = "cpu",
    save_dir: str | None = None,
) -> dict:
    env = QuarryEnv(cfg)
    dev = torch.device(device)
    K = cfg.window_K
    n = cfg.num_hunters

    captures, rim_drives, timeouts = 0, 0, 0
    capture_steps = []
    all_offsets = []

    all_logits, all_tr, all_tc = [], [], []

    actor.eval()
    predictor.eval()

    for _ in range(n_episodes):
        obs_dict, _ = env.reset()
        hiddens = [predictor.init_hidden(dev) for _ in range(n)]
        done = False

        while not done:
            min_dist = min(
                max(abs(hp[0] - env.prey_pos[0]), abs(hp[1] - env.prey_pos[1]))
                for hp in env.hunter_pos
            )
            all_offsets.append(min_dist)

            actions = {}
            for i in range(n):
                obs_t = torch.from_numpy(obs_dict[f"hunter_{i}"]).float().to(dev)
                logits, hiddens[i] = predictor.step(obs_t, hiddens[i])
                ghost = F.softmax(logits.flatten(), dim=-1).reshape(K, K).detach()

                r, c, _ = to_relative(env.hunter_pos[i], env.prey_pos, K)
                all_logits.append(logits.flatten().cpu())
                all_tr.append(r)
                all_tc.append(c)

                action, _, _ = actor.act(obs_t.unsqueeze(0), ghost.unsqueeze(0))
                actions[f"hunter_{i}"] = int(action.item())

            actions["prey"] = prey_act(obs_dict["prey"])
            obs_dict, _, terms, _, _ = env.step(actions)
            done = terms.get("prey", False)

        if env.winner == "hunters":
            if env.grid[env.prey_pos] == RIM:
                rim_drives += 1
            else:
                captures += 1
            capture_steps.append(env.step_count)
        else:
            timeouts += 1

    logits_t = torch.stack(all_logits)
    tr_t = torch.tensor(all_tr, dtype=torch.long)
    tc_t = torch.tensor(all_tc, dtype=torch.long)
    pred_metrics = compute_all(logits_t, tr_t, tc_t, K)

    offsets = np.array(all_offsets)
    result = {
        "n_episodes": n_episodes,
        "captures": captures,
        "rim_drives": rim_drives,
        "timeouts": timeouts,
        "capture_rate": (captures + rim_drives) / n_episodes,
        "mean_ttc": np.mean(capture_steps) if capture_steps else float("inf"),
        "pred_top1": pred_metrics["top1_hit"],
        "pred_argmax_cheby": pred_metrics["argmax_cheby"],
        "pred_expected_cheby": pred_metrics["expected_cheby"],
        "pred_true_cell_prob": pred_metrics["true_cell_prob"],
        "offset_mean": float(offsets.mean()),
        "offset_median": float(np.median(offsets)),
        "offsets": offsets,
    }

    if save_dir:
        _save_histogram(offsets, save_dir, cfg)

    return result


def _save_histogram(offsets, save_dir, cfg):
    out = pathlib.Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    max_off = int(offsets.max())
    bins = np.arange(0, max_off + 2) - 0.5
    ax.hist(offsets, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Min Chebyshev distance (hunter-prey)")
    ax.set_ylabel("Count")
    ax.set_title(f"Offset distribution (K={cfg.window_K})")
    ax.axvline(cfg.window_K // 2, color="red", linestyle="--", label=f"K//2 = {cfg.window_K // 2}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "offset_histogram.png", dpi=120)
    plt.close(fig)


def print_report(result: dict):
    print(f"\n{'='*50}")
    print(f"  Evaluation: {result['n_episodes']} episodes")
    print(f"{'='*50}")
    print(f"  Capture rate:  {result['capture_rate']:.3f}")
    print(f"    Caught:      {result['captures']}")
    print(f"    Rim-driven:  {result['rim_drives']}")
    print(f"    Timeouts:    {result['timeouts']}")
    print(f"  Mean TTC:      {result['mean_ttc']:.1f}")
    print(f"  Predictor:")
    print(f"    Top-1 hit:   {result['pred_top1']:.4f}")
    print(f"    Argmax Cheb: {result['pred_argmax_cheby']:.2f}")
    print(f"    E[Cheb]:     {result['pred_expected_cheby']:.2f}")
    print(f"    True cell p: {result['pred_true_cell_prob']:.4f}")
    print(f"  Offsets:")
    print(f"    Mean:        {result['offset_mean']:.2f}")
    print(f"    Median:      {result['offset_median']:.1f}")
    print(f"{'='*50}\n")
