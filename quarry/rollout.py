from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from quarry.agents import prey_act
from quarry.config import Config
from quarry.critic import build_global_state
from quarry.relative_frame import to_relative
from quarry.world import RIM


def _min_chebyshev(hunter_pos: list[tuple[int, int]], prey_pos: tuple[int, int]) -> int:
    pr, pc = prey_pos
    return min(max(abs(hr - pr), abs(hc - pc)) for hr, hc in hunter_pos)


@dataclass
class RewardState:
    prev_min_dist: int = -1

    def reset(self, hunter_pos: list[tuple[int, int]], prey_pos: tuple[int, int]):
        self.prev_min_dist = _min_chebyshev(hunter_pos, prey_pos)


def compute_reward(
    hunter_pos: list[tuple[int, int]],
    prey_pos: tuple[int, int],
    grid: np.ndarray,
    rs: RewardState,
    cfg: Config,
) -> tuple[float, dict[str, float]]:
    """Compute the shared team reward for one step. Returns (total, breakdown)."""
    cur_dist = _min_chebyshev(hunter_pos, prey_pos)

    captured = cur_dist <= 1
    on_rim = grid[prey_pos] == RIM
    closing = (rs.prev_min_dist - cur_dist) if rs.prev_min_dist >= 0 else 0

    r_capture = cfg.capture_reward if captured else 0.0
    r_rim = cfg.rim_drive_reward if on_rim else 0.0
    r_step = cfg.step_penalty
    r_closing = cfg.closing_reward_scale * closing

    rs.prev_min_dist = cur_dist

    total = r_capture + r_rim + r_step + r_closing
    breakdown = {
        "capture": r_capture,
        "rim_drive": r_rim,
        "step_penalty": r_step,
        "closing": r_closing,
    }
    return total, breakdown


@torch.no_grad()
def collect_episode(env, actor, predictor, cfg: Config, device=torch.device("cpu")):
    """Play one episode: RL actor drives hunters, scripted prey flees.

    Returns dict with 'ppo' buffer (steps*num_hunters transitions),
    'predictor' buffer (supervised examples), 'winner', 'length'.
    """
    obs_dict, _ = env.reset()
    n = cfg.num_hunters
    K = cfg.window_K
    H = cfg.history_len

    hiddens = [predictor.init_hidden(device) for _ in range(n)]
    obs_histories: list[list[torch.Tensor]] = [[] for _ in range(n)]

    ppo_buf: list[dict] = []
    pred_buf: list[dict] = []

    rs = RewardState()
    rs.reset(env.hunter_pos, env.prey_pos)

    done = False
    while not done:
        step_obs, step_ghosts = [], []
        step_actions, step_logprobs = [], []

        for i in range(n):
            obs_t = torch.from_numpy(obs_dict[f"hunter_{i}"]).float().to(device)

            obs_histories[i].append(obs_t)
            if len(obs_histories[i]) > H:
                obs_histories[i] = obs_histories[i][-H:]

            logits, hiddens[i] = predictor.step(obs_t, hiddens[i])
            ghost = F.softmax(logits.flatten(), dim=-1).reshape(K, K).detach()

            action, logprob, _ = actor.act(obs_t.unsqueeze(0), ghost.unsqueeze(0))

            step_obs.append(obs_t)
            step_ghosts.append(ghost)
            step_actions.append(int(action.item()))
            step_logprobs.append(float(logprob.item()))

        actions = {f"hunter_{i}": step_actions[i] for i in range(n)}
        actions["prey"] = prey_act(obs_dict["prey"])

        global_state = build_global_state(
            env.hunter_pos, env.prey_pos, env.step_count, cfg
        )

        obs_dict, _, terms, _, _ = env.step(actions)
        done = terms.get("prey", False)

        reward, _ = compute_reward(
            env.hunter_pos, env.prey_pos, env.grid, rs, cfg
        )

        for i in range(n):
            ppo_buf.append({
                "obs": step_obs[i],
                "ghost": step_ghosts[i],
                "action": step_actions[i],
                "logprob": step_logprobs[i],
                "reward": reward,
                "done": done,
                "global_state": global_state,
            })

            # predictor example: pad history if shorter than H
            seq = list(obs_histories[i])
            if len(seq) < H:
                seq = [torch.zeros_like(seq[0])] * (H - len(seq)) + seq
            obs_seq = torch.stack(seq)

            r, c, _ = to_relative(env.hunter_pos[i], env.prey_pos, K)
            pred_buf.append({
                "obs_seq": obs_seq,
                "target_r": r,
                "target_c": c,
            })

    return {
        "ppo": ppo_buf,
        "predictor": pred_buf,
        "winner": env.winner,
        "length": env.step_count,
    }
