import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from quarry.config import Config


def compute_gae(
    rewards: torch.Tensor, values: torch.Tensor,
    dones: torch.Tensor, gamma: float, gae_lambda: float,
) -> torch.Tensor:
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        next_value = values[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        gae = delta + gamma * gae_lambda * next_nonterminal * gae
        advantages[t] = gae
    return advantages


def ppo_update(
    actor, critic, ppo_optimizer,
    episodes: list[dict], cfg: Config,
    device: torch.device = torch.device("cpu"),
) -> list[dict]:
    """MAPPO update over one or more collected episodes. Returns per-epoch log dicts."""
    n = cfg.num_hunters

    all_obs, all_ghosts, all_actions, all_old_lp = [], [], [], []
    all_advantages, all_returns, all_gs = [], [], []

    for ep in episodes:
        buf = ep["ppo"]
        steps = ep["length"]

        gs = torch.stack([buf[t * n]["global_state"] for t in range(steps)]).to(device)
        rewards = torch.tensor([buf[t * n]["reward"] for t in range(steps)],
                               dtype=torch.float32, device=device)
        dones = torch.tensor([buf[t * n]["done"] for t in range(steps)],
                             dtype=torch.float32, device=device)

        with torch.no_grad():
            values = critic(gs).squeeze(-1)

        adv = compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)
        ret = adv + values

        all_advantages.append(adv.repeat_interleave(n))
        all_returns.append(ret.repeat_interleave(n))
        all_gs.append(gs.repeat_interleave(n, dim=0))

        for t in buf:
            all_obs.append(t["obs"])
            all_ghosts.append(t["ghost"])
            all_actions.append(t["action"])
            all_old_lp.append(t["logprob"])

    obs = torch.stack(all_obs).to(device)
    ghosts = torch.stack(all_ghosts).to(device)
    actions = torch.tensor(all_actions, dtype=torch.long, device=device)
    old_lp = torch.tensor(all_old_lp, dtype=torch.float32, device=device)
    advantages = torch.cat(all_advantages)
    returns = torch.cat(all_returns).detach()
    gs_all = torch.cat(all_gs)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    logs = []
    for _ in range(cfg.ppo_epochs):
        logits = actor(obs, ghosts)
        dist = Categorical(logits=logits)
        new_lp = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = (new_lp - old_lp).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        new_values = critic(gs_all).squeeze(-1)
        value_loss = F.mse_loss(new_values, returns)

        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

        ppo_optimizer.zero_grad()
        loss.backward()
        ppo_optimizer.step()

        with torch.no_grad():
            approx_kl = (old_lp - new_lp).mean().item()
            clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()

        logs.append({
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_frac,
        })

    return logs
