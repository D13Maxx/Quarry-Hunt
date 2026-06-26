import torch
import torch.nn as nn
from torch.distributions import Categorical

from quarry.config import Config


class Actor(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        ch1, ch2 = cfg.encoder_channels

        self.obs_conv = nn.Sequential(
            nn.Conv2d(6, ch1, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch1, ch2, 3, padding=1),
            nn.ReLU(),
        )
        obs_flat = ch2 * cfg.hunter_vision * cfg.hunter_vision

        ghost_flat = cfg.window_K * cfg.window_K
        self.ghost_mlp = nn.Sequential(
            nn.Linear(ghost_flat, cfg.encoder_out_dim),
            nn.ReLU(),
        )

        self.policy = nn.Sequential(
            nn.Linear(obs_flat + cfg.encoder_out_dim, cfg.actor_hidden),
            nn.ReLU(),
            nn.Linear(cfg.actor_hidden, 9),
        )

    def forward(self, obs: torch.Tensor, ghost: torch.Tensor) -> torch.Tensor:
        """obs: (B, 6, H, W), ghost: (B, K, K) detached -> (B, 9) action logits."""
        obs_feat = self.obs_conv(obs).flatten(1)
        ghost_feat = self.ghost_mlp(ghost.flatten(1))
        return self.policy(torch.cat([obs_feat, ghost_feat], dim=1))

    def act(self, obs: torch.Tensor, ghost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions. Returns (actions, log_probs, entropy), all (B,)."""
        dist = Categorical(logits=self.forward(obs, ghost))
        actions = dist.sample()
        return actions, dist.log_prob(actions), dist.entropy()
