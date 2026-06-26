import torch
import torch.nn as nn

from quarry.config import Config


def build_global_state(
    hunter_pos: list[tuple[int, int]],
    prey_pos: tuple[int, int],
    step: int,
    cfg: Config,
) -> torch.Tensor:
    """Fixed-size global-state vector from true world state. Returns (state_dim,)."""
    gs = cfg.grid_size
    parts = []
    for r, c in hunter_pos:
        parts.extend([r / gs, c / gs])
    parts.extend([prey_pos[0] / gs, prey_pos[1] / gs])
    parts.append(step / cfg.max_steps)
    return torch.tensor(parts, dtype=torch.float32)


def global_state_dim(cfg: Config) -> int:
    return 2 * cfg.num_hunters + 2 + 1


class CentralizedCritic(nn.Module):
    """CTDE critic — sees true global state (all positions). Training only."""

    def __init__(self, cfg: Config):
        super().__init__()
        sd = global_state_dim(cfg)
        self.net = nn.Sequential(
            nn.Linear(sd, cfg.critic_hidden),
            nn.ReLU(),
            nn.Linear(cfg.critic_hidden, cfg.critic_hidden),
            nn.ReLU(),
            nn.Linear(cfg.critic_hidden, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """global_state: (B, state_dim) -> (B, 1) value estimate."""
        return self.net(global_state)
