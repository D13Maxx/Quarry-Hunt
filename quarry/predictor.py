import torch
import torch.nn as nn

from quarry.config import Config


class PreyPredictor(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.K = cfg.window_K
        self.gru_hidden = cfg.gru_hidden
        ch1, ch2 = cfg.encoder_channels

        self.conv = nn.Sequential(
            nn.Conv2d(6, ch1, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch1, ch2, 3, padding=1),
            nn.ReLU(),
        )
        flat_dim = ch2 * cfg.hunter_vision * cfg.hunter_vision
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, cfg.encoder_out_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(cfg.encoder_out_dim, cfg.gru_hidden, batch_first=True)
        self.head = nn.Linear(cfg.gru_hidden, self.K * self.K)

    def _encode(self, obs):
        """obs: (..., C, H, W) -> (..., encoder_out_dim)"""
        shape = obs.shape[:-3]
        x = obs.reshape(-1, *obs.shape[-3:])
        x = self.conv(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x.reshape(*shape, -1)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) -> (B, K, K) logits from the last GRU timestep."""
        features = self._encode(obs_seq)
        _, h_n = self.gru(features)
        logits = self.head(h_n.squeeze(0))
        return logits.reshape(-1, self.K, self.K)

    def step(self, obs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step inference. obs: (C, H, W) -> ((K, K) logits, new_hidden)."""
        feat = self._encode(obs.unsqueeze(0).unsqueeze(0))
        out, h_n = self.gru(feat, hidden)
        logits = self.head(out.squeeze(0).squeeze(0))
        return logits.reshape(self.K, self.K), h_n

    def init_hidden(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.zeros(1, 1, self.gru_hidden, device=device)
