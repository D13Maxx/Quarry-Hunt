import torch
from quarry.config import Config
from quarry.actor import Actor


def test_forward_shape_and_no_nan():
    cfg = Config()
    actor = Actor(cfg)
    B = 8
    obs = torch.randn(B, 6, cfg.hunter_vision, cfg.hunter_vision)
    ghost = torch.randn(B, cfg.window_K, cfg.window_K)
    logits = actor(obs, ghost)
    assert logits.shape == (B, 9), f"got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in actor output"


def test_act_samples_valid():
    cfg = Config()
    actor = Actor(cfg)
    B = 16
    obs = torch.randn(B, 6, cfg.hunter_vision, cfg.hunter_vision)
    ghost = torch.randn(B, cfg.window_K, cfg.window_K)
    actions, logprobs, entropy = actor.act(obs, ghost)

    assert actions.shape == (B,)
    assert logprobs.shape == (B,)
    assert entropy.shape == (B,)
    assert (actions >= 0).all() and (actions <= 8).all(), f"actions out of range: {actions}"
    assert torch.isfinite(logprobs).all(), "non-finite logprobs"
    assert torch.isfinite(entropy).all(), "non-finite entropy"


def test_ghost_detached_no_grad():
    """Actor must not backprop into ghost (predictor)."""
    cfg = Config()
    actor = Actor(cfg)
    obs = torch.randn(1, 6, cfg.hunter_vision, cfg.hunter_vision)
    ghost = torch.randn(1, cfg.window_K, cfg.window_K, requires_grad=True)
    logits = actor(obs, ghost.detach())
    logits.sum().backward()
    assert ghost.grad is None, "gradient leaked into ghost tensor"
