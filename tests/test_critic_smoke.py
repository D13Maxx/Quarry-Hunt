import torch
from quarry.config import Config
from quarry.critic import CentralizedCritic, build_global_state
from quarry.actor import Actor


def test_forward_shape_and_no_nan():
    cfg = Config()
    critic = CentralizedCritic(cfg)
    hunter_pos = [(5, 5), (10, 10), (20, 20)]
    prey_pos = (12, 12)
    gs = build_global_state(hunter_pos, prey_pos, step=50, cfg=cfg)
    batch = gs.unsqueeze(0).expand(4, -1)
    out = critic(batch)
    assert out.shape == (4, 1), f"got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in critic output"


def test_global_state_vector():
    cfg = Config()
    hunter_pos = [(0, 0), (12, 12), (24, 24)]
    prey_pos = (6, 18)
    gs = build_global_state(hunter_pos, prey_pos, step=100, cfg=cfg)
    assert gs.shape == (9,), f"got {gs.shape}"
    assert gs[-1].item() == 100 / cfg.max_steps


def test_no_shared_params_with_actor():
    cfg = Config()
    actor = Actor(cfg)
    critic = CentralizedCritic(cfg)
    actor_ids = {id(p) for p in actor.parameters()}
    critic_ids = {id(p) for p in critic.parameters()}
    assert actor_ids.isdisjoint(critic_ids), "actor and critic share parameters"
