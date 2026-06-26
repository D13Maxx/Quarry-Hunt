import torch
from quarry.config import Config
from quarry.env import QuarryEnv
from quarry.actor import Actor
from quarry.predictor import PreyPredictor
from quarry.rollout import collect_episode
from quarry.critic import global_state_dim


def _setup():
    cfg = Config()
    env = QuarryEnv(cfg)
    actor = Actor(cfg)
    predictor = PreyPredictor(cfg)
    return cfg, env, actor, predictor


def test_episode_terminates_and_buffer_shapes():
    cfg, env, actor, predictor = _setup()
    result = collect_episode(env, actor, predictor, cfg)

    assert result["winner"] in ("hunters", "prey")
    steps = result["length"]
    assert steps > 0
    print(f"  episode: {result['winner']} win at step {steps}")

    n = cfg.num_hunters
    assert len(result["ppo"]) == steps * n, f"ppo: {len(result['ppo'])} != {steps * n}"
    assert len(result["predictor"]) == steps * n

    t = result["ppo"][0]
    assert t["obs"].shape == (6, cfg.hunter_vision, cfg.hunter_vision)
    assert t["ghost"].shape == (cfg.window_K, cfg.window_K)
    assert 0 <= t["action"] <= 8
    assert isinstance(t["logprob"], float)
    assert isinstance(t["reward"], float) and torch.isfinite(torch.tensor(t["reward"]))
    assert isinstance(t["done"], bool)
    assert t["global_state"].shape == (global_state_dim(cfg),)

    p = result["predictor"][0]
    assert p["obs_seq"].shape == (cfg.history_len, 6, cfg.hunter_vision, cfg.hunter_vision)
    assert 0 <= p["target_r"] < cfg.window_K
    assert 0 <= p["target_c"] < cfg.window_K


def test_ghost_shape_every_step():
    cfg, env, actor, predictor = _setup()
    result = collect_episode(env, actor, predictor, cfg)
    for t in result["ppo"]:
        assert t["ghost"].shape == (cfg.window_K, cfg.window_K)
        assert torch.isfinite(t["ghost"]).all()


def test_reward_finite():
    cfg, env, actor, predictor = _setup()
    result = collect_episode(env, actor, predictor, cfg)
    for t in result["ppo"]:
        assert torch.isfinite(torch.tensor(t["reward"]))


def test_no_state_bleed_across_episodes():
    cfg, env, actor, predictor = _setup()
    r1 = collect_episode(env, actor, predictor, cfg)
    r2 = collect_episode(env, actor, predictor, cfg)

    # both should terminate independently
    assert r1["winner"] in ("hunters", "prey")
    assert r2["winner"] in ("hunters", "prey")
    print(f"  ep1: {r1['winner']} at step {r1['length']}")
    print(f"  ep2: {r2['winner']} at step {r2['length']}")

    # first PPO obs of ep2 should NOT carry ep1's final obs
    first_obs_ep2 = r2["ppo"][0]["obs"]
    last_obs_ep1 = r1["ppo"][-1]["obs"]
    assert not torch.equal(first_obs_ep2, last_obs_ep1) or r1["length"] == r2["length"] == 1, \
        "obs from ep1 leaked into ep2"

    # predictor history should be fresh (padded zeros at start)
    first_pred = r2["predictor"][0]["obs_seq"]
    # first frame in history should be zero-padded (history_len > 1)
    if cfg.history_len > 1:
        assert (first_pred[0] == 0).all(), "predictor history not reset between episodes"
