import torch
from quarry.config import Config
from quarry.env import QuarryEnv
from quarry.actor import Actor
from quarry.critic import CentralizedCritic
from quarry.predictor import PreyPredictor
from quarry.rollout import collect_episode
from quarry.ppo import ppo_update


def test_ppo_update():
    cfg = Config()
    actor = Actor(cfg)
    critic = CentralizedCritic(cfg)
    predictor = PreyPredictor(cfg)
    env = QuarryEnv(cfg)

    ppo_opt = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=cfg.ppo_lr,
    )

    # snapshot predictor params before PPO update
    pred_snap = {k: v.clone() for k, v in predictor.state_dict().items()}

    # collect a small rollout
    ep = collect_episode(env, actor, predictor, cfg)

    # run several PPO updates on the same rollout
    all_logs = []
    for _ in range(3):
        logs = ppo_update(actor, critic, ppo_opt, [ep], cfg)
        all_logs.extend(logs)

    # assert all losses finite
    for i, log in enumerate(all_logs):
        for k, v in log.items():
            assert torch.isfinite(torch.tensor(v)), f"epoch {i} {k} not finite: {v}"

    # value loss should decrease across epochs
    vl = [l["value_loss"] for l in all_logs]
    assert vl[-1] < vl[0], f"value_loss didn't decrease: {vl[0]:.4f} -> {vl[-1]:.4f}"

    # approx KL should be small (< 0.1 is reasonable for early updates)
    for log in all_logs[:4]:  # check first update's epochs
        assert log["approx_kl"] < 0.5, f"approx_kl too large: {log['approx_kl']}"

    # clip fraction should be reasonable
    assert all_logs[0]["clip_fraction"] < 0.3, f"clip_fraction too high: {all_logs[0]['clip_fraction']}"

    # predictor params UNCHANGED
    for k, v in predictor.state_dict().items():
        assert torch.equal(v, pred_snap[k]), f"predictor param {k} changed during PPO update"

    print(f"  epochs logged: {len(all_logs)}")
    print(f"  value_loss: {vl[0]:.4f} -> {vl[-1]:.4f}")
    print(f"  policy_loss: {all_logs[0]['policy_loss']:.4f} -> {all_logs[-1]['policy_loss']:.4f}")
    print(f"  entropy: {all_logs[0]['entropy']:.4f}")
    print(f"  approx_kl (first): {all_logs[0]['approx_kl']:.6f}")
    print(f"  clip_fraction (first): {all_logs[0]['clip_fraction']:.4f}")
