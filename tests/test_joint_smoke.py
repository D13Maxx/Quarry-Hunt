import pathlib
import shutil

import torch

from quarry.config import Config
from quarry.predictor import PreyPredictor
from quarry.actor import Actor
from quarry.critic import CentralizedCritic
from train_joint import train, load_checkpoint


def test_tiny_run():
    ckpt_dir = "checkpoints_test"
    cfg = Config(
        num_iterations=5,
        rollout_episodes=4,
        max_steps=30,
        eval_every=5,
        ckpt_dir=ckpt_dir,
    )

    try:
        history = train(cfg, device="cpu")

        # completes with 5 iterations logged
        assert len(history) == 5

        # no NaN in any logged value
        for row in history:
            for k, v in row.items():
                if isinstance(v, float):
                    assert torch.isfinite(torch.tensor(v)), f"iter {row['iteration']} {k} = {v}"

        # both predictor and PPO losses logged each iteration
        for row in history:
            assert "pred_loss" in row and "policy_loss" in row and "value_loss" in row

        # checkpoint written at iteration 5 (eval_every=5)
        ckpt_path = pathlib.Path(ckpt_dir) / "joint_5.pt"
        assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"

        # checkpoint reloads cleanly
        actor2 = Actor(cfg)
        critic2 = CentralizedCritic(cfg)
        predictor2 = PreyPredictor(cfg)
        pred_opt2 = torch.optim.Adam(predictor2.parameters(), lr=cfg.predictor_lr)
        ppo_opt2 = torch.optim.Adam(
            list(actor2.parameters()) + list(critic2.parameters()), lr=cfg.ppo_lr,
        )
        it = load_checkpoint(ckpt_path, actor2, critic2, predictor2, pred_opt2, ppo_opt2)
        assert it == 5

        print(f"  iterations: {len(history)}")
        print(f"  pred_loss: {[f'{r['pred_loss']:.4f}' for r in history]}")
        print(f"  capture_rate: {[r['capture_rate'] for r in history]}")
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
