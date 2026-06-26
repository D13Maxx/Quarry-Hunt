import pathlib
import shutil
import subprocess
import sys

import torch

from quarry.config import Config
from quarry.actor import Actor
from quarry.critic import CentralizedCritic
from quarry.predictor import PreyPredictor


def test_render_joint_episode():
    ckpt_dir = "checkpoints_render_test"
    gif_path = "test_render.gif"

    cfg = Config(
        num_iterations=2,
        rollout_episodes=2,
        max_steps=20,
        eval_every=2,
        ckpt_dir=ckpt_dir,
    )

    try:
        # train a tiny checkpoint
        from train_joint import train
        train(cfg, device="cpu")

        ckpt_path = pathlib.Path(ckpt_dir) / "joint_2.pt"
        assert ckpt_path.exists(), "checkpoint not created"

        # run demo headless with joint checkpoint
        result = subprocess.run(
            [sys.executable, "run_demo.py",
             "--joint", str(ckpt_path),
             "--headless",
             "--gif", gif_path],
            capture_output=True, text=True, timeout=60,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
        assert result.returncode == 0, f"run_demo failed: {result.stderr[-300:]}"

        gif = pathlib.Path(gif_path)
        assert gif.exists(), "GIF not created"
        assert gif.stat().st_size > 1000, f"GIF too small: {gif.stat().st_size} bytes"
        print(f"  GIF saved: {gif.stat().st_size:,} bytes")

        # verify checkpoint loads cleanly
        actor = Actor(cfg)
        predictor = PreyPredictor(cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        actor.load_state_dict(ckpt["actor"])
        predictor.load_state_dict(ckpt["predictor"])

    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        pathlib.Path(gif_path).unlink(missing_ok=True)
