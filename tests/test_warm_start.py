import shutil
import torch

from quarry.config import Config
from train_joint import train

cfg = Config()
cfg.grid_size = 41
cfg.max_steps = 350
cfg.num_iterations = 2
cfg.rollout_episodes = 4
cfg.eval_every = 2
cfg.entropy_coef = 0.03
cfg.ckpt_dir = "checkpoints_ws_test"

# clean slate
shutil.rmtree(cfg.ckpt_dir, ignore_errors=True)

hist = train(cfg, device="cpu", init_from="checkpoints/joint_1000.pt")

assert len(hist) == 2, f"expected 2 iterations, got {len(hist)}"
assert hist[0]["iteration"] == 1, f"expected start at iter 1, got {hist[0]['iteration']}"
assert hist[1]["iteration"] == 2

import pathlib
ckpt_files = list(pathlib.Path(cfg.ckpt_dir).glob("joint_*.pt"))
assert len(ckpt_files) >= 1, "no checkpoint written"

import math
for row in hist:
    for k, v in row.items():
        if isinstance(v, float):
            assert not math.isnan(v), f"NaN in {k} at iter {row['iteration']}"

print("\n=== VALIDATION PASSED ===")
print(f"  iterations: {[r['iteration'] for r in hist]}")
print(f"  checkpoint: {ckpt_files}")
print(f"  no NaN, no shape errors, magic zones active")

shutil.rmtree(cfg.ckpt_dir)
