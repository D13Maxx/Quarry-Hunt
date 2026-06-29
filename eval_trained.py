import numpy as np
import torch

from quarry.config import Config
from quarry.actor import Actor
from quarry.predictor import PreyPredictor
from quarry.evaluate import evaluate, print_report

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load("checkpoints/joint_1000.pt", map_location=device, weights_only=False)
actor = Actor(cfg).to(device);        actor.load_state_dict(ckpt["actor"])
predictor = PreyPredictor(cfg).to(device); predictor.load_state_dict(ckpt["predictor"])

result = evaluate(actor, predictor, cfg, n_episodes=200, device=device, save_dir="eval_trained")
print_report(result)

off = result["offsets"]
print("offset percentiles (min hunter-prey Chebyshev):")
for p in [50, 75, 90, 95, 99]:
    print(f"  p{p:02d} = {np.percentile(off, p):.0f}")
print(f"  max = {int(off.max())}")
