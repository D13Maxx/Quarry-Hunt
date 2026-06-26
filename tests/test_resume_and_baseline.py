import json
import pathlib
import shutil

from quarry.config import Config
from quarry.actor import Actor
from quarry.predictor import PreyPredictor
from quarry.evaluate import evaluate, print_report
from train_joint import train


def test_resume():
    ckpt_dir = "checkpoints_resume_test"
    cfg = Config(
        num_iterations=6,
        rollout_episodes=2,
        max_steps=20,
        eval_every=3,
        ckpt_dir=ckpt_dir,
    )

    try:
        # run 1: train iterations 1-6, checkpoint at 3 and 6
        h1 = train(cfg, device="cpu")
        assert len(h1) == 6
        assert (pathlib.Path(ckpt_dir) / "joint_3.pt").exists()
        assert (pathlib.Path(ckpt_dir) / "joint_6.pt").exists()

        # delete the iter-6 checkpoint to simulate a crash after iter 5
        (pathlib.Path(ckpt_dir) / "joint_6.pt").unlink()

        # run 2: resume — should pick up from iter 3 and run 4-6
        h2 = train(cfg, device="cpu", resume=True)
        assert len(h2) == 3, f"expected 3 iterations after resume, got {len(h2)}"
        assert h2[0]["iteration"] == 4, f"expected first resumed iter=4, got {h2[0]['iteration']}"
        assert h2[-1]["iteration"] == 6

        # checkpoint at 6 should now exist again
        assert (pathlib.Path(ckpt_dir) / "joint_6.pt").exists()
        print(f"  run1: {len(h1)} iterations")
        print(f"  run2 (resumed from 3): {len(h2)} iterations ({h2[0]['iteration']}-{h2[-1]['iteration']})")
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_baseline_saved():
    """Run untrained baseline and save results to disk."""
    cfg = Config()
    actor = Actor(cfg)
    predictor = PreyPredictor(cfg)
    save_dir = "baselines"

    result = evaluate(actor, predictor, cfg, n_episodes=100, save_dir=save_dir)
    print_report(result)

    # save numeric results
    out = pathlib.Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in result.items() if k != "offsets"}
    serializable["offset_percentiles"] = {
        "p50": float(result["offset_median"]),
        "p90": float(result["offsets"].tolist()[int(len(result["offsets"]) * 0.9)]) if len(result["offsets"]) > 0 else 0,
    }
    with open(out / "untrained_baseline.json", "w") as f:
        json.dump(serializable, f, indent=2)

    assert (out / "untrained_baseline.json").exists()
    assert (out / "offset_histogram.png").exists()
    print(f"  Saved to {out}/")
