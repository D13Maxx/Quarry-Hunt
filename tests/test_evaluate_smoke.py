import shutil

import torch

from quarry.config import Config
from quarry.actor import Actor
from quarry.predictor import PreyPredictor
from quarry.evaluate import evaluate, print_report


def test_untrained_baseline():
    cfg = Config()
    actor = Actor(cfg)
    predictor = PreyPredictor(cfg)
    save_dir = "eval_test_output"

    try:
        result = evaluate(actor, predictor, cfg, n_episodes=50, save_dir=save_dir)
        print_report(result)

        K = cfg.window_K
        chance_top1 = 1.0 / (K * K)

        # capture rate should be low for random actors
        assert result["capture_rate"] < 0.5, f"capture rate suspiciously high: {result['capture_rate']}"

        # predictor accuracy near chance
        assert result["pred_top1"] < chance_top1 * 5, \
            f"pred top1 {result['pred_top1']:.4f} too high for untrained (chance={chance_top1:.4f})"

        # all metrics computed and finite
        for k in ("capture_rate", "pred_top1", "pred_argmax_cheby",
                   "pred_expected_cheby", "pred_true_cell_prob", "offset_mean"):
            assert torch.isfinite(torch.tensor(result[k])), f"{k} not finite"

        # caught + rim + timeout == n_episodes
        assert result["captures"] + result["rim_drives"] + result["timeouts"] == 50

        # histogram saved
        import pathlib
        assert (pathlib.Path(save_dir) / "offset_histogram.png").exists()

        print(f"chance top1 = {chance_top1:.5f}, actual = {result['pred_top1']:.5f}")
    finally:
        shutil.rmtree(save_dir, ignore_errors=True)
