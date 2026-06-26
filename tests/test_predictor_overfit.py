import torch
from quarry.config import Config
from quarry.predictor import PreyPredictor
from quarry.env import QuarryEnv
from quarry.actor import Actor
from quarry.rollout import collect_episode
from quarry.train_loop import update_predictor, pred_batch_from_rollout


def test_overfit_one_batch():
    cfg = Config()
    predictor = PreyPredictor(cfg)  # fresh, no checkpoint
    actor = Actor(cfg)
    env = QuarryEnv(cfg)

    # collect a small episode for training data
    result = collect_episode(env, actor, predictor, cfg)
    pred_buf = result["predictor"]
    # take a small fixed batch (first 32 or all if fewer)
    batch = pred_buf[:min(32, len(pred_buf))]
    obs_seq, target_r, target_c = pred_batch_from_rollout(batch)

    pred_optimizer = torch.optim.Adam(predictor.parameters(), lr=cfg.predictor_lr)

    # confirm optimizer is separate from actor
    actor_param_ids = {id(p) for p in actor.parameters()}
    pred_param_ids = {id(p) for p in predictor.parameters()}
    assert actor_param_ids.isdisjoint(pred_param_ids), "predictor shares params with actor"

    predictor.train()
    loss_start, hit_start = None, None
    loss_end, hit_end = None, None

    for step in range(500):
        loss, hit = update_predictor(predictor, pred_optimizer, obs_seq, target_r, target_c)
        if step == 0:
            loss_start, hit_start = loss, hit
        if step == 499:
            loss_end, hit_end = loss, hit

    print(f"  start: loss={loss_start:.4f}  top1={hit_start:.3f}")
    print(f"  end:   loss={loss_end:.4f}  top1={hit_end:.3f}")

    assert loss_end < loss_start * 0.05, f"loss didn't drop enough: {loss_start:.4f} -> {loss_end:.4f}"
    assert hit_end > 0.85, f"top1 hit rate too low after overfit: {hit_end:.3f}"
    assert loss_end < 1.0, f"loss still high: {loss_end:.4f}"
