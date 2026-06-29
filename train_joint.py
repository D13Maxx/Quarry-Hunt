import pathlib

import torch

from quarry.config import Config
from quarry.predictor import PreyPredictor
from quarry.actor import Actor
from quarry.critic import CentralizedCritic
from quarry.env import QuarryEnv
from quarry.rollout import collect_episode
from quarry.train_loop import update_predictor, pred_batch_from_rollout
from quarry.ppo import ppo_update


def _save_checkpoint(actor, critic, predictor, pred_opt, ppo_opt, cfg, iteration):
    ckpt_dir = pathlib.Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iteration": iteration,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "predictor": predictor.state_dict(),
        "pred_opt": pred_opt.state_dict(),
        "ppo_opt": ppo_opt.state_dict(),
    }, ckpt_dir / f"joint_{iteration}.pt")


def _find_latest_checkpoint(cfg: Config) -> pathlib.Path | None:
    ckpt_dir = pathlib.Path(cfg.ckpt_dir)
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("joint_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


def load_checkpoint(path, actor, critic, predictor, pred_opt, ppo_opt, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    predictor.load_state_dict(ckpt["predictor"])
    pred_opt.load_state_dict(ckpt["pred_opt"])
    ppo_opt.load_state_dict(ckpt["ppo_opt"])
    return ckpt["iteration"]


def load_weights_only(path, actor, critic, predictor, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    predictor.load_state_dict(ckpt["predictor"])
    print(f"Warm-started weights from {path} (fresh optimizers, starting iteration 1)")


def train(cfg: Config, device: str = "cpu", resume: bool = False,
          init_from: str | None = None):
    dev = torch.device(device)
    predictor = PreyPredictor(cfg).to(dev)
    actor = Actor(cfg).to(dev)
    critic = CentralizedCritic(cfg).to(dev)
    env = QuarryEnv(cfg)

    pred_opt = torch.optim.Adam(predictor.parameters(), lr=cfg.predictor_lr)
    ppo_opt = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=cfg.ppo_lr,
    )

    start_it = 0
    if resume:
        if init_from:
            print("NOTE: both --resume and --init-from set; resume takes precedence")
        ckpt_path = _find_latest_checkpoint(cfg)
        if ckpt_path:
            start_it = load_checkpoint(ckpt_path, actor, critic, predictor,
                                       pred_opt, ppo_opt, device)
            print(f"Resumed from {ckpt_path} (iteration {start_it})")
    elif init_from:
        load_weights_only(init_from, actor, critic, predictor, device)

    history = []

    for it in range(start_it + 1, cfg.num_iterations + 1):
        episodes = []
        for _ in range(cfg.rollout_episodes):
            ep = collect_episode(env, actor, predictor, cfg, dev)
            episodes.append(ep)

        pred_buf = []
        for ep in episodes:
            pred_buf.extend(ep["predictor"])
        obs_seq, target_r, target_c = pred_batch_from_rollout(pred_buf, dev)
        predictor.train()
        pred_loss, pred_hit = update_predictor(predictor, pred_opt, obs_seq, target_r, target_c)

        actor.train()
        critic.train()
        ppo_logs = ppo_update(actor, critic, ppo_opt, episodes, cfg, dev)

        captures = sum(1 for ep in episodes if ep["winner"] == "hunters")
        capture_rate = captures / len(episodes)
        avg_len = sum(ep["length"] for ep in episodes) / len(episodes)
        last_ppo = ppo_logs[-1]

        row = {
            "iteration": it,
            "pred_loss": pred_loss,
            "pred_hit": pred_hit,
            "policy_loss": last_ppo["policy_loss"],
            "value_loss": last_ppo["value_loss"],
            "entropy": last_ppo["entropy"],
            "capture_rate": capture_rate,
            "avg_ep_len": avg_len,
        }
        history.append(row)

        print(f"[{it:4d}/{cfg.num_iterations}]  "
              f"pred={pred_loss:.4f}/{pred_hit:.3f}  "
              f"pi={last_ppo['policy_loss']:.4f}  V={last_ppo['value_loss']:.4f}  "
              f"H={last_ppo['entropy']:.3f}  "
              f"cap={capture_rate:.2f}  len={avg_len:.0f}")

        if it % cfg.eval_every == 0:
            _save_checkpoint(actor, critic, predictor, pred_opt, ppo_opt, cfg, it)
            print(f"  -> checkpoint saved at iteration {it}")

    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--init-from", default=None, help="Warm-start weights from checkpoint (fresh optimizers)")
    args = parser.parse_args()
    train(Config(), device=args.device, resume=args.resume, init_from=args.init_from)
