import argparse
import pathlib

import numpy as np
import torch

from quarry.config import Config
from quarry.train_loop import train
from quarry.dataset import TrajectoryDataset, split_episodes
from quarry.metrics import uniform_baseline, stay_baseline, last_seen_baseline


def run_baselines(cfg: Config, data_dir: str):
    _, val_eps = split_episodes(data_dir, cfg.train_val_split)
    ds = TrajectoryDataset(data_dir, val_eps, cfg.num_hunters, cfg.history_len)
    K = cfg.window_K

    all_obs, all_tr, all_tc = [], [], []
    for i in range(len(ds)):
        obs, tr, tc = ds[i]
        all_obs.append(obs.numpy())
        all_tr.append(tr)
        all_tc.append(tc)

    tr_arr = np.array(all_tr)
    tc_arr = np.array(all_tc)

    uni = uniform_baseline(K)
    sta = stay_baseline(tr_arr, tc_arr, K)
    las = last_seen_baseline(all_obs, tr_arr, tc_arr, K)

    print(f"\n{'Baseline':<12} {'top1_hit':>10} {'argmax_cheby':>14}")
    print("-" * 40)
    print(f"{'uniform':<12} {uni['top1_hit']:>10.4f} {'':>14}")
    print(f"{'stay':<12} {sta['top1_hit']:>10.4f} {sta['argmax_cheby']:>14.2f}")
    print(f"{'last_seen':<12} {las['top1_hit']:>10.4f} {las['argmax_cheby']:>14.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-baselines", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = Config(lr=args.lr, batch_size=args.batch_size)
    pathlib.Path(cfg.predictor_ckpt).parent.mkdir(parents=True, exist_ok=True)

    if args.run_baselines:
        run_baselines(cfg, args.data)
    else:
        train(cfg, args.data, device=args.device, epochs=args.epochs)
