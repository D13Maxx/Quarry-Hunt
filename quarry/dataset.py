import pathlib
import re

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):

    def __init__(self, data_dir: str | pathlib.Path, episode_ids: list[int],
                 num_hunters: int = 3, history_len: int = 8):
        self.history_len = history_len
        self.samples = []

        data_dir = pathlib.Path(data_dir)
        for ep in episode_ids:
            for h in range(num_hunters):
                obs_path = data_dir / f"obs_{ep}_{h}.npy"
                tgt_path = data_dir / f"target_{ep}_{h}.npy"
                if not obs_path.exists():
                    continue
                T = np.load(tgt_path, mmap_mode="r").shape[0]
                for t in range(history_len - 1, T):
                    self.samples.append((str(obs_path), str(tgt_path), t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_path, tgt_path, t = self.samples[idx]
        obs_mmap = np.load(obs_path, mmap_mode="r")
        tgt_mmap = np.load(tgt_path, mmap_mode="r")

        start = t - self.history_len + 1
        obs_window = obs_mmap[start:t + 1].astype(np.float32) / 255.0
        target = tgt_mmap[t]
        return torch.from_numpy(obs_window.copy()), int(target[0]), int(target[1])


def split_episodes(data_dir: str | pathlib.Path, train_frac: float = 0.8) -> tuple[list[int], list[int]]:
    data_dir = pathlib.Path(data_dir)
    eps = set()
    for f in data_dir.glob("obs_*_*.npy"):
        m = re.match(r"obs_(\d+)_\d+\.npy", f.name)
        if m:
            eps.add(int(m.group(1)))
    eps = sorted(eps)
    split = int(len(eps) * train_frac)
    return eps[:split], eps[split:]
