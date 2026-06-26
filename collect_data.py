import argparse
import pathlib

import numpy as np
from tqdm import trange

from quarry.config import Config
from quarry.env import QuarryEnv
from quarry.agents import prey_act, hunter_act
from quarry.relative_frame import to_relative
from quarry.world import ego_obs


def collect(cfg: Config, output_dir: pathlib.Path, seed: int = 0):
    output_dir.mkdir(parents=True, exist_ok=True)
    env = QuarryEnv(cfg)
    rng = np.random.default_rng(seed)

    oof_count = total_count = same_count = 0
    ep_lengths = []

    for ep in trange(cfg.num_collect_episodes, desc="collecting"):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        # per-hunter trajectories for this episode
        h_obs = [[] for _ in range(cfg.num_hunters)]
        h_targets = [[] for _ in range(cfg.num_hunters)]
        h_raw_offsets = [[] for _ in range(cfg.num_hunters)]

        while env.agents:
            actions = {
                f"hunter_{i}": hunter_act(
                    obs[f"hunter_{i}"], rng,
                    hunter_pos=env.hunter_pos[i],
                    state=env.hunter_states[i],
                    cfg=cfg,
                )
                for i in range(cfg.num_hunters)
            }
            actions["prey"] = prey_act(obs["prey"])

            prev_prey = env.prey_pos
            obs, _, _, _, _ = env.step(actions)
            next_prey = env.prey_pos

            for i in range(cfg.num_hunters):
                h_obs[i].append(obs[f"hunter_{i}"])
                r, c, oof = to_relative(env.hunter_pos[i], next_prey, cfg.window_K)
                h_targets[i].append((r, c))
                dr = next_prey[0] - env.hunter_pos[i][0]
                dc = next_prey[1] - env.hunter_pos[i][1]
                h_raw_offsets[i].append((dr, dc))
                oof_count += oof
                same_count += (prev_prey == next_prey)
                total_count += 1

        ep_lengths.append(env.step_count)

        for i in range(cfg.num_hunters):
            obs_arr = (np.array(h_obs[i]) * 255).astype(np.uint8)
            tgt_arr = np.array(h_targets[i], dtype=np.int16)
            raw_arr = np.array(h_raw_offsets[i], dtype=np.int16)
            np.save(output_dir / f"obs_{ep}_{i}.npy", obs_arr)
            np.save(output_dir / f"target_{ep}_{i}.npy", tgt_arr)
            np.save(output_dir / f"raw_offset_{ep}_{i}.npy", raw_arr)

    print(f"\n{cfg.num_collect_episodes} episodes, {total_count} total steps")
    print(f"episode length: mean={np.mean(ep_lengths):.1f} median={np.median(ep_lengths):.0f}")
    print(f"OOF rate: {oof_count/total_count:.3%}")
    print(f"same-cell rate: {same_count/total_count:.3%}")

    meta = {
        "episodes": cfg.num_collect_episodes,
        "total_steps": total_count,
        "oof_rate": oof_count / total_count,
        "same_cell_rate": same_count / total_count,
        "seed": seed,
        "window_K": cfg.window_K,
        "grid_size": cfg.grid_size,
    }
    np.savez(output_dir / "meta.npz", **meta)


if __name__ == "__main__":
    _defaults = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=_defaults.num_collect_episodes)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--grid-size", type=int, default=_defaults.grid_size)
    parser.add_argument("--window-K", type=int, default=_defaults.window_K)
    args = parser.parse_args()

    cfg = Config(
        num_collect_episodes=args.episodes,
        grid_size=args.grid_size,
        window_K=args.window_K,
    )
    collect(cfg, pathlib.Path(args.output), seed=args.seed)
