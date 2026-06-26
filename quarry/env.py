import numpy as np
from gymnasium import spaces
# pyrefly: ignore [missing-import]
from pettingzoo import ParallelEnv

from quarry.config import Config
from quarry.agents import make_hunter_states
from quarry.world import (
    NUM_ACTIONS, NUM_CHANNELS,
    create_grid, spawn_hunters, spawn_prey, apply_move, check_win, ego_obs,
)


class QuarryEnv(ParallelEnv):
    metadata = {"name": "quarry_v0"}

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.possible_agents = [f"hunter_{i}" for i in range(self.cfg.num_hunters)] + ["prey"]
        self.agents = []

    def observation_space(self, agent: str) -> spaces.Box:
        v = self.cfg.prey_vision if agent == "prey" else self.cfg.hunter_vision
        return spaces.Box(0, 1, shape=(NUM_CHANNELS, v, v), dtype=np.float32)

    def action_space(self, agent: str) -> spaces.Discrete:
        return spaces.Discrete(NUM_ACTIONS)

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed)
        self.grid = create_grid(self.cfg.grid_size)
        self.hunter_pos = spawn_hunters(self.cfg.grid_size)
        self.prey_pos = spawn_prey(self.cfg.grid_size, self.rng)
        self.step_count = 0
        self.agents = list(self.possible_agents)
        self.winner = None
        self.hunter_states = make_hunter_states(self.cfg.num_hunters)
        return self._obs(), {a: {} for a in self.agents}

    def step(self, actions):
        sz = self.cfg.grid_size
        self.hunter_pos = [
            apply_move(self.hunter_pos[i], actions.get(f"hunter_{i}", 0), sz)
            for i in range(self.cfg.num_hunters)
        ]
        self.prey_pos = apply_move(self.prey_pos, actions.get("prey", 0), sz)
        self.step_count += 1

        self.winner = check_win(
            self.grid, self.hunter_pos, self.prey_pos,
            self.step_count, self.cfg.max_steps,
        )
        done = self.winner is not None

        obs = self._obs()
        zeros = {a: 0.0 for a in self.possible_agents}
        terms = {a: done for a in self.possible_agents}
        truncs = {a: False for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}

        if done:
            self.agents = []

        return obs, zeros, terms, truncs, infos

    def _obs(self):
        out = {}
        for i in range(self.cfg.num_hunters):
            out[f"hunter_{i}"] = ego_obs(
                self.grid, self.hunter_pos[i], self.cfg.hunter_vision,
                self.hunter_pos, self.prey_pos, i, False,
            )
        out["prey"] = ego_obs(
            self.grid, self.prey_pos, self.cfg.prey_vision,
            self.hunter_pos, self.prey_pos, -1, True,
        )
        return out
