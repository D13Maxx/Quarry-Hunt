import numpy as np
from quarry.config import Config
from quarry.rollout import compute_reward, RewardState, _min_chebyshev
from quarry.world import create_grid, RIM


def _cfg():
    return Config(
        capture_reward=10.0,
        rim_drive_reward=-1.0,
        step_penalty=-0.01,
        closing_reward_scale=0.1,
    )


def test_capture():
    cfg = _cfg()
    grid = create_grid(cfg.grid_size)
    hunters = [(10, 10), (5, 5), (20, 20)]
    prey = (10, 11)  # Chebyshev 1 from hunter 0
    rs = RewardState()
    rs.reset(hunters, prey)
    total, bd = compute_reward(hunters, prey, grid, rs, cfg)
    assert bd["capture"] == 10.0
    assert bd["rim_drive"] == 0.0
    assert total > 0, "capture should dominate"


def test_prey_on_rim():
    cfg = _cfg()
    grid = create_grid(cfg.grid_size)
    hunters = [(3, 3), (4, 4), (5, 5)]
    prey = (0, 12)  # row 0 is rim
    assert grid[prey] == RIM
    rs = RewardState()
    rs.reset(hunters, prey)
    total, bd = compute_reward(hunters, prey, grid, rs, cfg)
    assert bd["rim_drive"] == -1.0


def test_closing():
    cfg = _cfg()
    grid = create_grid(cfg.grid_size)
    # step 1: hunters far
    hunters_far = [(10, 10), (15, 15), (20, 20)]
    prey = (5, 5)
    rs = RewardState()
    rs.reset(hunters_far, prey)  # prev_min_dist = 5

    # step 2: one hunter closes to dist 3
    hunters_close = [(8, 8), (15, 15), (20, 20)]
    total, bd = compute_reward(hunters_close, prey, grid, rs, cfg)
    assert bd["closing"] == 0.1 * (5 - 3), f"got {bd['closing']}"
    assert bd["closing"] > 0


def test_gap_opening():
    cfg = _cfg()
    grid = create_grid(cfg.grid_size)
    hunters_close = [(6, 6), (15, 15), (20, 20)]
    prey = (5, 5)
    rs = RewardState()
    rs.reset(hunters_close, prey)  # prev_min_dist = 1

    hunters_far = [(10, 10), (15, 15), (20, 20)]
    total, bd = compute_reward(hunters_far, prey, grid, rs, cfg)
    assert bd["closing"] == 0.1 * (1 - 5)
    assert bd["closing"] < 0, "gap opening should be negative"


def test_plain_step():
    cfg = _cfg()
    grid = create_grid(cfg.grid_size)
    hunters = [(10, 10), (15, 15), (20, 20)]
    prey = (5, 5)
    rs = RewardState()
    rs.reset(hunters, prey)  # prev_min_dist = 5

    # same positions — no capture, no rim, no closing
    total, bd = compute_reward(hunters, prey, grid, rs, cfg)
    assert bd["capture"] == 0.0
    assert bd["rim_drive"] == 0.0
    assert bd["closing"] == 0.0
    assert bd["step_penalty"] == -0.01
    assert total == -0.01
