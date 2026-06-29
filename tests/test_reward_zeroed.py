import numpy as np
from quarry.config import Config
from quarry.rollout import compute_reward, RewardState
from quarry.world import create_grid, RIM

cfg = Config()
grid = create_grid(cfg.grid_size)

# --- Test 1: prey on rim, hunters closing, no capture ---
# Hunter at (5,5), prey on rim at (0,3) -> dist=5, no capture
rs = RewardState()
rs.prev_min_dist = 7  # was 7, now 5 -> closing = 2
hunters = [(5, 5)]
prey = (0, 3)
assert grid[prey] == RIM
r, bd = compute_reward(hunters, prey, grid, rs, cfg)
assert bd["rim_drive"] == 0.0, f"rim should be 0.0, got {bd['rim_drive']}"
assert bd["closing"] == 0.0, f"closing should be 0.0, got {bd['closing']}"
assert r == cfg.step_penalty, f"expected {cfg.step_penalty}, got {r}"
print(f"Test 1 PASSED: {bd}")

# --- Test 2: capture (Chebyshev <= 1) ---
rs2 = RewardState()
rs2.prev_min_dist = 2
hunters2 = [(10, 10)]
prey2 = (10, 11)  # dist = 1 -> capture
r2, bd2 = compute_reward(hunters2, prey2, grid, rs2, cfg)
assert r2 == 10.0 + (-0.01), f"expected 9.99, got {r2}"
print(f"Test 2 PASSED: reward={r2}, {bd2}")

# --- Test 3: no capture, off rim, hunters drifting away ---
rs3 = RewardState()
rs3.prev_min_dist = 5
hunters3 = [(3, 3)]
prey3 = (12, 12)  # dist = 9, drifted from 5
r3, bd3 = compute_reward(hunters3, prey3, grid, rs3, cfg)
assert r3 == -0.01, f"expected -0.01, got {r3}"
print(f"Test 3 PASSED: reward={r3}, {bd3}")

# --- Test 4: restore shaping via config ---
cfg_shaped = Config(rim_drive_reward=-1.0, closing_reward_scale=0.1)

rs4 = RewardState()
rs4.prev_min_dist = 7
r4, bd4 = compute_reward([(5, 5)], (0, 3), grid, rs4, cfg_shaped)
# dist=5, closing=7-5=2, on rim
assert bd4["rim_drive"] == -1.0, f"expected -1.0, got {bd4['rim_drive']}"
assert bd4["closing"] == 0.1 * 2, f"expected 0.2, got {bd4['closing']}"
expected = 0.0 + (-1.0) + (-0.01) + 0.2  # no capture
assert abs(r4 - expected) < 1e-9, f"expected {expected}, got {r4}"
print(f"Test 4 PASSED (shaping restored): reward={r4}, {bd4}")

print("\n=== ALL 4 VALIDATIONS PASSED ===")
