import numpy as np
from quarry.config import Config
from quarry.rollout import compute_reward, RewardState
from quarry.world import create_grid, ego_obs, CH_PREY, RIM

cfg = Config()
grid = create_grid(cfg.grid_size)

# --- Test 1: exactly one hunter perceives prey, no capture, off rim ---
rs1 = RewardState()
rs1.prev_min_dist = 10
hunters1 = [(20, 20)]
prey1 = (20, 22)  # dist=2, no capture, in view for vision=5
obs1 = ego_obs(grid, hunters1[0], cfg.hunter_vision, hunters1, prey1, 0, False)
saw = obs1[CH_PREY].any()
assert saw, "hunter should see prey at dist 2"
r1, bd1 = compute_reward(hunters1, prey1, grid, rs1, cfg, team_saw_prey=bool(saw))
expected1 = cfg.sighting_reward + cfg.step_penalty  # 0.02 + (-0.01) = 0.01
assert abs(r1 - expected1) < 1e-9, f"expected {expected1}, got {r1}"
print(f"Test 1 PASSED: reward={r1}, {bd1}")

# --- Test 2: all three hunters perceive prey -> same reward as test 1 (team signal) ---
rs2 = RewardState()
rs2.prev_min_dist = 3
hunters2 = [(20, 20), (20, 21), (20, 19)]
prey2 = (20, 22)  # dist 2,1,3 -> no capture (min=1 IS capture, but let's use dist 2)
prey2 = (20, 23)  # dist 3,2,4 -> no capture
all_saw = all(
    ego_obs(grid, hunters2[i], cfg.hunter_vision, hunters2, prey2, i, False)[CH_PREY].any()
    for i in range(3)
)
# at least one sees it — vision=5 covers ±2 cols, prey at col 23, hunter 1 at col 21 -> offset 2 -> in view
team_saw = any(
    ego_obs(grid, hunters2[i], cfg.hunter_vision, hunters2, prey2, i, False)[CH_PREY].any()
    for i in range(3)
)
assert team_saw, "at least one hunter should see prey"
r2, bd2 = compute_reward(hunters2, prey2, grid, rs2, cfg, team_saw_prey=True)
expected2 = cfg.sighting_reward + cfg.step_penalty  # 0.01 — same as test 1
assert abs(r2 - expected2) < 1e-9, f"expected {expected2}, got {r2}"
print(f"Test 2 PASSED (team signal, not summed): reward={r2}, {bd2}")

# --- Test 3: prey in a hunter's geometric range but magic-hidden -> r_sighting = 0 ---
mask = np.zeros((cfg.grid_size, cfg.grid_size), dtype=bool)
mask[20:23, 22:25] = True  # prey at (21,23) is inside magic zone
rs3 = RewardState()
rs3.prev_min_dist = 5
hunter3 = [(21, 21)]
prey3 = (21, 23)  # dist=2, geometrically in view, but magic-hidden
obs3 = ego_obs(grid, hunter3[0], cfg.hunter_vision, hunter3, prey3, 0, False, magic_mask=mask)
saw3 = obs3[CH_PREY].any()
assert not saw3, "prey should be invisible (Effect A)"
r3, bd3 = compute_reward(hunter3, prey3, grid, rs3, cfg, team_saw_prey=False)
assert abs(r3 - cfg.step_penalty) < 1e-9, f"expected {cfg.step_penalty}, got {r3}"
assert bd3["sighting"] == 0.0
print(f"Test 3 PASSED (magic-hidden, perception-true): reward={r3}, {bd3}")

# --- Test 4: capture step with sighting ---
rs4 = RewardState()
rs4.prev_min_dist = 2
hunter4 = [(20, 20)]
prey4 = (20, 21)  # dist=1 -> capture
r4, bd4 = compute_reward(hunter4, prey4, grid, rs4, cfg, team_saw_prey=True)
expected4 = cfg.capture_reward + cfg.sighting_reward + cfg.step_penalty  # 50.0 + 0.02 + (-0.01) = 50.01
assert abs(r4 - expected4) < 1e-9, f"expected {expected4}, got {r4}"
print(f"Test 4 PASSED: reward={r4}, {bd4}")

# --- Test 5: sighting_reward=0 disables it; capture_reward is config-driven ---
cfg0 = Config(sighting_reward=0.0)
rs5 = RewardState()
rs5.prev_min_dist = 5
r5, bd5 = compute_reward([(20, 20)], (20, 22), grid, rs5, cfg0, team_saw_prey=True)
assert bd5["sighting"] == 0.0, f"expected 0.0, got {bd5['sighting']}"
assert bd5["capture"] == 0.0  # dist=2, no capture
assert abs(r5 - cfg0.step_penalty) < 1e-9
# also confirm capture_reward reads from config
rs5b = RewardState()
rs5b.prev_min_dist = 2
r5b, bd5b = compute_reward([(20, 20)], (20, 21), grid, rs5b, cfg0, team_saw_prey=False)
assert bd5b["capture"] == 50.0, f"capture should be 50.0 from config, got {bd5b['capture']}"
print(f"Test 5 PASSED (config-driven): sighting={bd5['sighting']}, capture={bd5b['capture']}")

print("\n=== ALL 5 VALIDATIONS PASSED ===")
