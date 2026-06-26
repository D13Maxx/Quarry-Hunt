# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
TEST 2 — Prey spawned adjacent to a hunter → instant capture
=============================================================
Verifies that:
  a) When prey spawns in ANY of the 8 cells directly surrounding a hunter,
     check_win immediately returns 'hunters' (Chebyshev dist == 1).
  b) When prey is exactly 2 cells away (just outside capture radius), no win.
  c) All 8 diagonal directions are covered by the capture rule.
  d) A hunter stepping onto the prey's cell (dist == 0) also wins.

Background: check_win uses max(|dr|, |dc|) <= 1 — so all 8 neighbours
            AND the exact same cell count as "capture."

Run with:
    python tests/test_adjacent_spawn.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quarry.world import create_grid, check_win, DELTAS
from quarry.config import Config

GRID_SIZE = 15          # big enough that hunters won't touch the rim
HUNTER_ANCHOR = (7, 7)  # centre hunter — all adjacency tested around this
MAX_STEPS = 200


def make_grid():
    return create_grid(GRID_SIZE)


# ── helpers ───────────────────────────────────────────────────────────────────

def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def direction_name(dr, dc):
    compass = {(-1, 0): "UP", (1, 0): "DOWN", (0, -1): "LEFT", (0, 1): "RIGHT",
               (-1, -1): "UP-LEFT", (-1, 1): "UP-RIGHT",
               (1, -1): "DOWN-LEFT", (1, 1): "DOWN-RIGHT", (0, 0): "SAME CELL"}
    return compass.get((dr, dc), f"({dr},{dc})")


# ── test 2A — all 8 adjacent cells trigger capture ────────────────────────────

def test_all_8_adjacent_directions():
    """Each of the 8 neighbours of the anchor hunter must yield 'hunters' win."""
    grid = make_grid()
    hr, hc = HUNTER_ANCHOR
    # two decoy hunters far away — won't interfere
    hunters = [HUNTER_ANCHOR, (2, 2), (2, 12)]

    adjacent_offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    passed, failed = 0, []

    for dr, dc in adjacent_offsets:
        prey_pos = (hr + dr, hc + dc)
        dist = chebyshev(HUNTER_ANCHOR, prey_pos)
        assert dist == 1, f"Offset ({dr},{dc}) gives dist={dist}, expected 1"

        result = check_win(grid, hunters, prey_pos, step=1, max_steps=MAX_STEPS)
        label = direction_name(dr, dc)
        if result == "hunters":
            passed += 1
            print(f"  ✓  prey {label} of hunter (dist={dist})  →  '{result}'")
        else:
            failed.append((dr, dc))
            print(f"  ✗  prey {label} of hunter (dist={dist})  →  '{result}'  (EXPECTED 'hunters'!)")

    return passed, failed


# ── test 2B — hunter overlaps prey (dist == 0) ───────────────────────────────

def test_hunter_same_cell_as_prey():
    """Prey on the exact same cell as a hunter must also register as capture."""
    grid = make_grid()
    hunters = [HUNTER_ANCHOR, (2, 2), (2, 12)]
    prey_pos = HUNTER_ANCHOR   # same cell

    result = check_win(grid, hunters, prey_pos, step=1, max_steps=MAX_STEPS)
    dist = chebyshev(HUNTER_ANCHOR, prey_pos)
    if result == "hunters":
        print(f"  ✓  prey on same cell as hunter (dist={dist})  →  '{result}'")
        return 1, []
    else:
        print(f"  ✗  prey on same cell (dist={dist})  →  '{result}'  (EXPECTED 'hunters'!)")
        return 0, [HUNTER_ANCHOR]


# ── test 2C — distance 2 is OUTSIDE capture radius ───────────────────────────

def test_distance_two_no_capture():
    """Prey exactly 2 Chebyshev steps from every hunter → no capture win."""
    grid = make_grid()
    hunters = [HUNTER_ANCHOR, (2, 2), (2, 12)]
    hr, hc = HUNTER_ANCHOR

    # offsets at Chebyshev distance exactly 2 (cardinal + diagonal)
    dist2_offsets = [
        (-2, 0), (2, 0), (0, -2), (0, 2),
        (-2, -2), (-2, 2), (2, -2), (2, 2),
    ]
    passed, failed = 0, []

    for dr, dc in dist2_offsets:
        prey_pos = (hr + dr, hc + dc)
        # make sure this cell isn't inside the other decoy hunters' capture radius
        safe = all(chebyshev(h, prey_pos) > 1 for h in hunters)
        if not safe:
            print(f"  (skip)  {prey_pos} — too close to a decoy hunter")
            continue

        result = check_win(grid, hunters, prey_pos, step=1, max_steps=MAX_STEPS)
        dist = chebyshev(HUNTER_ANCHOR, prey_pos)
        if result is None:
            passed += 1
            print(f"  ✓  prey offset ({dr:+},{dc:+}) dist={dist}  →  no win  (outside capture radius)")
        else:
            failed.append((dr, dc))
            print(f"  ✗  prey offset ({dr:+},{dc:+}) dist={dist}  →  '{result}'  (SHOULD be no win!)")

    return passed, failed


# ── test 2D — full env.step() with adjacent spawn ────────────────────────────

def test_env_step_adjacent_spawn():
    """Integration check: QuarryEnv.step() on the very first tick with
       prey placed one cell east of hunter_0 returns done=True."""
    from quarry.env import QuarryEnv
    from quarry.config import Config
    import numpy as np

    cfg = Config(grid_size=GRID_SIZE)
    env = QuarryEnv(cfg)
    obs, _ = env.reset(seed=0)

    # Manually plant prey one step east of hunter_0 (guaranteed adjacent)
    h0r, h0c = env.hunter_pos[0]
    env.prey_pos = (h0r, h0c + 1)   # Chebyshev dist == 1

    # Both stay-still — just let check_win fire
    actions = {f"hunter_{i}": 0 for i in range(cfg.num_hunters)}
    actions["prey"] = 0

    _, _, terms, _, _ = env.step(actions)
    winner = env.winner

    if winner == "hunters":
        print(f"  ✓  env.step() with adjacent prey  →  winner='{winner}'  done={all(terms.values())}")
        return 1, []
    else:
        print(f"  ✗  env.step() with adjacent prey  →  winner='{winner}'  (EXPECTED 'hunters'!)")
        return 0, ["env_step_adjacent"]


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    all_results = []

    print("\n" + "=" * 60)
    print(" TEST 2A — All 8 adjacent directions trigger capture")
    print("=" * 60)
    p, f = test_all_8_adjacent_directions()
    all_results.append(("2A  8 adjacent directions", p, len(f)))

    print("\n" + "=" * 60)
    print(" TEST 2B — Hunter on same cell as prey → capture")
    print("=" * 60)
    p, f = test_hunter_same_cell_as_prey()
    all_results.append(("2B  same-cell overlap", p, len(f)))

    print("\n" + "=" * 60)
    print(" TEST 2C — Distance 2 is OUTSIDE capture radius")
    print("=" * 60)
    p, f = test_distance_two_no_capture()
    all_results.append(("2C  dist=2 is safe", p, len(f)))

    print("\n" + "=" * 60)
    print(" TEST 2D — Full env.step() integration with adjacent spawn")
    print("=" * 60)
    p, f = test_env_step_adjacent_spawn()
    all_results.append(("2D  env.step() adjacent spawn", p, len(f)))

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    total_p = total_f = 0
    for label, p, f in all_results:
        status = "PASS" if f == 0 else "FAIL"
        print(f"  [{status}]  {label}  ({p} passed, {f} failed)")
        total_p += p; total_f += f
    print(f"\n  Total: {total_p} passed, {total_f} failed")

    if total_f:
        print("\n  ⚠  SOME TESTS FAILED — capture radius has bugs!\n")
        sys.exit(1)
    else:
        print("\n  ✅  ALL ADJACENT-SPAWN TESTS PASSED\n")


if __name__ == "__main__":
    main()
