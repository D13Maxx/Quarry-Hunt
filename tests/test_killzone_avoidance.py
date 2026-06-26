# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
TEST 3 — Can hunters enter the kill-zone? Do agents avoid it?
==============================================================
This test answers two related questions from the design:

  Q1: Can a HUNTER physically move into the RIM (kill zone)?
      → Expected: YES. apply_move() and env.step() have NO restriction on
        hunters entering the rim. Only the PREY dies there; hunters are fine.

  Q2: Does the PREY scripted agent (prey_act) actively avoid the rim?
      → Expected: YES (sort-of). prey_act avoids RIM cells via the CH_RIM
        channel check. It won't voluntarily step onto a rim cell IF a safe
        direction exists. But it has no long-range rim-awareness — it only
        checks the cell it's about to enter.

  Q3: Does the HUNTER scripted agent (hunter_act) avoid the rim?
      → Expected: NO. hunter_act picks a move purely toward the prey (or
        random). There is no rim-avoidance logic at all.

The tests verify these facts at three levels:
  3A — apply_move() clamps but does not block hunters at the boundary.
  3B — hunter_act() will happily choose a rim-entering direction.
  3C — prey_act() refuses rim cells when a safer move exists.
  3D — Integration: hunter walks into the rim via env.step(); still alive.

Run with:
    python tests/test_killzone_avoidance.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from quarry.world import (
    create_grid, apply_move, check_win,
    DELTAS, RIM, NUM_CHANNELS,
    CH_FOREST, CH_RIM, CH_SELF, CH_HUNTERS, CH_PREY, CH_MAGIC,
)
from quarry.agents import hunter_act, prey_act
from quarry.config import Config

GRID_SIZE = 15
MAX_STEPS = 200


def make_grid(size=GRID_SIZE):
    return create_grid(size)


# ── test 3A — apply_move does NOT block hunters from entering the rim ─────────

def test_hunter_can_physically_enter_rim():
    """apply_move moves a hunter onto a rim cell without restriction."""
    sz = GRID_SIZE
    passed, failed = 0, []

    # Test all four edges: start one step inside, move outward
    cases = [
        ((1, sz // 2), 1, "UP   → top rim"),     # move up → row 0
        ((sz-2, sz // 2), 3, "DOWN → bottom rim"), # move down → row sz-1
        ((sz // 2, 1), 4, "LEFT → left rim"),      # move left → col 0  (action 4 = right? check DELTAS)
        ((sz // 2, sz-2), 5, "RIGHT → right rim"),
    ]

    # Correct action indices from DELTAS:
    # 0=stay,1=up(-1,0),2=down(+1,0),3=left(0,-1),4=right(0,+1)
    # 5=up-left,6=up-right,7=down-left,8=down-right
    edge_moves = [
        ((1, sz//2),    1, "UP → top rim",     (0, sz//2)),
        ((sz-2, sz//2), 2, "DOWN → bottom rim",(sz-1, sz//2)),
        ((sz//2, 1),    3, "LEFT → left rim",  (sz//2, 0)),
        ((sz//2, sz-2), 4, "RIGHT → right rim",(sz//2, sz-1)),
    ]

    grid = make_grid(sz)
    for start, action, label, expected in edge_moves:
        new_pos = apply_move(start, action, sz)
        on_rim = grid[new_pos] == RIM
        if new_pos == expected and on_rim:
            passed += 1
            print(f"  ✓  Hunter {label}: {start} → {new_pos}  (RIM cell, NOT blocked)")
        else:
            failed.append(label)
            print(f"  ✗  {label}: expected {expected}, got {new_pos}, on_rim={on_rim}")

    return passed, failed


# ── test 3B — hunter_act() picks rim-entering moves freely ───────────────────

def test_hunter_act_no_rim_avoidance():
    """hunter_act should choose to step onto a rim cell when prey is beyond it.
       If hunter_act had rim avoidance, it would pick a different direction.
       We verify that hunter_act's returned action POINTS toward the rim when
       prey is in that direction.
    """
    sz = GRID_SIZE
    rng = np.random.default_rng(42)
    vision = 5
    half = vision // 2

    # Build a minimal observation: hunter at row 1 (one step from top rim),
    # prey 'above' at row 0 (but we put it visibly at the top of the obs window).
    # In ego-obs frame, hunter sees itself at centre (2,2) of a 5x5 window.
    obs = np.zeros((NUM_CHANNELS, vision, vision), dtype=np.float32)
    obs[CH_FOREST] = 1
    obs[CH_SELF, half, half] = 1
    # Place prey one step above center in the observation
    obs[CH_PREY, half - 1, half] = 1
    # The rim is at the TOP of the window (row 0)
    obs[CH_RIM, 0, :] = 1

    action = hunter_act(obs, rng)
    dr, dc = DELTAS[action]
    moves_toward_rim = (dr == -1)   # action UP = toward rim

    if moves_toward_rim:
        print(f"  ✓  hunter_act chose action {action} (UP toward rim) — NO avoidance logic (expected)")
        return 1, []
    else:
        # If it avoids the rim, that's actually a bug relative to current design
        print(f"  ✗  hunter_act chose action {action} (DELTAS={DELTAS[action]}) — avoided rim! (unexpected)")
        return 0, ["hunter_act_avoided_rim"]


# ── test 3C — prey_act() refuses to step onto a rim cell if avoidable ─────────

def test_prey_act_avoids_rim():
    """prey_act checks obs[CH_RIM] at the destination and skips rim moves.
       Set up an observation where only the rim direction is 'toward safety'
       (hunter behind, rim in front) — prey must prefer a non-rim direction.
    """
    vision = 13
    half = vision // 2
    obs = np.zeros((NUM_CHANNELS, vision, vision), dtype=np.float32)
    obs[CH_FOREST] = 1
    obs[CH_SELF, half, half] = 1

    # Hunter directly above the prey in the obs window (prey should flee down)
    obs[CH_HUNTERS, half - 2, half] = 1

    # Mark the cell directly BELOW centre as RIM (to test if prey avoids it)
    obs[CH_RIM, half + 1, half] = 1

    action = prey_act(obs)
    dr, dc = DELTAS[action]

    # action DOWN (2) would step onto the rim cell below — prey should NOT do that
    steps_into_rim = (dr == 1 and dc == 0)

    if not steps_into_rim:
        print(f"  ✓  prey_act chose action {action} DELTAS={DELTAS[action]} — avoided the rim cell below (correct)")
        return 1, []
    else:
        print(f"  ✗  prey_act chose DOWN (onto rim)  →  no avoidance! action={action}")
        return 0, ["prey_stepped_into_rim"]


# ── test 3D — integration: hunter walks onto rim, prey survives, game continues ──

def test_hunter_enters_rim_game_continues():
    """Move hunter_0 to a rim cell via env.step(). The episode must NOT end
       (hunters don't die on the rim; only prey does).
       Prey is placed far from rim and all hunters.
    """
    from quarry.env import QuarryEnv

    cfg = Config(grid_size=GRID_SIZE)
    env = QuarryEnv(cfg)
    obs, _ = env.reset(seed=99)

    # Place hunter_0 one step from the top rim; others in safe interior
    env.hunter_pos = [(1, GRID_SIZE // 2), (7, 5), (7, 9)]
    # Place prey safely in the centre — far from rim and all hunters
    env.prey_pos = (7, 7)

    # hunter_0 moves UP → steps onto rim row 0
    actions = {"hunter_0": 1, "hunter_1": 0, "hunter_2": 0, "prey": 0}
    _, _, terms, _, _ = env.step(actions)

    h0 = env.hunter_pos[0]
    on_rim = env.grid[h0] == RIM
    game_over = any(terms.values())

    if on_rim and not game_over:
        print(f"  ✓  hunter_0 moved to {h0}  on_rim={on_rim}  game_over={game_over}")
        print(f"     → Hunter entered kill zone and is FINE. Game continues. (correct)")
        return 1, []
    elif on_rim and game_over:
        print(f"  ✗  hunter_0 on rim but game ended! winner='{env.winner}'  (hunter shouldn't die on rim!)")
        return 0, ["hunter_died_on_rim"]
    else:
        print(f"  ✗  hunter_0 ended at {h0} — not on rim? Check grid size / action mapping.")
        return 0, ["hunter_not_on_rim"]


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    all_results = []

    print("\n" + "=" * 60)
    print(" TEST 3A — apply_move() does NOT block hunters at the rim")
    print("=" * 60)
    p, f = test_hunter_can_physically_enter_rim()
    all_results.append(("3A  hunter can enter rim physically", p, len(f)))

    print("\n" + "=" * 60)
    print(" TEST 3B — hunter_act() has NO rim avoidance (by design)")
    print("=" * 60)
    p, f = test_hunter_act_no_rim_avoidance()
    all_results.append(("3B  hunter_act no rim avoidance", p, len(f)))

    print("\n" + "=" * 60)
    print(" TEST 3C — prey_act() avoids rim cells when possible")
    print("=" * 60)
    p, f = test_prey_act_avoids_rim()
    all_results.append(("3C  prey_act refuses rim moves", p, len(f)))

    print("\n" + "=" * 60)
    print(" TEST 3D — Integration: hunter walks onto rim, game continues")
    print("=" * 60)
    p, f = test_hunter_enters_rim_game_continues()
    all_results.append(("3D  hunter on rim — game not over", p, len(f)))

    print("\n" + "=" * 60)
    print(" SUMMARY — Kill-zone avoidance behaviour")
    print("=" * 60)
    total_p = total_f = 0
    for label, p, f in all_results:
        status = "PASS" if f == 0 else "FAIL"
        print(f"  [{status}]  {label}  ({p} passed, {f} failed)")
        total_p += p; total_f += f
    print(f"\n  Total: {total_p} passed, {total_f} failed")

    print("\n  Design expectations (from design doc §4.2):")
    print("  • Hunters can freely enter the rim — it is NOT a hazard for them.")
    print("  • Prey entering the rim triggers instant sharpshooter kill → hunters win.")
    print("  • prey_act() has basic 1-step rim avoidance via the CH_RIM obs channel.")
    print("  • hunter_act() has ZERO rim avoidance — it purely chases prey.\n")

    if total_f:
        print("  ⚠  SOME TESTS FAILED — avoidance logic differs from design!\n")
        sys.exit(1)
    else:
        print("  ✅  ALL AVOIDANCE TESTS PASSED\n")


if __name__ == "__main__":
    main()
