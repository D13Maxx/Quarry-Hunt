# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
TEST 1 — Kill-Zone (RIM) works correctly
=========================================
Verifies that:
  a) Prey stepping onto ANY rim cell immediately ends the episode as 'hunters' win.
  b) Prey in interior cells does NOT trigger the rim-kill rule.
  c) The rim is exactly 1 cell wide (not 0, not 2).
  d) Rule fires on all four sides AND corners.

Run with:
    python tests/test_killzone.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quarry.world import create_grid, check_win, RIM, FOREST
from quarry.config import Config

GRID_SIZE = 10


def make_grid(size=GRID_SIZE):
    return create_grid(size)


# Hunters parked in the interior — Chebyshev dist >= 2 from EVERY test cell:
#   Interior test cells: (2,2), (3,7), (5,5), (7,3), (8,8)
#   Rim test cells:      (0,0), (0,5), (0,9), (9,0), (9,5), (9,9), (5,0), (5,9)
# Verified:
#   (5,3): min dist to interior = 2 [(5,5),(7,3)], min dist to rim = 3 [(5,0)]
#   (3,5): min dist to interior = 2 [(3,7),(5,5)], min dist to rim = 3 [(0,5)]
#   (7,6): min dist to interior = 2 [(5,5),(8,8)], min dist to rim = 2 [(9,5)]
SAFE_HUNTERS = [(5, 3), (3, 5), (7, 6)]


# ── helpers ───────────────────────────────────────────────────────────────────

def check_rim(grid, pos):
    return grid[pos] == RIM


# ── test 1A — rim fires on every side & corner ────────────────────────────────

def test_prey_on_rim_triggers_win():
    grid = make_grid()
    sz = GRID_SIZE
    rim_cells = [
        (0, 0),         # top-left corner
        (0, sz // 2),   # top edge (mid)
        (0, sz - 1),    # top-right corner
        (sz - 1, 0),    # bottom-left corner
        (sz - 1, sz // 2),  # bottom edge (mid)
        (sz - 1, sz - 1),   # bottom-right corner
        (sz // 2, 0),   # left edge (mid)
        (sz // 2, sz - 1),  # right edge (mid)
    ]
    passed, failed = 0, []
    for prey_pos in rim_cells:
        assert check_rim(grid, prey_pos), (
            f"Setup error: {prey_pos} should be RIM but grid says otherwise"
        )
        result = check_win(grid, SAFE_HUNTERS, prey_pos, step=1, max_steps=200)
        if result == "hunters":
            passed += 1
            print(f"  ✓  prey @ {prey_pos}  →  '{result}'  (kill-zone fired)")
        else:
            failed.append(prey_pos)
            print(f"  ✗  prey @ {prey_pos}  →  '{result}'  (EXPECTED 'hunters'!)")
    return passed, failed


# ── test 1B — interior cells are safe ────────────────────────────────────────

def test_prey_interior_no_rim_kill():
    grid = make_grid()
    interior_cells = [(2, 2), (3, 7), (5, 5), (7, 3), (8, 8)]
    passed, failed = 0, []
    for prey_pos in interior_cells:
        assert not check_rim(grid, prey_pos), (
            f"Setup error: {prey_pos} flagged as RIM unexpectedly"
        )
        result = check_win(grid, SAFE_HUNTERS, prey_pos, step=1, max_steps=200)
        if result is None:
            passed += 1
            print(f"  ✓  prey @ {prey_pos}  →  no win  (interior is safe ✓)")
        else:
            failed.append(prey_pos)
            print(f"  ✗  prey @ {prey_pos}  →  '{result}'  (UNEXPECTED win!)")
    return passed, failed


# ── test 1C — rim is exactly 1 cell wide ─────────────────────────────────────

def test_rim_width_is_one_cell():
    grid = make_grid()
    sz = GRID_SIZE
    errors = []

    # cells at index 1 and sz-2 must be FOREST, not RIM
    inner = [
        (1, 1), (1, sz - 2), (sz - 2, 1), (sz - 2, sz - 2),
        (sz // 2, 1), (1, sz // 2),
    ]
    for pos in inner:
        if grid[pos] == RIM:
            errors.append(pos)
            print(f"  ✗  {pos} should be FOREST but is RIM (rim wider than 1 cell!)")
        else:
            print(f"  ✓  {pos} is FOREST (rim is exactly 1 cell wide)")

    # row/col 0 and sz-1 must all be RIM
    rim_sample = [(0, sz // 2), (sz // 2, 0), (sz - 1, sz // 2), (sz // 2, sz - 1)]
    for pos in rim_sample:
        if grid[pos] != RIM:
            errors.append(pos)
            print(f"  ✗  {pos} should be RIM but is NOT")
        else:
            print(f"  ✓  {pos} is RIM (confirmed)")

    total = len(inner) + len(rim_sample)
    return total - len(errors), errors


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    all_results = []

    print("\n" + "=" * 60)
    print(" TEST 1A — Prey on RIM → 'hunters' win (all sides & corners)")
    print("=" * 60)
    p, f = test_prey_on_rim_triggers_win()
    all_results.append(("1A  rim fires on all sides/corners", p, len(f)))

    print("\n" + "=" * 60)
    print(" TEST 1B — Prey in interior → no rim-kill")
    print("=" * 60)
    p, f = test_prey_interior_no_rim_kill()
    all_results.append(("1B  interior cells are safe", p, len(f)))

    print("\n" + "=" * 60)
    print(" TEST 1C — RIM is exactly 1 cell wide")
    print("=" * 60)
    p, f = test_rim_width_is_one_cell()
    all_results.append(("1C  rim width = 1 cell", p, len(f)))

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
        print("\n  ⚠  SOME TESTS FAILED — kill-zone has bugs!\n")
        sys.exit(1)
    else:
        print("\n  ✅  ALL KILL-ZONE TESTS PASSED\n")


if __name__ == "__main__":
    main()
