import numpy as np
from dataclasses import dataclass, field

from quarry.config import Config
from quarry.world import DELTAS, CH_HUNTERS, CH_PREY, CH_RIM


@dataclass
class HunterSearchState:
    """Per-hunter state for the search/persistence behaviour.

    Fields
    ------
    prev_action : int
        The action index this hunter chose last step (0 = stay at reset).
    last_seen_prey : tuple[int, int] | None
        Board (row, col) where this hunter *personally* last saw the prey,
        or None if never seen.
    steps_since_seen : int
        Steps elapsed since this hunter last saw the prey.
        Starts at a large sentinel so the "memory expired" check is true
        until the hunter actually spots the prey for the first time.
    """
    prev_action: int = 0
    last_seen_prey: tuple[int, int] | None = None
    steps_since_seen: int = 999


def make_hunter_states(num_hunters: int) -> list[HunterSearchState]:
    """Create a fresh list of per-hunter search states (one per hunter)."""
    return [HunterSearchState() for _ in range(num_hunters)]


def prey_act(obs: np.ndarray) -> int:
    """Flee nearest visible hunter; stay if none visible."""
    hunters = np.argwhere(obs[CH_HUNTERS])
    if len(hunters) == 0:
        return 0

    center = obs.shape[1] // 2
    nearest = hunters[np.abs(hunters - center).max(axis=1).argmin()]
    dr = int(np.sign(center - nearest[0]))
    dc = int(np.sign(center - nearest[1]))

    for action in _ranked_actions(dr, dc):
        adr, adc = DELTAS[action]
        tr, tc = center + adr, center + adc
        if 0 <= tr < obs.shape[1] and 0 <= tc < obs.shape[2] and obs[CH_RIM, tr, tc] == 0:
            return action
    return 0


def _ranked_actions(dr: int, dc: int) -> list[int]:
    """Actions sorted by alignment with (dr, dc), stay last."""
    moves = [(DELTAS[i][0] * dr + DELTAS[i][1] * dc, i) for i in range(1, len(DELTAS))]
    moves.sort(key=lambda x: -x[0])
    return [a for _, a in moves] + [0]


def hunter_act(
    obs: np.ndarray,
    rng: np.random.Generator,
    hunter_pos: tuple[int, int] | None = None,
    state: HunterSearchState | None = None,
    cfg: Config | None = None,
) -> int:
    """Decide one hunter's action and update its search state.

    Priority ladder
    ---------------
    1. Prey visible in the 5×5  → step toward it (unchanged State 2);
       record prey's board cell as last-seen with age 0.
    2. Memory alive (age < cfg.hunter_memory_steps) → step toward remembered
       cell; increment age.
    3. Persistent walk: with P = cfg.hunter_persistence repeat previous
       direction, otherwise uniform random.  Re-roll if the chosen direction
       would leave the grid.

    After every step the chosen action is stored as prev_action.
    """
    center = obs.shape[1] // 2
    prey = np.argwhere(obs[CH_PREY])

    # ------------------------------------------------------------------
    # Priority 1 — prey visible (existing State 2, unchanged logic)
    # ------------------------------------------------------------------
    if len(prey) > 0:
        target = prey[0]
        dr = int(np.sign(target[0] - center))
        dc = int(np.sign(target[1] - center))
        action = _direction_to_action(dr, dc)

        # Update search state: record prey's board position
        if state is not None and hunter_pos is not None:
            prey_board_r = hunter_pos[0] + (target[0] - center)
            prey_board_c = hunter_pos[1] + (target[1] - center)
            state.last_seen_prey = (prey_board_r, prey_board_c)
            state.steps_since_seen = 0
            state.prev_action = action
        return action

    # ------------------------------------------------------------------
    # Fallback: legacy behaviour when no state/config is provided
    # (keeps old call sites and tests working without changes)
    # ------------------------------------------------------------------
    if state is None or cfg is None or hunter_pos is None:
        action = int(rng.integers(0, len(DELTAS)))
        return action

    # ------------------------------------------------------------------
    # Priority 2 — memory alive → head toward remembered cell
    # ------------------------------------------------------------------
    if state.last_seen_prey is not None and state.steps_since_seen < cfg.hunter_memory_steps:
        tr, tc = state.last_seen_prey
        dr = int(np.sign(tr - hunter_pos[0]))
        dc = int(np.sign(tc - hunter_pos[1]))
        action = _direction_to_action(dr, dc)
        state.steps_since_seen += 1
        state.prev_action = action
        return action

    # ------------------------------------------------------------------
    # Priority 3 — persistent walk with boundary bounce
    # ------------------------------------------------------------------
    grid_size = cfg.grid_size
    if state.prev_action != 0 and rng.random() < cfg.hunter_persistence:
        action = state.prev_action
    else:
        action = int(rng.integers(1, len(DELTAS)))  # exclude stay (0)

    # If the chosen direction would leave the grid, re-roll
    if _would_leave_grid(hunter_pos, action, grid_size):
        action = _pick_inbounds_direction(hunter_pos, grid_size, rng)

    state.prev_action = action
    return action


def _direction_to_action(dr: int, dc: int) -> int:
    """Map a (dr, dc) sign vector to the matching DELTAS index."""
    for i, (adr, adc) in enumerate(DELTAS):
        if adr == dr and adc == dc:
            return i
    return 0  # fallback: stay


def _would_leave_grid(pos: tuple[int, int], action: int, grid_size: int) -> bool:
    """True if applying *action* at *pos* clips against a boundary."""
    dr, dc = DELTAS[action]
    nr, nc = pos[0] + dr, pos[1] + dc
    return not (0 <= nr < grid_size and 0 <= nc < grid_size)


def _pick_inbounds_direction(
    pos: tuple[int, int], grid_size: int, rng: np.random.Generator,
) -> int:
    """Uniformly pick a movement action (1-8) that stays inside the grid."""
    valid = [
        i for i in range(1, len(DELTAS))
        if not _would_leave_grid(pos, i, grid_size)
    ]
    if not valid:
        return 0
    return int(rng.choice(valid))

