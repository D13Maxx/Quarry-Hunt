import numpy as np

FOREST = 1
RIM = 2

DELTAS = (
    (0, 0),   # stay
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1),   # right
    (-1, -1), # up-left
    (-1, 1),  # up-right
    (1, -1),  # down-left
    (1, 1),   # down-right
)
NUM_ACTIONS = len(DELTAS)

NUM_CHANNELS = 6
CH_FOREST, CH_RIM, CH_SELF, CH_HUNTERS, CH_PREY, CH_MAGIC = range(6)


def create_grid(size: int) -> np.ndarray:
    grid = np.full((size, size), FOREST, dtype=np.int8)
    grid[0, :] = RIM
    grid[-1, :] = RIM
    grid[:, 0] = RIM
    grid[:, -1] = RIM
    return grid


def spawn_hunters(size: int) -> list[tuple[int, int]]:
    """Three middle cells of the south rim row."""
    mid = size // 2
    row = size - 1
    return [(row, mid - 1), (row, mid), (row, mid + 1)]


def spawn_prey(size: int, rng: np.random.Generator) -> tuple[int, int]:
    r = rng.integers(1, size - 1)
    c = rng.integers(1, size - 1)
    return (int(r), int(c))


def apply_move(pos: tuple[int, int], action: int, size: int) -> tuple[int, int]:
    dr, dc = DELTAS[action]
    return (
        max(0, min(size - 1, pos[0] + dr)),
        max(0, min(size - 1, pos[1] + dc)),
    )


def check_win(grid, hunter_pos, prey_pos, step, max_steps) -> str | None:
    if grid[prey_pos] == RIM:
        return "hunters"
    pr, pc = prey_pos
    for hr, hc in hunter_pos:
        if max(abs(hr - pr), abs(hc - pc)) <= 1:
            return "hunters"
    if step >= max_steps:
        return "prey"
    return None


def place_magic_zones(
    size: int, rng: np.random.Generator,
    n_min: int, n_max: int,
    exclude_cells: set[tuple[int, int]],
) -> np.ndarray:
    n = int(rng.integers(n_min, n_max + 1))
    mask = np.zeros((size, size), dtype=bool)
    for _ in range(n):
        placed = False
        for _ in range(20):
            r0 = int(rng.integers(1, size - 3))
            c0 = int(rng.integers(1, size - 3))
            zone = {(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)}
            if zone & exclude_cells:
                continue
            mask[r0:r0 + 3, c0:c0 + 3] = True
            placed = True
            break
    return mask


def ego_obs(grid, pos, vision, hunter_pos, prey_pos, agent_idx, is_prey,
            magic_mask=None) -> np.ndarray:
    size = grid.shape[0]
    half = vision // 2
    obs = np.zeros((NUM_CHANNELS, vision, vision), dtype=np.float32)

    r, c = pos
    g_r0, g_r1 = max(0, r - half), min(size, r + half + 1)
    g_c0, g_c1 = max(0, c - half), min(size, c + half + 1)
    o_r0 = g_r0 - (r - half)
    o_c0 = g_c0 - (c - half)
    patch = grid[g_r0:g_r1, g_c0:g_c1]

    obs[CH_FOREST, o_r0:o_r0 + patch.shape[0], o_c0:o_c0 + patch.shape[1]] = (patch == FOREST)
    obs[CH_RIM, o_r0:o_r0 + patch.shape[0], o_c0:o_c0 + patch.shape[1]] = (patch == RIM)
    obs[CH_SELF, half, half] = 1

    # magic channel — visible to everyone
    if magic_mask is not None:
        mpatch = magic_mask[g_r0:g_r1, g_c0:g_c1]
        obs[CH_MAGIC, o_r0:o_r0 + mpatch.shape[0], o_c0:o_c0 + mpatch.shape[1]] = mpatch

    # Effect B: hunter standing on a magic cell sees no agents at all
    hunter_blinded = (not is_prey) and magic_mask is not None and magic_mask[r, c]
    if hunter_blinded:
        return obs

    others = hunter_pos if is_prey else [hp for i, hp in enumerate(hunter_pos) if i != agent_idx]
    for hr, hc in others:
        # Effect A: agents inside a magic cell are invisible to hunters
        if (not is_prey) and magic_mask is not None and magic_mask[hr, hc]:
            continue
        vr, vc = hr - r + half, hc - c + half
        if 0 <= vr < vision and 0 <= vc < vision:
            obs[CH_HUNTERS, vr, vc] = 1

    pr, pc = prey_pos
    # Effect A for prey target
    if (not is_prey) and magic_mask is not None and magic_mask[pr, pc]:
        pass
    else:
        vr, vc = pr - r + half, pc - c + half
        if 0 <= vr < vision and 0 <= vc < vision:
            obs[CH_PREY, vr, vc] = 1

    return obs

