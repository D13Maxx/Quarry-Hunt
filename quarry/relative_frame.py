import numpy as np


def to_relative(hunter_pos: tuple[int, int], prey_pos: tuple[int, int], K: int) -> tuple[int, int, bool]:
    """Returns (row, col) in the K×K window and whether the target was out of range."""
    half = K // 2
    dr = prey_pos[0] - hunter_pos[0]
    dc = prey_pos[1] - hunter_pos[1]
    r = np.clip(dr + half, 0, K - 1)
    c = np.clip(dc + half, 0, K - 1)
    oof = not (0 <= dr + half < K and 0 <= dc + half < K)
    return int(r), int(c), oof


def heatmap_to_world(hunter_pos: tuple[int, int], heatmap: np.ndarray) -> list[tuple[int, int, float]]:
    """Convert a K×K heatmap back to (board_row, board_col, prob) triples."""
    K = heatmap.shape[0]
    half = K // 2
    hr, hc = hunter_pos
    cells = []
    for r in range(K):
        for c in range(K):
            p = float(heatmap[r, c])
            if p > 1e-4:
                cells.append((hr + r - half, hc + c - half, p))
    return cells
