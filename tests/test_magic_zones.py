import numpy as np
import pytest

from quarry.config import Config
from quarry.env import QuarryEnv
from quarry.world import (
    NUM_CHANNELS, CH_FOREST, CH_RIM, CH_SELF, CH_HUNTERS, CH_PREY, CH_MAGIC,
    create_grid, place_magic_zones, ego_obs, spawn_hunters,
)


def test_config_fields():
    cfg = Config()
    assert cfg.magic_zones_min == 1
    assert cfg.magic_zones_max == 5


def test_place_magic_zones_properties():
    rng = np.random.default_rng(42)
    size = 41
    hunters = spawn_hunters(size)
    exclude = set(hunters)
    mask = place_magic_zones(size, rng, 1, 5, exclude)
    assert mask.shape == (size, size)
    assert mask.dtype == bool

    # no True on the rim
    assert not mask[0, :].any()
    assert not mask[-1, :].any()
    assert not mask[:, 0].any()
    assert not mask[:, -1].any()

    # none on hunter spawn cells
    for hr, hc in hunters:
        assert not mask[hr, hc]

    # True-cell count between 9 and 45 (1..5 non-overlapping zones)
    n_true = int(mask.sum())
    assert 9 <= n_true <= 45


def test_effect_a_occlusion():
    """Prey inside magic cell is invisible to hunters; stepping out restores visibility."""
    cfg = Config(grid_size=25, hunter_vision=11)
    grid = create_grid(25)
    mask = np.zeros((25, 25), dtype=bool)
    magic_r, magic_c = 10, 10
    mask[magic_r:magic_r + 3, magic_c:magic_c + 3] = True

    hunter_pos = [(10, 5)]
    prey_in_zone = (11, 11)  # inside magic zone
    obs = ego_obs(grid, hunter_pos[0], 11, hunter_pos, prey_in_zone, 0, False, magic_mask=mask)
    assert obs[CH_PREY].sum() == 0, "prey inside magic zone should be invisible to hunter"

    prey_outside = (11, 9)  # outside zone (zone covers cols 10-12), within view
    obs2 = ego_obs(grid, hunter_pos[0], 11, hunter_pos, prey_outside, 0, False, magic_mask=mask)
    assert obs2[CH_PREY].sum() == 1, "prey outside zone should be visible"


def test_effect_b_blinded_hunter():
    """Hunter standing on a magic cell sees no agents, but CH_SELF and CH_MAGIC are populated."""
    cfg = Config(grid_size=25, hunter_vision=11)
    grid = create_grid(25)
    mask = np.zeros((25, 25), dtype=bool)
    mask[10:13, 10:13] = True

    hunter_pos = [(11, 11)]  # on magic cell
    prey_pos = (11, 12)  # adjacent
    obs = ego_obs(grid, hunter_pos[0], 11, hunter_pos, prey_pos, 0, False, magic_mask=mask)
    assert obs[CH_PREY].sum() == 0, "blinded hunter should not see prey"
    assert obs[CH_HUNTERS].sum() == 0, "blinded hunter should not see other hunters"
    assert obs[CH_SELF, 5, 5] == 1, "CH_SELF should still be set"
    assert obs[CH_MAGIC].sum() > 0, "CH_MAGIC should still be populated"


def test_prey_exemption():
    """Prey can see hunters standing inside magic zones."""
    cfg = Config(grid_size=25, prey_vision=13)
    grid = create_grid(25)
    mask = np.zeros((25, 25), dtype=bool)
    mask[10:13, 10:13] = True

    hunter_in_zone = [(11, 11)]
    prey_pos = (11, 6)  # in view range
    obs = ego_obs(grid, prey_pos, 13, hunter_in_zone, prey_pos, -1, True, magic_mask=mask)
    half = 13 // 2
    expected_vr = 11 - 11 + half
    expected_vc = 11 - 6 + half
    assert obs[CH_HUNTERS, expected_vr, expected_vc] == 1, "prey should see hunter in magic zone"


def test_shape_guard_full_episode():
    """Full episode with zones active — obs shape stays (6, vision, vision), no crash."""
    cfg = Config(grid_size=25, max_steps=50)
    env = QuarryEnv(cfg)
    obs, _ = env.reset(seed=123)
    rng = np.random.default_rng(999)

    for agent, o in obs.items():
        v = cfg.prey_vision if agent == "prey" else cfg.hunter_vision
        assert o.shape == (NUM_CHANNELS, v, v)
        assert not np.isnan(o).any()

    done = False
    while not done:
        actions = {a: int(rng.integers(0, 9)) for a in env.agents}
        obs, _, terms, _, _ = env.step(actions)
        done = terms.get("prey", False)
        if not done:
            for agent, o in obs.items():
                v = cfg.prey_vision if agent == "prey" else cfg.hunter_vision
                assert o.shape == (NUM_CHANNELS, v, v), f"shape mismatch for {agent}"
                assert not np.isnan(o).any(), f"NaN in obs for {agent}"
