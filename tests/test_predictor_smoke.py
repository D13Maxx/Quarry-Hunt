import torch
from quarry.config import Config
from quarry.predictor import PreyPredictor


def test_forward_shape_and_no_nan():
    cfg = Config()
    model = PreyPredictor(cfg)
    x = torch.randn(4, cfg.history_len, 6, cfg.hunter_vision, cfg.hunter_vision)
    out = model(x)
    assert out.shape == (4, cfg.window_K, cfg.window_K), f"got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in forward output"


def test_step_threading():
    cfg = Config()
    model = PreyPredictor(cfg)
    h = model.init_hidden()
    obs = torch.randn(6, cfg.hunter_vision, cfg.hunter_vision)

    logits1, h1 = model.step(obs, h)
    assert logits1.shape == (cfg.window_K, cfg.window_K)
    assert h1.shape == h.shape

    logits2, h2 = model.step(obs, h1)
    assert logits2.shape == (cfg.window_K, cfg.window_K)
    assert not torch.equal(h1, h2), "hidden should change between steps"


def test_grid_size_agnostic():
    cfg_small = Config(grid_size=15)
    cfg_large = Config(grid_size=100)
    m1 = PreyPredictor(cfg_small)
    m2 = PreyPredictor(cfg_large)
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        assert p1.shape == p2.shape, f"{n1}: {p1.shape} vs {p2.shape}"
