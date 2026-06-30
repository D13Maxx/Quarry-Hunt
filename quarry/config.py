from dataclasses import dataclass


@dataclass
class Config:
    # --- environment ---
    grid_size: int = 41
    num_hunters: int = 3
    hunter_vision: int = 5
    prey_vision: int = 13
    max_steps: int = 350
    cell_px: int = 24
    gif_fps: int = 10

    # --- magic zones (hunter-blinding 3×3 patches) ---
    magic_zones_min: int = 1
    magic_zones_max: int = 5

    # --- hunter search behaviour (heuristic baseline) ---
    hunter_persistence: float = 0.8
    hunter_memory_steps: int = 12

    # --- predictor ---
    window_K: int = 15  # provisional — sweep later
    history_len: int = 8
    gru_hidden: int = 128
    encoder_channels: tuple[int, int] = (32, 64)
    encoder_out_dim: int = 64
    predictor_lr: float = 1e-3

    # --- policy / MAPPO ---
    actor_hidden: int = 128
    critic_hidden: int = 256
    ppo_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    rollout_episodes: int = 8

    # --- reward shaping ---
    capture_reward: float = 10.0
    rim_drive_reward: float = 0.0       # zeroed for objective-only reward; restore to -1.0 for shaping
    step_penalty: float = -0.01
    closing_reward_scale: float = 0.0   # zeroed for objective-only reward; restore to 0.1 for shaping
    sighting_reward: float = 0.02       # per-step TEAM bonus if any hunter perceives the prey

    # --- training ---
    batch_size: int = 64
    num_collect_episodes: int = 1000
    train_val_split: float = 0.8
    num_iterations: int = 1000
    eval_every: int = 50
    ckpt_dir: str = "checkpoints"
    predictor_ckpt: str = "data/predictor.pt"
