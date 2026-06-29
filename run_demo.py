import argparse
import pathlib

import numpy as np
# pyrefly: ignore [missing-import]
import pygame
import torch
import torch.nn.functional as F

from quarry.config import Config
from quarry.env import QuarryEnv
from quarry.agents import prey_act, hunter_act
from quarry.actor import Actor
from quarry.renderer import Renderer
from quarry.predictor import PreyPredictor
from quarry.relative_frame import heatmap_to_world
from quarry.world import CH_PREY


def load_joint(ckpt_path: str, cfg: Config, device: torch.device):
    path = pathlib.Path(ckpt_path)
    if not path.exists():
        return None, None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor = Actor(cfg).to(device)
    predictor = PreyPredictor(cfg).to(device)
    try:
        actor.load_state_dict(ckpt["actor"])
        predictor.load_state_dict(ckpt["predictor"])
    except (RuntimeError, KeyError) as e:
        print(f"WARNING: joint checkpoint incompatible: {e}")
        return None, None
    actor.eval()
    predictor.eval()
    return actor, predictor


def load_predictor_only(cfg: Config, device: torch.device):
    ckpt = pathlib.Path(cfg.predictor_ckpt)
    if not ckpt.exists():
        return None
    model = PreyPredictor(cfg)
    try:
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    except RuntimeError:
        print(f"WARNING: predictor checkpoint incompatible (K={cfg.window_K}), skipping")
        return None
    model.eval()
    model.to(device)
    return model


def draw_start_button(renderer) -> pygame.Rect:
    btn_w, btn_h = 120, 40
    x = renderer.grid_px + (renderer.sidebar - btn_w) // 2
    y = renderer.h - 60
    rect = pygame.Rect(x, y, btn_w, btn_h)
    pygame.draw.rect(renderer.surface, (40, 120, 40), rect, border_radius=6)
    label = renderer.font.render("START", True, (255, 255, 255))
    renderer.surface.blit(label, (rect.centerx - label.get_width() // 2,
                                  rect.centery - label.get_height() // 2))
    return rect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", type=str, default=None, help="Path to joint checkpoint")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gif", type=str, default="episode.gif")
    args = parser.parse_args()

    cfg = Config()
    env = QuarryEnv(cfg)
    renderer = Renderer(cfg, headless=args.headless)
    rng = np.random.default_rng()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rl_actor, predictor = None, None
    if args.joint:
        rl_actor, predictor = load_joint(args.joint, cfg, device)
        if rl_actor:
            print(f"Loaded joint checkpoint: {args.joint}")
    if predictor is None:
        predictor = load_predictor_only(cfg, device)

    has_predictor = predictor is not None
    use_rl = rl_actor is not None
    K = cfg.window_K

    while True:
        obs, _ = env.reset()
        renderer.draw(env.grid, env.hunter_pos, env.prey_pos, 0, cfg.max_steps, None,
                      pred_info=[] if has_predictor else None,
                      magic_mask=env.magic_mask)
        if not args.headless:
            btn = draw_start_button(renderer)
            pygame.display.flip()

        hiddens = [predictor.init_hidden(device) if has_predictor else None
                   for _ in range(cfg.num_hunters)]

        if not args.headless:
            started = False
            while not started:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        renderer.close()
                        return
                    if event.type == pygame.MOUSEBUTTONDOWN and btn.collidepoint(event.pos):
                        started = True

        renderer.frames.clear()
        renderer.capture_frame()

        while env.agents:
            if not args.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        renderer.close()
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                        renderer.cycle_ghost_mode()

            ghost_cells = None
            pred_info = None
            actions = {}

            if has_predictor:
                ghost_cells, pred_info = [], []
                ghost_tensors = []
                with torch.no_grad():
                    for i in range(cfg.num_hunters):
                        obs_t = torch.from_numpy(obs[f"hunter_{i}"]).float().to(device)
                        logits, hiddens[i] = predictor.step(obs_t, hiddens[i])
                        probs = F.softmax(logits.flatten(), dim=-1).reshape(K, K)
                        probs_np = probs.cpu().numpy()

                        ghost_cells.append(heatmap_to_world(env.hunter_pos[i], probs_np))
                        ghost_tensors.append(probs.detach())

                        peak = probs_np.argmax()
                        pr, pc = divmod(int(peak), K)
                        sees = obs[f"hunter_{i}"][CH_PREY].any()
                        pred_info.append({"r": pr, "c": pc, "p": float(probs_np.max()),
                                          "sees": bool(sees)})

                        if use_rl:
                            action, _, _ = rl_actor.act(obs_t.unsqueeze(0),
                                                        ghost_tensors[i].unsqueeze(0))
                            actions[f"hunter_{i}"] = int(action.item())

            if not use_rl:
                for i in range(cfg.num_hunters):
                    actions[f"hunter_{i}"] = hunter_act(
                        obs[f"hunter_{i}"], rng,
                        hunter_pos=env.hunter_pos[i],
                        state=env.hunter_states[i],
                        cfg=cfg,
                    )
            actions["prey"] = prey_act(obs["prey"])

            obs, _, terms, _, _ = env.step(actions)

            renderer.draw(env.grid, env.hunter_pos, env.prey_pos,
                          env.step_count, cfg.max_steps, env.winner,
                          ghost_cells=ghost_cells, pred_info=pred_info,
                          magic_mask=env.magic_mask)
            renderer.capture_frame()
            if not args.headless:
                renderer.clock.tick(cfg.gif_fps)

        print(f"{env.winner} win at step {env.step_count}")
        renderer.save_gif(args.gif)
        print(f"Saved {args.gif}")

        if args.headless:
            break


if __name__ == "__main__":
    main()
