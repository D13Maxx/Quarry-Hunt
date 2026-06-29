# pyrefly: ignore [missing-import]
import pygame
from PIL import Image

from quarry.config import Config
from quarry.world import FOREST, RIM

COLORS = {
    FOREST: (26, 71, 42),
    RIM: (139, 26, 26),
    "hunter": (65, 105, 225),
    "prey": (255, 140, 0),
    "grid": (40, 40, 40),
    "hud_bg": (30, 30, 30),
    "text": (220, 220, 220),
    "magic": (160, 50, 220),
}

# per-hunter ghost colors (RGB)
GHOST_COLORS = [
    (100, 149, 237),  # H0 cornflower blue
    (50, 205, 50),    # H1 lime green
    (178, 102, 255),  # H2 purple
]


class Renderer:
    def __init__(self, cfg: Config, headless=False):
        self.cfg = cfg
        self.headless = headless
        self.grid_px = cfg.grid_size * cfg.cell_px
        self.sidebar = 200
        self.w = self.grid_px + self.sidebar
        self.h = self.grid_px
        self.ghost_mode = 0  # 0=all, 1/2/3=single hunter

        pygame.init()
        if headless:
            self.surface = pygame.Surface((self.w, self.h))
        else:
            self.surface = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Quarry")

        self.font = pygame.font.SysFont("consolas", 14)
        self.frames: list[Image.Image] = []
        self.clock = pygame.time.Clock()

    def cycle_ghost_mode(self):
        self.ghost_mode = (self.ghost_mode + 1) % (self.cfg.num_hunters + 1)

    def draw(self, grid, hunter_pos, prey_pos, step, max_steps, winner,
             ghost_cells=None, pred_info=None, magic_mask=None):
        cpx = self.cfg.cell_px
        self.surface.fill(COLORS["hud_bg"])

        for r in range(self.cfg.grid_size):
            for c in range(self.cfg.grid_size):
                x, y = c * cpx, r * cpx
                color = COLORS.get(grid[r, c], COLORS["hud_bg"])
                pygame.draw.rect(self.surface, color, (x, y, cpx, cpx))
                pygame.draw.rect(self.surface, COLORS["grid"], (x, y, cpx, cpx), 1)

        if magic_mask is not None:
            self._draw_magic_zones(magic_mask)

        if ghost_cells:
            self._draw_ghosts(ghost_cells)

        radius = cpx // 3
        for i, (hr, hc) in enumerate(hunter_pos):
            cx, cy = hc * cpx + cpx // 2, hr * cpx + cpx // 2
            blinded = magic_mask is not None and magic_mask[hr, hc]
            color = (100, 100, 100) if blinded else COLORS["hunter"]
            pygame.draw.circle(self.surface, color, (cx, cy), radius)
            label = self.font.render(str(i), True, (255, 255, 255))
            self.surface.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))
            if blinded:
                # red X over blinded hunter
                d = radius
                pygame.draw.line(self.surface, (255, 60, 60), (cx - d, cy - d), (cx + d, cy + d), 2)
                pygame.draw.line(self.surface, (255, 60, 60), (cx - d, cy + d), (cx + d, cy - d), 2)

        pr, pc = prey_pos
        cx, cy = pc * cpx + cpx // 2, pr * cpx + cpx // 2
        # highlight prey if it's inside a magic zone (invisible to hunters)
        prey_hidden = magic_mask is not None and magic_mask[pr, pc]
        prey_color = (255, 200, 80) if prey_hidden else COLORS["prey"]
        pygame.draw.circle(self.surface, prey_color, (cx, cy), radius)
        if prey_hidden:
            pygame.draw.circle(self.surface, COLORS["magic"], (cx, cy), radius, 2)

        self._draw_vision(hunter_pos, self.cfg.hunter_vision, (65, 105, 225, 80), (65, 105, 225))
        self._draw_vision([prey_pos], self.cfg.prey_vision, (255, 140, 0, 80), (255, 140, 0))
        self._draw_hud(hunter_pos, prey_pos, step, max_steps, winner, pred_info, magic_mask)

        if not self.headless:
            pygame.display.flip()

    def _draw_ghosts(self, ghost_cells):
        """ghost_cells: list of 3 lists of (board_row, board_col, prob)."""
        cpx = self.cfg.cell_px
        sz = self.cfg.grid_size
        for i, cells in enumerate(ghost_cells):
            if self.ghost_mode != 0 and self.ghost_mode != i + 1:
                continue
            if not cells:
                continue
            rgb = GHOST_COLORS[i % len(GHOST_COLORS)]
            max_p = max(p for _, _, p in cells)
            if max_p < 1e-6:
                continue
            for br, bc, prob in cells:
                if not (0 <= br < sz and 0 <= bc < sz):
                    continue
                alpha = min(int((prob / max_p) * 180), 180)
                if alpha < 5:
                    continue
                overlay = pygame.Surface((cpx, cpx), pygame.SRCALPHA)
                overlay.fill((*rgb, alpha))
                self.surface.blit(overlay, (bc * cpx, br * cpx))

    def _draw_magic_zones(self, magic_mask):
        cpx = self.cfg.cell_px
        overlay = pygame.Surface((cpx, cpx), pygame.SRCALPHA)
        overlay.fill((*COLORS["magic"], 70))
        for r in range(self.cfg.grid_size):
            for c in range(self.cfg.grid_size):
                if magic_mask[r, c]:
                    self.surface.blit(overlay, (c * cpx, r * cpx))

    def _draw_vision(self, positions, vision, rgba, border_rgb):
        cpx = self.cfg.cell_px
        half = vision // 2
        size = self.cfg.grid_size
        for r, c in positions:
            r0 = max(0, r - half)
            c0 = max(0, c - half)
            r1 = min(size, r + half + 1)
            c1 = min(size, c + half + 1)
            px_x, px_y = c0 * cpx, r0 * cpx
            pw, ph = (c1 - c0) * cpx, (r1 - r0) * cpx
            overlay = pygame.Surface((pw, ph), pygame.SRCALPHA)
            overlay.fill(rgba)
            self.surface.blit(overlay, (px_x, px_y))
            pygame.draw.rect(self.surface, border_rgb, (px_x, px_y, pw, ph), 2)

    def _draw_hud(self, hunter_pos, prey_pos, step, max_steps, winner, pred_info=None,
                  magic_mask=None):
        sx = self.grid_px + 10
        pr, pc = prey_pos
        lines = [
            f"Step: {step}/{max_steps}",
            f"Status: {winner.upper() + ' WIN' if winner else 'RUNNING'}",
            "",
        ]

        if magic_mask is not None:
            n_zones = int(magic_mask.sum())
            prey_hidden = bool(magic_mask[pr, pc])
            lines.append(f"Magic cells: {n_zones}")
            if prey_hidden:
                lines.append("Prey: HIDDEN")
            lines.append("")

        for i, (hr, hc) in enumerate(hunter_pos):
            d = max(abs(hr - pr), abs(hc - pc))
            line = f"H{i} dist:{d}"
            if magic_mask is not None and magic_mask[hr, hc]:
                line += " [BLIND]"
            if pred_info and i < len(pred_info):
                pi = pred_info[i]
                line += f" pred({pi['r']},{pi['c']}) p={pi['p']:.2f}"
                line += " [see]" if pi["sees"] else " [blind]"
            lines.append(line)

        if pred_info:
            lines.append("")
            mode_label = "all" if self.ghost_mode == 0 else f"H{self.ghost_mode - 1}"
            lines.append(f"Ghost: {mode_label} [G]")
        elif pred_info is None:
            lines.append("")
            lines.append("no model loaded")

        for i, line in enumerate(lines):
            if line:
                surf = self.font.render(line, True, COLORS["text"])
                self.surface.blit(surf, (sx, 10 + i * 22))

    def capture_frame(self):
        raw = pygame.image.tobytes(self.surface, "RGB")
        self.frames.append(Image.frombytes("RGB", (self.w, self.h), raw))

    def save_gif(self, path: str):
        if not self.frames:
            return
        self.frames[0].save(
            path, save_all=True, append_images=self.frames[1:],
            duration=1000 // self.cfg.gif_fps, loop=0,
        )

    def close(self):
        pygame.quit()
