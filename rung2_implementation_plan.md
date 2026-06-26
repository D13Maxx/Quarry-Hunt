# Rung 2: World Model — Implementation Plan

Add the per-hunter prediction net and the ghost overlay. The net learns by **supervised self-play**: agents play, the net predicts the prey's **next-step** location from one hunter's recent fogged trail, the simulator hands back the true next cell as a free label. Output is a **hunter-centric K×K probability heatmap** in relative coordinates. Control stays heuristic — the only new things this rung are the predictor and the overlay lighting up. No RL, no trained hunter policy (that is Rung 3).

Builds directly on the Rung 1 `quarry/` package.

---

## Locked Decisions (from interview)

| Decision | Answer |
|---|---|
| Learning paradigm | Supervised, from self-play data (not a canned dataset) |
| What it predicts | The prey's **next-step** position |
| Output form | Probability **heatmap** (softmax over a K×K window) |
| Coordinate frame | **Hunter-centric / relative** (offsets from the hunter), never absolute board cells |
| Window size `K` | **Variable — decided later.** Lives in config; choose from the out-of-range rate seen during data collection |
| Memory | **GRU** (LSTM is a one-line swap if memory ever proves the bottleneck) |
| History length | ~8 recent steps |
| Net size | Medium (starting point) |
| Encoder | Small CNN over the `(6,5,5)` patch (default) |
| Loss | Cross-entropy vs. the true next cell |
| Per-hunter | Each hunter has its **own** net and its **own** ghost (3 independent nets) |
| Control | Heuristic, unchanged from Rung 1 |
| Prey | **Upgraded to the scripted *fleeing* prey** (flee nearest visible hunter + rim avoidance). The frozen Rung-1 placeholder produces no motion to learn from. |
| Data split | By **episode**, never by timestep |
| Overlay | Render each hunter's predicted heatmap on the board + the true prey |
| Magic zones | Still count = 0 |
| Field | Still 25×25 |

---

## Why three phases
Rung 2 is not one program — it is three, run in order:

1. **Data collection** — play self-play episodes, log `(observation history, true next-step prey cell in the hunter's relative frame)` for every hunter at every step, write to disk.
2. **Offline training** — train the predictor on the logged data (Kaggle T4), cross-entropy, episode-split validation, until val flattens.
3. **Integration + overlay** — load the trained net into the live sim, run it per hunter each tick, render the ghost.

Keeping them separate is the point: data and training are reproducible offline; only phase 3 touches the live loop.

---

## Proposed Changes

### Project Structure (additions to Rung 1)
```
Chase/
├── quarry/
│   ├── config.py              [MODIFY] add predictor + training params
│   ├── agents.py              [MODIFY] upgrade ScriptedPrey to fleeing
│   ├── renderer.py            [MODIFY] fill the reserved ghost-overlay slot
│   ├── predictor.py           [NEW] the net: encoder + GRU + heatmap head
│   ├── relative_frame.py      [NEW] absolute <-> hunter-relative conversions
│   ├── data_collection.py     [NEW] self-play logger -> dataset on disk
│   └── train_predictor.py     [NEW] offline training loop + checkpoints
├── run_demo.py                [MODIFY] load net, predict per step, feed overlay
├── train_on_kaggle.ipynb      [NEW] thin notebook wrapping train_predictor
└── requirements.txt           [MODIFY] add torch
```

---

### [MODIFY] `config.py`
Add to `QuarryConfig` (all tunable; `window_K` deliberately a variable):
- `window_K: int = 11` — placeholder; **the side length of the relative belief window, to be finalized later**. Centered on the hunter.
- `history_len: int = 8`
- `gru_hidden: int = 128` (medium)
- `encoder_out_dim: int = 64`
- `predict_horizon: str = "next"` (next-step)
- `lr: float = 1e-3`
- `batch_size: int = 64`
- `num_collect_episodes: int = 1000` (tune)
- `train_val_split: float = 0.8` (by episode)
- `predictor_ckpt: str = "predictor.pt"`

### [NEW] `predictor.py`
`PreyPredictor(nn.Module)`:
- **encoder** — small CNN over `(6, 5, 5)` → feature vector of `encoder_out_dim` per frame. Preserves the spatial cue (a flicker upper-right) that a flattened MLP would throw away.
- **memory** — `nn.GRU(input_size=encoder_out_dim, hidden_size=gru_hidden, batch_first=True)`. Consumes the sequence of per-frame features.
- **head** — `nn.Linear(gru_hidden, K*K)` → reshape to `(K, K)` logits. Softmax at inference; cross-entropy against the target cell at train.
- `forward(obs_seq)` → `(K, K)` logits. Provide a **stateful single-step** path for live inference (carry the GRU hidden state across ticks rather than re-feeding the whole history every frame).
- **Size-agnostic by construction:** input is the fixed 5×5 patch, output is the relative K×K window — neither depends on grid size, so weights survive grid-growth at Rung 4.

### [NEW] `relative_frame.py`
- `to_relative_target(hunter_pos, prey_next_pos, K)` → `(row, col)` index inside the K×K window, **or** an out-of-range flag. v1 policy: **clamp** an out-of-range prey to the nearest window-border cell, and log the out-of-range rate (this rate is how `K` gets chosen — pick `K` so the prey's next cell lands in-window the large majority of the time).
- `heatmap_to_world(hunter_pos, heatmap, K)` → list of `(board_cell, prob)` for rendering the ghost back in world space.

### [MODIFY] `agents.py`
- Upgrade `ScriptedPrey` from the frozen Rung-1 version to the **design's fleeing behavior**: if any hunter is visible in the 13×13 view, move away from the nearest, filtered to avoid the rim; if every flee move is rim-ward (cornered), take the best non-rim move, else `STAY`; if no hunter is visible, behave per the locked design (stay). Fleeing episodes supply the motion the predictor learns from.
- `HeuristicHunter` unchanged.

### [NEW] `data_collection.py`
- Loop `num_collect_episodes`: `reset()`, run to termination. At each step, for each hunter: append its current observation to that hunter's rolling history, and record the prey's **next-step** position converted to that hunter's **relative** frame via `to_relative_target`.
- Persist examples grouped by **episode id** (so the split can be by episode). One training example = one hunter's `(history sequence, target cell)` at one step.
- Track and report: fraction of targets that were out-of-range (informs `K`), and fraction of targets equal to "same cell" (sanity that the data has motion — should not be ~100%).

### [NEW] `train_predictor.py`
- Load dataset; **split by episode** (`train_val_split`).
- Batch sequences; per batch: `forward` → `cross_entropy(logits, target_cell)` → backward → Adam.
- Metrics: train/val loss, plus an interpretable val metric (top-1 cell hit rate and/or expected cell-distance error) so "is it actually predicting" is legible, not just a loss number.
- Checkpoint best val to `predictor_ckpt`. Runs on Kaggle T4.

### [MODIFY] `renderer.py`
- Fill the ghost-overlay slot reserved in Rung 1: for each hunter, draw its predicted heatmap as translucent cells (opacity ∝ probability) over the board, with the true prey drawn as before.
- Add a toggle: show one hunter's ghost vs. all three (three ghosts will often disagree — that is expected and worth seeing).

### [MODIFY] `run_demo.py`
- Load the `predictor.pt` checkpoint. Maintain per-hunter rolling history **and** GRU hidden state. Each step: build each hunter's observation, run the stateful predictor, get the K×K heatmap, convert to world cells, hand to the renderer. **Control stays heuristic** — the prediction is displayed, not yet acted on.

---

## Verification Plan

### Data (phase 1)
- [ ] Targets are not overwhelmingly "same cell" — the data contains real motion.
- [ ] Out-of-range rate logged; used to pick `K` (raise `K` if too high).
- [ ] Episode ids present so the split is honest.

### Training (phase 2)
- [ ] Val loss falls and flattens; **val is on held-out whole episodes**, not shuffled timesteps.
- [ ] Trained net beats a naive baseline (e.g. "prey stays at its last-seen cell" or uniform) on the val metric. **This is the Rung-2 success bar** — not capture rate.
- [ ] Interpretable metric (top-1 hit / distance error) reported, not loss alone.

### Overlay (phase 3)
- [ ] Ghost renders on the board in the correct world position (relative→world mapping is right).
- [ ] Freshly-seen prey → tight ghost on/near it; long-unseen prey → ghost spreads. (The cloud sharpening and spreading is the demo.)
- [ ] Three hunters show three independent ghosts.

### Size-agnostic smoke test
- [ ] Changing `grid_size` does not change any net tensor shape (output is relative K×K) — quick check now to confirm Rung 4 won't force a rebuild.

---

## Parked (revisit, don't solve now)
- **`K` value** — set after seeing the out-of-range rate from data collection + compute budget.
- **Distribution shift** — this net trains on *heuristic-hunter* data; at Rung 3 the trained hunters will visit new states. Fix then by regenerating data with the improved hunters and retraining. Non-issue this rung (control is heuristic).
- **LSTM** — `nn.GRU` → `nn.LSTM` is one line; A/B only if memory looks like the bottleneck.
- **Out-of-range targets** — v1 clamps to the nearest border cell; if the rate is high, raise `K` instead.
- **Net internals** (`gru_hidden`, `encoder_out_dim`, `history_len`) — tune while watching training, not on paper.
