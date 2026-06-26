# Project Quarry — Design Document
*Multi-agent pursuit-evasion with learned world models.*
*Working title — rename freely (alternatives: Phantom, Foglands). Version 0.1 — environment + build plan locked; per-rung internals marked TBD.*

---

## 1. What we're building
A grid-based hunting simulation you can watch unfold live: three hunters pursue a single evasive prey through a foggy forest. The twist — and the entire point — is that the hunters can barely see. Each one acts on a **learned prediction** of where the unseen prey is, not on direct sight, and an LLM "spotter" provides sparse high-level coordination. The signature visual is a **ghost overlay**: the real prey rendered alongside each hunter's predicted belief, so you can literally watch belief diverge from reality and snap back.

## 2. Goals & constraints
- **Dual goal:** an impressive portfolio piece *and* a genuine deep-dive into world models / model-based multi-agent RL.
- **Compute (zero-budget):** Kaggle (30 GPU-hr/week, T4 16GB) as the training workhorse; Colab as overflow; Groq free tier for LLM inference; local RTX 3060 (6GB) for inference, light iteration, and rendering. No paid cloud, no credit card.
- **Method:** built in ordered *rungs*, one hard thing introduced at a time, with a documented decision history. Every source of difficulty is a logged, tunable dial.

## 3. Core thesis
The hunters are nearly blind to the prey, so the work of *finding* it falls entirely on (a) a learned belief about its hidden location and (b) the spotter. This makes "world model" something concrete and visible rather than abstract: the predicted ghost is the world model, on screen, being right and being wrong in real time. The project is, at heart, a study of agents acting on belief instead of sight — and of whether explicit prediction plus coordination beats reactive baselines.

## 4. Environment specification

### 4.1 Field
- Discrete grid, top-down, viewed from above.
- **Field size is a curriculum dial**, starting at **25×25** and growing as competence builds.
- **Vision sizes are absolute** (they do *not* scale with the grid). A larger field therefore means relatively blinder hunters = harder prediction. Growing the field also naturally gives the **prey** partial observability (its 13×13 view becomes a small fraction of a large board) — exactly what the arms-race rung needs. So the size curriculum and the rung progression are coupled: small field for the hunter-learning rungs, larger field as the learning prey comes online.
- **Architectural consequence (load-bearing):** to make size a smooth dial, **all networks must be size-agnostic** — ego-centric local observations (the fixed 5×5 patch) and predictions in a relative / ego-centric frame (or fully-convolutional), never a flattened whole-grid vector. This lets weights transfer across field sizes instead of forcing a retrain per size. Must be in place from rung 1.

### 4.2 Forest & kill rim
- The **forest** (safe ground) fills the interior — ~90% of the field at the hard target. **Coverage is a difficulty dial:** start more generous (closer edges, easier to drive out), tighten toward 90%.
- The remaining outer band is the **kill zone**: a thin rim around the *entire* perimeter. If the prey ends a step anywhere in the rim, a sharpshooter eliminates it (100% accuracy) → hunters win. This is a one-line rule, not an agent to train.
- **Rationale:** the rim replaces physical walls. It turns the prey's evasion reflex lethal near the edge (flee = step out = death), so a far-sighted prey cannot simply run forever. The hunters effectively carry a wall on all four sides and win by compressing the prey outward.

### 4.3 Agents
- **3 hunters** (homogeneous) and **1 prey**.
- **Movement:** simultaneous — all four choose a move on the same tick, the world advances once (PettingZoo Parallel API semantics). **Equal movement speed** (the prey's edge is perception, not legs).

### 4.4 Vision (asymmetric, absolute)
- **Hunter:** 5×5 ego-centric (radius 2). *Floor rationale:* vision radius must exceed capture radius (1), or a hunter only sees the prey at the instant it could already grab it — sight would be useless for the search.
- **Prey:** 13×13 ego-centric (radius 6) — 3× the hunters' reach. The prey detects approaches long before the hunters detect it.

### 4.5 Magic zones
- Fixed **3×3 tiles**, **randomly placed each episode**. Part of the world from rung 1.
- **Effect:** a **hunter** standing inside a zone has its view clamped to the zone's 9 cells — it loses peripheral sight and cannot see out. The **prey is unaffected** and can deliberately lure hunters through them.
- **Count is a dial, starting at 0**, ramped up as the hunters learn to cope. (Mechanic is core to the world; difficulty is staged.)
- *Open edge cases (TBD):* zone overlapping the rim; hunter standing on a zone boundary cell.

### 4.6 Win / lose (checked in this order after each simultaneous step)
1. Prey is in the kill rim → **hunters win** (sharpshooter).
2. Any hunter is diagonally adjacent (any of the 8 surrounding cells) to the prey → **hunters win** (capture).
3. Time limit reached → **prey wins**.
- **Time-limit length is a dial**, tuned by watching capture rates.

### 4.7 Reward shaping (principle)
Partial progress must pay: closing distance to the *predicted* prey location, shrinking the prey's escape options, herding it rim-ward. A reward that fires only on the (initially rare) capture gives no gradient — the classic sparse-reward dead run. Exact reward terms are TBD per rung.

### 4.8 Dials summary
| Dial | Start | Direction | Purpose |
|---|---|---|---|
| Field size | 25×25 | grow | harder prediction; gives prey partial-obs |
| Forest coverage | generous | → ~90% | harder to drive prey out |
| Magic-zone count | 0 | increase | hunter sensing hazard |
| Time limit | TBD | tune | balance win rates |
| Hunter vision | 5×5 | fixed | — |
| Prey vision | 13×13 | fixed | — |
| Speed | equal | fixed | — |

## 5. The world model
- A small **per-hunter** network that turns the hunter's recent ego-centric observations into a **belief about the unseen prey's location** (the "ghost").
- **v1 (light):** predict the prey's position in an ego-centric / relative frame (size-agnostic).
- **Deepen later:** fuller latent forward dynamics — multi-step rollouts, Dreamer-flavored.
- **Hero visual:** render the real prey plus each hunter's predicted ghost; the visible gap is the story.

## 6. The LLM spotter
- A "scope on the cliff." **Sparse** calls (every N steps), **never** in the fast control loop (per-tick LLM calls are killed by latency + rate limits).
- Sees **one N×N window of its own choosing** with perfect clarity; cannot see the whole board. **It aims its own scope** — deciding where to look is itself a reasoning-about-the-hidden-world task.
- **Output:** high-level role / tactic assignment to the hunters (e.g. "P1 hold the south edge; P2 and P3 push north").
- **Model:** Groq Llama 3.3 70B (free tier).
- **Ablations:** spotter on vs off; LLM-aimed scope vs randomly-aimed scope.

## 7. Build plan — the rungs
One hard thing per rung. Each rung is a complete, demoable artifact; we only climb once the current rung works and is understood.

| Rung | Adds | Standalone artifact? | Exit criteria |
|---|---|---|---|
| **1. Skeleton** | Grid, forest, kill rim, magic-zone mechanic (count 0), scripted prey, heuristic hunters, live pygame render | Watchable sim, no ML | Full loop runs; render + GIF export work; env tests green |
| **2. World model** | Per-hunter prediction net + ghost overlay; heuristic control still | **Yes — the hero demo** | Prediction beats a naive baseline; overlay renders cleanly |
| **3. Trained hunter policy** | RL policy (PPO, discrete) consuming the prediction; prey still scripted | Yes | Capture rate clearly beats heuristic/random baseline |
| **4. Learning prey** | Prey becomes RL — the arms race; grow the field | **Yes — the showcase** | Stable co-adaptation; emergent evasion / anticipation |
| **5. LLM spotter** | Sparse Groq role assignment; turn magic-zone count up | Yes | Spotter-on beats spotter-off on capture rate / time-to-capture |

*SAC enters at the continuous-space port, a deepening of rung 3+.*

## 8. Tech stack
- **Environment:** custom, conforming to the **PettingZoo Parallel API**. Study `simple_tag` (now in the `mpe2` package) and `pursuit_v4` as reference designs and sanity baselines — run them first, build our own after.
- **Learning:** PyTorch; size-agnostic (CNN / ego-centric) networks.
- **Rendering:** pygame live view + GIF/video export for the portfolio.
- **Training:** Kaggle T4 (overflow on Colab). **Inference / iteration / rendering:** local RTX 3060.
- **LLM:** Groq (Llama 3.3 70B).

## 9. Open items (to detail when we reach them)
- Exact world-model architecture (patch-sequence encoder; relative-frame prediction head).
- Concrete reward terms and magnitudes, per rung.
- Confirm RL algorithm path (PPO for discrete grid → SAC at the continuous port).
- Scope window size N; LLM call cadence; prompt + state-snapshot format.
- Time-limit value; forest-coverage schedule; field-growth schedule; magic-zone ramp schedule.
- Magic-zone edge cases (overlap with rim; boundary cells).
- Scripted-prey heuristic (boundary-aware evasion) for rungs 1–3.

## 10. Decision log (the *why*, not just the *what*)
- **Kill-boundary instead of walls** — makes the entire perimeter a corner and turns a far-sighted prey's flee reflex lethal at the edge; solves "the prey just runs forever."
- **Vision radius (2) > capture radius (1)** — otherwise hunters only see the prey at the moment of capture, so sight can't aid the hunt.
- **Asymmetric sight (prey sees 3× the hunters)** — forces coordination + prediction to carry the work; matches validated pursuit-evasion designs.
- **Size-agnostic networks** — so field size can be a curriculum dial without retraining per size; couples neatly with giving the prey partial observability as the board grows.
- **One hard thing per rung; arms race last** — simultaneous co-learning is the most unstable part; everything beneath it must work and be attributable before it's switched on.
- **Magic zones core but count-dialable** — the world has them from day one (an honest world), but difficulty still ramps from zero.
- **Difficulty as logged dials** — preserves control and a documented history; keeps the hunt "hard but winnable" by watching capture rates.
- **Custom env on a standard API** — own the logic, rendering, and knobs (which off-the-shelf envs fight); borrow the ecosystem via the PettingZoo Parallel interface.
