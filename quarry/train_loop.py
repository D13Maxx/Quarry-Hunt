import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from quarry.config import Config
from quarry.predictor import PreyPredictor
from quarry.dataset import TrajectoryDataset, split_episodes
from quarry.metrics import compute_all, top1_hit


def update_predictor(
    predictor: PreyPredictor,
    optimizer: torch.optim.Optimizer,
    obs_seq: torch.Tensor,
    target_r: torch.Tensor,
    target_c: torch.Tensor,
) -> tuple[float, float]:
    """One gradient step on the predictor. Returns (loss, top1_hit_rate)."""
    K = predictor.K
    logits = predictor(obs_seq).flatten(1)
    target = (target_r * K + target_c).long()
    loss = F.cross_entropy(logits, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    hit = top1_hit(logits.detach(), target_r, target_c, K)
    return loss.item(), hit


def pred_batch_from_rollout(
    pred_buf: list[dict], device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert rollout predictor buffer to (obs_seq, target_r, target_c) batch."""
    obs_seq = torch.stack([e["obs_seq"] for e in pred_buf]).to(device)
    target_r = torch.tensor([e["target_r"] for e in pred_buf], dtype=torch.long, device=device)
    target_c = torch.tensor([e["target_c"] for e in pred_buf], dtype=torch.long, device=device)
    return obs_seq, target_r, target_c


def train(cfg: Config, data_dir: str, device: str = "cpu", epochs: int = 50) -> dict:
    train_eps, val_eps = split_episodes(data_dir, cfg.train_val_split)
    print(f"train episodes: {len(train_eps)}, val episodes: {len(val_eps)}")

    train_ds = TrajectoryDataset(data_dir, train_eps, cfg.num_hunters, cfg.history_len)
    val_ds = TrajectoryDataset(data_dir, val_eps, cfg.num_hunters, cfg.history_len)
    print(f"train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = PreyPredictor(cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.predictor_lr)
    K = cfg.window_K
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for obs_seq, tr, tc in train_dl:
            obs_seq = obs_seq.to(device)
            tr, tc = tr.to(device), tc.to(device)
            loss_val, _ = update_predictor(model, optimizer, obs_seq, tr, tc)
            train_loss += loss_val
            n_batches += 1

        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        all_logits, all_tr, all_tc = [], [], []
        n_val = 0
        with torch.no_grad():
            for obs_seq, tr, tc in val_dl:
                obs_seq = obs_seq.to(device)
                target = (tr * K + tc).long().to(device)
                logits = model(obs_seq).flatten(1)
                val_loss += nn.functional.cross_entropy(logits, target).item()
                all_logits.append(logits.cpu())
                all_tr.append(tr)
                all_tc.append(tc)
                n_val += 1

        val_loss /= max(n_val, 1)
        all_logits = torch.cat(all_logits)
        all_tr = torch.cat(all_tr).long()
        all_tc = torch.cat(all_tc).long()
        val_m = compute_all(all_logits, all_tr, all_tc, K)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_m)

        print(f"epoch {epoch+1:3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"top1={val_m['top1_hit']:.3f}  cheby={val_m['argmax_cheby']:.2f}  "
              f"exp_cheby={val_m['expected_cheby']:.2f}  true_p={val_m['true_cell_prob']:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.predictor_ckpt)
            print(f"  -> saved checkpoint (val_loss={val_loss:.4f})")

    return history
