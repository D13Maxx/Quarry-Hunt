import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_raw_offsets(data_dir: pathlib.Path) -> np.ndarray:
    files = sorted(data_dir.glob("raw_offset_*_*.npy"))
    if not files:
        raise FileNotFoundError(f"No raw_offset_*.npy files in {data_dir}")
    chunks = [np.load(f) for f in files]
    return np.concatenate(chunks, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Analyze OOF rates across K values")
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    offsets = load_raw_offsets(data_dir)
    cheby = np.maximum(np.abs(offsets[:, 0]), np.abs(offsets[:, 1]))
    n = len(cheby)

    print(f"Loaded {n:,} examples from {data_dir}\n")

    # --- Percentile distribution ---
    pcts = [50, 90, 95, 99]
    vals = np.percentile(cheby, pcts)
    print("Chebyshev offset distribution:")
    for p, v in zip(pcts, vals):
        print(f"  p{p:02d} = {v:.0f}")
    print(f"  max = {cheby.max()}")
    print()

    # --- OOF table ---
    ks = [11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 49]
    print(f"{'K':>4}  {'radius':>6}  {'OOF%':>8}  {'OOF count':>10}")
    print("-" * 36)
    oof_rates = []
    for k in ks:
        r = (k - 1) // 2
        oof = (cheby > r).sum()
        rate = oof / n
        oof_rates.append(rate)
        print(f"{k:>4}  {r:>6}  {rate:>8.3%}  {oof:>10,}")
    print()

    # --- Plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ks, [r * 100 for r in oof_rates], "o-", color="#e74c3c", linewidth=2, markersize=6)
    ax1.set_xlabel("Window size K")
    ax1.set_ylabel("Out-of-frame rate (%)")
    ax1.set_title("OOF Rate vs K")
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.hist(cheby, bins=range(0, int(cheby.max()) + 2), color="#3498db", edgecolor="white", alpha=0.85)
    ax2.set_xlabel("Chebyshev distance (max(|dr|, |dc|))")
    ax2.set_ylabel("Count")
    ax2.set_title("Chebyshev Offset Distribution")
    ax2.grid(True, alpha=0.3)
    for p, v in zip(pcts, vals):
        ax2.axvline(v, color="#e74c3c", linestyle="--", alpha=0.7, label=f"p{p}={v:.0f}")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_path = data_dir / "oof_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
