#!/usr/bin/env python3
"""Compute the Geometric Fragility Score (GFS) from extracted hidden states.

For each model:

    GFS(M) = sum_l  w_l * d_l * (1 - cos_l)

where, at layer ``l`` over the difference ``D_l = instruct_l - base_l``:
    PC1_l   = top principal direction of D_l
    d_l     = |Cohen's d| of the PC1_l projection (safe vs general)
    Arditi_l= mean(instruct_safe_l) - mean(instruct_general_l)   (normalized)
    cos_l   = |cos(PC1_l, Arditi_l)|
    w_l     = depth weights linspace(0.5, 1.5, n_layers), normalized to sum 1

This reproduces the GFS reported in the paper. The cross-model
compliance-prediction step is intentionally NOT included here: it depends on
fine-tuning (attack-side) artifacts that are withheld per the paper's Ethics
Statement. GFS is a *pre-attack* diagnostic computed from activations alone.

Input: one ``.npz`` per model at ``<results_dir>/<model>/hidden_states.npz``
with arrays ``base_hs``, ``instruct_hs``, ``labels`` (see extract_hidden_states.py).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def _safe_mask(labels):
    return np.array([str(x) == "safe" for x in labels])


def cohens_d_profile(base_h, inst_h, labels):
    """Layer-wise |Cohen's d| of the PC1 projection of (instruct - base)."""
    safe = _safe_mask(labels)
    gen = ~safe
    profile = []
    for layer in range(base_h.shape[1]):
        diff = inst_h[:, layer, :] - base_h[:, layer, :]
        proj = PCA(n_components=1).fit_transform(diff).ravel()
        a, b = proj[safe], proj[gen]
        n1, n2 = len(a), len(b)
        pooled = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
        profile.append(float(abs(a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0)
    return profile


def cos_sim_profile(base_h, inst_h, labels):
    """Layer-wise |cos(PC1, Arditi difference-in-means direction)|."""
    safe = _safe_mask(labels)
    profile = []
    for layer in range(base_h.shape[1]):
        inst_layer = inst_h[:, layer, :]
        diff = inst_layer - base_h[:, layer, :]
        pc1 = PCA(n_components=1).fit(diff).components_[0]
        pc1 = pc1 / (np.linalg.norm(pc1) + 1e-10)
        arditi = inst_layer[safe].mean(0) - inst_layer[~safe].mean(0)
        arditi = arditi / (np.linalg.norm(arditi) + 1e-10)
        profile.append(float(abs(np.dot(pc1, arditi))))
    return profile


def compute_gfs(d_profile, cos_profile):
    n_layers = len(d_profile)
    weights = np.linspace(0.5, 1.5, n_layers)
    weights = weights / weights.sum()
    return float(sum(w * d * (1.0 - c) for w, d, c in zip(weights, d_profile, cos_profile)))


def load_npz(path: Path):
    data = np.load(str(path), allow_pickle=True)
    return (
        data["base_hs"].astype(np.float32),
        data["instruct_hs"].astype(np.float32),
        data["labels"],
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results_dir", required=True,
                    help="Dir with <model>/hidden_states.npz subfolders")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--models", default="llama,qwen,mistral,gemma",
                    help="Comma-separated model subfolder names")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for m in [s.strip() for s in args.models.split(",") if s.strip()]:
        npz = results_dir / m / "hidden_states.npz"
        if not npz.exists():
            print(f"{m}: NO DATA at {npz}")
            continue
        base_h, inst_h, labels = load_npz(npz)
        d_prof = cohens_d_profile(base_h, inst_h, labels)
        c_prof = cos_sim_profile(base_h, inst_h, labels)
        gfs = compute_gfs(d_prof, c_prof)
        peak_layer = int(np.argmax(d_prof))
        results[m] = {
            "gfs": gfs,
            "n_layers": len(d_prof),
            "peak_d": float(max(d_prof)),
            "peak_layer": peak_layer,
            "mean_cos_sim": float(np.mean(c_prof)),
            "cohens_d_profile": d_prof,
            "cos_sim_profile": c_prof,
        }
        print(f"{m}: GFS={gfs:.4f}  peak d={max(d_prof):.3f} @ layer "
              f"{peak_layer}/{len(d_prof)}  mean|cos|={np.mean(c_prof):.3f}")

    if not results:
        raise SystemExit("No hidden_states.npz found for any model.")

    with open(out_dir / "gfs_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_dir / 'gfs_results.json'}")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models = list(results)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].bar(models, [results[m]["gfs"] for m in models])
        axes[0].set_ylabel("GFS")
        axes[0].set_title("Geometric Fragility Score")
        for ax, key, title in [
            (axes[1], "cohens_d_profile", "Per-layer |Cohen's d| (PC1)"),
            (axes[2], "cos_sim_profile", "|cos(PC1, Arditi)|"),
        ]:
            for m in models:
                r = results[m]
                ax.plot(np.linspace(0, 1, r["n_layers"]), r[key], label=m, linewidth=2)
            ax.set_xlabel("Relative layer depth")
            ax.set_title(title)
            ax.legend()
        plt.tight_layout()
        png = out_dir / "gfs_analysis.png"
        plt.savefig(png, dpi=150)
        print(f"Saved {png}")


if __name__ == "__main__":
    main()
