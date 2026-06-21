#!/usr/bin/env python3
"""Per-layer safety-subspace analysis (paper Claim 1).

At each layer, contrastive PCA (instruct vs base covariance) yields a candidate
safety-separating direction. We then measure:
    * |Cohen's d| of the cPCA-PC1 projection (safe vs general),
    * PERMANOVA pseudo-F and p-value in the full activation space,
    * RBF-kernel MMD between safe and general activations,
with Benjamini-Hochberg FDR correction across layers for the PERMANOVA p-values.

Input: ``<results_dir>/<model>/hidden_states.npz`` (see extract_hidden_states.py).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common_utils import (
    bh_fdr_correction,
    compute_cohens_d,
    compute_mmd,
    contrastive_pca,
    permanova_test,
)


def analyze_model(base_h, inst_h, labels, alpha, n_permutations, mmd_subsample):
    safe = np.array([str(x) == "safe" for x in labels])
    gen = ~safe
    n_layers = base_h.shape[1]

    layers = []
    p_raw = []
    for layer in range(n_layers):
        inst_layer = inst_h[:, layer, :]
        base_layer = base_h[:, layer, :]

        comps, _ = contrastive_pca(inst_layer, base_layer, n_components=1, alpha=alpha)
        v = comps[0]
        proj = inst_layer @ v
        d = abs(compute_cohens_d(proj[safe], proj[gen]))

        f_stat, p_val = permanova_test(inst_layer, safe.astype(int), n_permutations=n_permutations)

        if mmd_subsample and inst_layer.shape[0] > mmd_subsample:
            rng = np.random.default_rng(0)
            si = rng.choice(np.where(safe)[0], mmd_subsample // 2, replace=False)
            gi = rng.choice(np.where(gen)[0], mmd_subsample // 2, replace=False)
            mmd = compute_mmd(inst_layer[si], inst_layer[gi])
        else:
            mmd = compute_mmd(inst_layer[safe], inst_layer[gen])

        layers.append({"layer": layer, "cohens_d": float(d),
                       "permanova_F": float(f_stat), "permanova_p": float(p_val),
                       "mmd": float(mmd)})
        p_raw.append(p_val)

    rejected, p_adj = bh_fdr_correction(np.array(p_raw), q=0.05)
    for i, lay in enumerate(layers):
        lay["permanova_p_bh"] = float(p_adj[i])
        lay["permanova_sig_bh"] = bool(rejected[i])

    peak = max(layers, key=lambda x: x["cohens_d"])
    return {
        "n_layers": n_layers,
        "peak_layer": peak["layer"],
        "peak_cohens_d": peak["cohens_d"],
        "n_layers_bh_significant": int(sum(l["permanova_sig_bh"] for l in layers)),
        "layers": layers,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--models", default="llama,qwen,mistral,gemma")
    ap.add_argument("--alpha", type=float, default=100.0, help="cPCA contrast strength")
    ap.add_argument("--n_permutations", type=int, default=1000)
    ap.add_argument("--mmd_subsample", type=int, default=0,
                    help="If >0, subsample this many points for MMD (speed)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out = {}
    for m in [s.strip() for s in args.models.split(",") if s.strip()]:
        npz = results_dir / m / "hidden_states.npz"
        if not npz.exists():
            print(f"{m}: NO DATA at {npz}")
            continue
        data = np.load(str(npz), allow_pickle=True)
        res = analyze_model(
            data["base_hs"].astype(np.float32),
            data["instruct_hs"].astype(np.float32),
            data["labels"],
            args.alpha, args.n_permutations, args.mmd_subsample,
        )
        out[m] = res
        print(f"{m}: peak |d|={res['peak_cohens_d']:.3f} @ layer "
              f"{res['peak_layer']}/{res['n_layers']}  "
              f"BH-sig layers={res['n_layers_bh_significant']}/{res['n_layers']}")

    if not out:
        raise SystemExit("No hidden_states.npz found for any model.")
    with open(out_dir / "subspace_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_dir / 'subspace_results.json'}")


if __name__ == "__main__":
    main()
