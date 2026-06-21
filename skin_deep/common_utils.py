"""Shared utilities for the Skin-Deep pre-attack diagnostic.

Includes hidden-state extraction, contrastive PCA, Cohen's d, RBF-MMD,
PERMANOVA, and Benjamini-Hochberg FDR correction.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# JSON serialization
# --------------------------------------------------------------------------- #
def convert_for_json(obj):
    """Recursively convert numpy / torch types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    return obj


def load_prompts(path: str) -> List[Dict]:
    """Load prompts from a JSONL file (one JSON object per line)."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    logger.info("Loaded %d prompts from %s", len(prompts), path)
    return prompts


# --------------------------------------------------------------------------- #
# Hidden-state extraction (final attended token, all transformer layers)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 8,
    gpu: int = 0,
    max_length: int = 128,
) -> np.ndarray:
    """Return last-token hidden states for every layer.

    Output shape: (n_prompts, n_layers, hidden_dim). The embedding layer
    (index 0 of ``hidden_states``) is dropped, so ``n_layers`` is the number
    of transformer blocks.
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_hidden = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting hidden states"):
        batch_texts = prompts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        outputs = model(**enc, output_hidden_states=True)
        # (batch, n_layers+1, seq, hidden)
        hidden_states = torch.stack(outputs.hidden_states, dim=1)

        attention_mask = enc["attention_mask"]            # (batch, seq)
        seq_lengths = attention_mask.sum(dim=1) - 1        # index of last valid token

        for i in range(hidden_states.size(0)):
            last_pos = int(seq_lengths[i].item())
            h = hidden_states[i, 1:, last_pos, :]           # (n_layers, hidden)
            all_hidden.append(h.cpu().float().numpy())

        del outputs, hidden_states, enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = np.stack(all_hidden, axis=0)
    logger.info("Extracted hidden states: %s", result.shape)
    return result


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation (1-D inputs)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((m1 - m2) / pooled_std)


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    """Unbiased RBF-kernel MMD^2 between samples X (n, d) and Y (m, d)."""
    from sklearn.metrics.pairwise import rbf_kernel

    if gamma is None:
        from scipy.spatial.distance import cdist
        median_dist = np.median(cdist(X, Y, metric="euclidean").ravel())
        if median_dist == 0:
            median_dist = 1.0
        gamma = 1.0 / (2.0 * median_dist ** 2)

    K_xx = rbf_kernel(X, X, gamma=gamma)
    K_yy = rbf_kernel(Y, Y, gamma=gamma)
    K_xy = rbf_kernel(X, Y, gamma=gamma)
    n, m = K_xx.shape[0], K_yy.shape[0]
    np.fill_diagonal(K_xx, 0)
    np.fill_diagonal(K_yy, 0)
    mmd2 = K_xx.sum() / (n * (n - 1)) + K_yy.sum() / (m * (m - 1)) - 2 * K_xy.sum() / (n * m)
    return float(mmd2)


def permanova_test(
    X: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 1000,
    metric: str = "euclidean",
    seed: int = 42,
) -> Tuple[float, float]:
    """PERMANOVA pseudo-F test for a difference in multivariate group centroids.

    Returns ``(F_statistic, p_value)``. The smallest attainable p-value is
    ``1 / (n_permutations + 1)``.
    """
    from scipy.spatial.distance import pdist, squareform

    n = len(labels)
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    D2 = squareform(pdist(X, metric=metric)) ** 2

    def pseudo_f(lab):
        ss_total = D2.sum() / (2 * n)
        ss_within = 0.0
        for g in unique_labels:
            mask = lab == g
            n_g = mask.sum()
            if n_g > 0:
                ss_within += D2[np.ix_(mask, mask)].sum() / (2 * n_g)
        ss_between = ss_total - ss_within
        if ss_within == 0:
            return float("inf")
        return (ss_between / (k - 1)) / (ss_within / (n - k))

    f_obs = pseudo_f(labels)
    rng = np.random.default_rng(seed)
    count = sum(1 for _ in range(n_permutations) if pseudo_f(rng.permutation(labels)) >= f_obs)
    p_value = (count + 1) / (n_permutations + 1)
    return float(f_obs), float(p_value)


def bh_fdr_correction(p_values: np.ndarray, q: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction. Returns (rejected, adjusted_p)."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    corrected = np.zeros(n)
    corrected[n - 1] = sorted_p[n - 1]
    for i in range(n - 2, -1, -1):
        corrected[i] = min(corrected[i + 1], sorted_p[i] * n / (i + 1))
    corrected = np.clip(corrected, 0.0, 1.0)
    corrected_p = np.zeros(n)
    corrected_p[sorted_idx] = corrected
    return corrected_p < q, corrected_p


# --------------------------------------------------------------------------- #
# Contrastive PCA
# --------------------------------------------------------------------------- #
def contrastive_pca(
    target_data: np.ndarray,
    background_data: np.ndarray,
    n_components: int = 10,
    alpha: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Contrastive PCA: top eigenvectors of ``cov(target) - alpha * cov(background)``.

    Args:
        target_data:     (n_target, d) aligned-model activations.
        background_data: (n_bg, d) base-model activations.
        n_components:    number of directions to return.
        alpha:           contrast strength.

    Returns:
        components:  (n_components, d) contrastive principal directions.
        eigenvalues: (n_components,) corresponding eigenvalues.
    """
    target_centered = target_data - target_data.mean(axis=0)
    bg_centered = background_data - background_data.mean(axis=0)
    cov_contrast = np.cov(target_centered, rowvar=False) - alpha * np.cov(bg_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_contrast)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx[:n_components]].T, eigenvalues[idx[:n_components]]
