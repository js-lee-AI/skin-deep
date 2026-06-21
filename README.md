# Skin-Deep: A Geometric Diagnostic for Alignment Fragility in LLM Representations

Reference implementation of the **Geometric Fragility Score (GFS)** — a
*pre-attack* diagnostic that reads alignment fragility directly from an aligned
model's hidden-state activations, before any prompt- or weight-level attack is
run.

> Paper: *Skin-Deep: A Geometric Diagnostic for Alignment Fragility in Large
> Language Model Representations*.

## What this repository contains

A self-contained pipeline for the **pre-attack** analyses in the paper:

| Script | Paper claim | What it does |
| --- | --- | --- |
| `extract_hidden_states.py` | — | Extract last-token, all-layer activations from an aligned model and its version-matched base model (raw prompts, or `--chat_template`). |
| `analyze_subspace.py` | C1 (subspace) | Per-layer contrastive-PCA direction, Cohen's *d*, PERMANOVA, RBF-MMD, with BH-FDR correction. |
| `compute_gfs.py` | C4 (diagnostic) | Compute the Geometric Fragility Score and per-layer profiles. |
| `common_utils.py` | — | Shared library (extraction, cPCA, Cohen's *d*, MMD, PERMANOVA, BH-FDR). |

### Responsible release

Consistent with the paper's **Ethics Statement**, this repository releases only
the **pre-attack diagnostic** pipeline. The following attack-side artifacts are
**deliberately withheld** and are *not* part of this repository:

1. layer-level direction-ablation hooks and the generation-time code path that applies them;
2. model-specific peak-layer indices at attack-coordinate granularity;
3. LoRA adapter weights from the fine-tuning fragility curve;
4. the difference-in-means ("Arditi") refusal-direction extraction specialized to specific model families.

GFS is intended as a **defensive** tool: flagging fragile refusal behavior in a
checkpoint *before* release. Please use it accordingly.

## Installation

```bash
pip install -r requirements.txt
```

Gated checkpoints (e.g. Llama, Gemma) require authentication:

```bash
huggingface-cli login
```

## Data

Prepare two JSONL prompt files (a harmful-request "safe" set and a benign
"general" set) as described in [`data/README.md`](data/README.md). The source
datasets are not redistributed here.

## Usage

**1. Extract activations** for an aligned model and its version-matched base:

```bash
python skin_deep/extract_hidden_states.py \
    --instruct_model meta-llama/Llama-3.1-8B-Instruct \
    --base_model     meta-llama/Llama-3.1-8B \
    --safe_prompts    data/safe.jsonl \
    --general_prompts data/general.jsonl \
    --output results/llama/hidden_states.npz
```

Repeat for each model into `results/<model>/hidden_states.npz`
(e.g. `llama`, `qwen`, `mistral`, `gemma`).

By default prompts are passed **raw** (no chat template) — the path used by the GFS ranking. To reproduce the **chat-template robustness** analyses, add `--chat_template` (and a larger `--max_length`, e.g. `256`); each prompt is then wrapped as a single user turn with the instruct tokenizer's `apply_chat_template` (`add_generation_prompt=True`) before extraction.

**2. Subspace analysis (Claim 1):**

```bash
python skin_deep/analyze_subspace.py \
    --results_dir results --output_dir results/subspace \
    --models llama,qwen,mistral,gemma
```

**3. Geometric Fragility Score (Claim 4):**

```bash
python skin_deep/compute_gfs.py \
    --results_dir results --output_dir results/gfs \
    --models llama,qwen,mistral,gemma
```

This writes `gfs_results.json` and `gfs_analysis.png` (GFS bar chart plus
per-layer Cohen's *d* and PC1–Arditi cosine profiles).

## Repository structure

```
skin_deep/
  common_utils.py          # shared library
  extract_hidden_states.py # step 1: activation extraction
  analyze_subspace.py      # step 2: cPCA subspace + full-space tests (C1)
  compute_gfs.py           # step 3: Geometric Fragility Score (C4)
data/
  README.md                # how to assemble the prompt sets
requirements.txt
LICENSE
```

## Citation

A BibTeX entry will be added once the arXiv version is available.

## License

The code in this repository is released under the [MIT License](LICENSE). The paper itself is distributed under CC BY 4.0 via arXiv.
