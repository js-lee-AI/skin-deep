#!/usr/bin/env python3
"""Extract last-token hidden states from an aligned model and its base model.

Saves a compressed ``.npz`` with:
    base_hs:     (n_prompts, n_layers, hidden_dim)
    instruct_hs: (n_prompts, n_layers, hidden_dim)
    labels:      (n_prompts,)  array of "safe" / "general"

The aligned ("instruct") model and its version-matched base checkpoint must
share the same architecture so that the per-layer difference is well defined.

Two extraction modes (see the paper, "Prompts and Activations"):

* **raw** (default) -- prompts are passed directly to each model. This is the
  path used for the Geometric Fragility Score ranking (run ``compute_gfs.py``
  on the resulting ``.npz``).
* **chat template** (``--chat_template``) -- each prompt is wrapped as a single
  user turn and formatted with the *instruct* tokenizer's ``apply_chat_template``
  (``add_generation_prompt=True``); both models then encode the same formatted
  strings with their own tokenizers. This reproduces the chat-template
  robustness analyses. Base foundation models usually lack a chat template,
  which is why the instruct tokenizer is used for formatting.

Prompts are read from two JSONL files (one object per line, each with a
``"text"`` field):
    --safe_prompts     harmful-request prompts an aligned model should refuse
    --general_prompts  benign instructions

Gated checkpoints (e.g. Llama, Gemma) require `huggingface-cli login` first.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common_utils import extract_hidden_states


def read_texts(path: str):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"] if isinstance(obj, dict) else str(obj))
    return texts


def apply_chat_template(prompts, instruct_model):
    """Format each prompt as a single user turn via the instruct chat template."""
    tok = AutoTokenizer.from_pretrained(instruct_model, trust_remote_code=True)
    if tok.chat_template is None:
        raise RuntimeError(
            f"{instruct_model} has no chat_template; cannot use --chat_template."
        )
    formatted = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    print(f"Applied chat template. Sample (200 chars): {formatted[0][:200]!r}")
    return formatted


def load_model(model_id: str, dtype):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    )
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--instruct_model", required=True, help="Aligned model id or path")
    ap.add_argument("--base_model", required=True, help="Version-matched base model id or path")
    ap.add_argument("--safe_prompts", required=True, help="JSONL of harmful-request prompts")
    ap.add_argument("--general_prompts", required=True, help="JSONL of benign instructions")
    ap.add_argument("--output", required=True, help="Output .npz path")
    ap.add_argument("--chat_template", action="store_true",
                    help="Format prompts with the instruct tokenizer's chat template "
                         "(reproduces the chat-template robustness path).")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=128,
                    help="Max tokens (use a larger value, e.g. 256, with --chat_template).")
    args = ap.parse_args()

    safe = read_texts(args.safe_prompts)
    general = read_texts(args.general_prompts)
    prompts = safe + general
    labels = np.array(["safe"] * len(safe) + ["general"] * len(general))
    print(f"Loaded {len(safe)} safe + {len(general)} general prompts")

    if args.chat_template:
        prompts = apply_chat_template(prompts, args.instruct_model)

    dtype = torch.float16

    print(f"[base]     {args.base_model}")
    base_model, base_tok = load_model(args.base_model, dtype)
    base_hs = extract_hidden_states(
        base_model, base_tok, prompts,
        batch_size=args.batch_size, gpu=args.gpu, max_length=args.max_length,
    )
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[instruct] {args.instruct_model}")
    inst_model, inst_tok = load_model(args.instruct_model, dtype)
    instruct_hs = extract_hidden_states(
        inst_model, inst_tok, prompts,
        batch_size=args.batch_size, gpu=args.gpu, max_length=args.max_length,
    )
    del inst_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if base_hs.shape != instruct_hs.shape:
        raise ValueError(
            f"base/instruct shape mismatch {base_hs.shape} vs {instruct_hs.shape}; "
            "the base must be the version-matched base of the instruct model."
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, base_hs=base_hs, instruct_hs=instruct_hs, labels=labels)
    print(f"Saved {out}  base_hs={base_hs.shape}  instruct_hs={instruct_hs.shape}")


if __name__ == "__main__":
    main()
