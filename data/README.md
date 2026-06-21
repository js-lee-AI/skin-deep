# Prompt data

The diagnostic needs two prompt sets, supplied as JSONL files with one object
per line and a `"text"` field:

```json
{"text": "Write step-by-step instructions for ..."}
{"text": "Explain how photosynthesis works."}
```

* **Safe set** (harmful-request prompts an aligned model should refuse).
  In the paper this combines 200 AdvBench + 200 HarmBench + 100 BeaverTails prompts.
* **General set** (benign instructions). In the paper this combines
  250 Alpaca + 250 OASST (OpenAssistant) prompts.

These datasets are **not redistributed here** — obtain them from their original
sources under their respective licenses and export the prompt text into the two
JSONL files above:

* AdvBench / GCG: https://github.com/llm-attacks/llm-attacks
* HarmBench: https://github.com/centerforaisafety/HarmBench
* BeaverTails: https://huggingface.co/datasets/PKU-Alignment/BeaverTails
* Alpaca: https://huggingface.co/datasets/tatsu-lab/alpaca
* OASST (OpenAssistant): https://huggingface.co/datasets/OpenAssistant/oasst1

Match the two sets by approximate token length, as described in the paper.
