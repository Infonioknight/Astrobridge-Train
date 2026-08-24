# captioner

n-modality (image, spectra) captioner: Q-Former + LoRA-tuned Qwen3.5-9B. Full reasoning/details:
`../07_captioner_operational_reference.md`.

| Command | Requires | Does |
|---|---|---|
| `make install` | CPU | Installs the package |
| `make test` | CPU | Runs the test suite |
| `make manifest` | CPU, HF access | Builds the object manifest + train/val/test split |
| `make captions` | CPU, HF access | Generates + validates per-tier captions |
| `make cache` | GPU | Encodes spectra with AION, caches embeddings (image: blocked, see reference doc) |
| `make stage1` | GPU | Trains the fusion stack, LLM frozen |
| `make eval --checkpoint-dir <dir>` | GPU | Runs the groundedness gate on a checkpoint |
| `make stage2` | GPU | LoRA fine-tunes the LLM (only after eval passes) |

Run in order, top to bottom. `make stage1`/`make stage2` use `accelerate` — set GPU count in
`configs/accelerate_ddp.yaml`.
