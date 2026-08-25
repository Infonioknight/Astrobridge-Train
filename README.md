# captioner

Trains a model that writes captions for astronomy images and spectra. Follow these steps in
order — each one should just work if the one before it did.

## 1. Install

```bash
pip install -e ".[dev]"
```

## 2. Log into Hugging Face

```bash
huggingface-cli login
```

Paste a token from https://huggingface.co/settings/tokens when asked. Then request access on
these three pages (click "Agree", approval isn't always instant):
- https://huggingface.co/datasets/gapatron/legacy_survey_south_images_captions
- https://huggingface.co/polymathic-ai/aion-base
- https://huggingface.co/Qwen/Qwen3.5-9B

## 3. Check access actually works

```bash
make check-access
```

Every line should say `OK`. If any says `FAIL`, it tells you exactly what's missing — fix that
before moving on, don't skip ahead.

## 4. Run the tests

```bash
make test
```

Should finish with `19 passed`. Runs in seconds, no GPU needed. **If this isn't green, stop and
get help — nothing below this is trustworthy until it is.**

## 5. Build the data (no GPU needed)

```bash
make manifest
make captions
```

## 6. Everything below needs a GPU

```bash
make cache                              # one-time: encode images + spectra
make stage1                             # the actual training run
make eval CKPT=outputs/checkpoints/stage1/best   # sanity-check the result
make stage2                             # only run this if step above looks good
```

## Multiple GPUs on one machine?

Open `configs/accelerate_ddp.yaml`, set `num_processes` to your GPU count, save. `make stage1`
and `make stage2` pick it up automatically — nothing else to configure.

## Anything else

Full detail — what each step actually checks, why things are built this way, known limitations —
lives in `../07_captioner_operational_reference.md`. This file stays short on purpose.
