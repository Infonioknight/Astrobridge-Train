# stage1/stage2 run through `accelerate launch` so the same command works on one GPU or several
# on one node — accelerate no-ops down to plain single-process/single-GPU if unconfigured.
# Override on the command line, e.g.: make stage1 ACCELERATE_CONFIG=configs/my_cluster.yaml
ACCELERATE_CONFIG ?= configs/accelerate_ddp.yaml

.PHONY: test manifest captions cache stage1 eval stage2 install check-access publish infer

install:
	pip install -e ".[dev]"

test:
	pytest -q tests/

check-access:
	python scripts/check_access.py

manifest:
	python scripts/00_build_manifest.py

captions:
	python scripts/01_generate_captions.py

cache:
	python scripts/02_cache_embeddings.py

stage1:
	accelerate launch --config_file $(ACCELERATE_CONFIG) scripts/03_train_stage1.py

eval:
	@test -n "$(CKPT)" || (echo "Usage: make eval CKPT=outputs/checkpoints/stage1/best" && exit 1)
	python scripts/04_eval.py --checkpoint-dir $(CKPT)

stage2:
	accelerate launch --config_file $(ACCELERATE_CONFIG) scripts/05_train_stage2.py

publish:
	@test -n "$(CKPT)" || (echo "Usage: make publish CKPT=outputs/checkpoints/stage2/best REPO=your-org/astrobridge-captioner-v1" && exit 1)
	@test -n "$(REPO)" || (echo "Usage: make publish CKPT=outputs/checkpoints/stage2/best REPO=your-org/astrobridge-captioner-v1" && exit 1)
	python scripts/06_publish_model.py --checkpoint-dir $(CKPT) --repo-id $(REPO)

infer:
	@test -n "$(CKPT)" || (echo "Usage: make infer CKPT=outputs/checkpoints/stage2/best QUESTION='What kind of object is this?' [LORA=...] [IMAGE=cutout.npy] [SPECTRUM=spectrum.npz] [SURVEY=desi]" && exit 1)
	@test -n "$(QUESTION)" || (echo "Usage: make infer CKPT=outputs/checkpoints/stage2/best QUESTION='What kind of object is this?' [LORA=...] [IMAGE=cutout.npy] [SPECTRUM=spectrum.npz] [SURVEY=desi]" && exit 1)
	python scripts/07_infer.py --checkpoint-dir $(CKPT) --question "$(QUESTION)" \
		$(if $(LORA),--lora-dir $(LORA)) \
		$(if $(IMAGE),--image-npy $(IMAGE)) \
		$(if $(SPECTRUM),--spectrum-npz $(SPECTRUM)) \
		$(if $(SURVEY),--spectrum-survey $(SURVEY))
