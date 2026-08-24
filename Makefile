# stage1/stage2 run through `accelerate launch` so the same command works on one GPU or several
# on one node — accelerate no-ops down to plain single-process/single-GPU if unconfigured.
# Override on the command line, e.g.: make stage1 ACCELERATE_CONFIG=configs/my_cluster.yaml
ACCELERATE_CONFIG ?= configs/accelerate_ddp.yaml

.PHONY: test manifest captions cache stage1 eval stage2 install

install:
	pip install -e ".[dev]"

test:
	pytest -q tests/

manifest:
	python scripts/00_build_manifest.py

captions:
	python scripts/01_generate_captions.py

cache:
	python scripts/02_cache_embeddings.py

stage1:
	accelerate launch --config_file $(ACCELERATE_CONFIG) scripts/03_train_stage1.py

eval:
	python scripts/04_eval.py

stage2:
	accelerate launch --config_file $(ACCELERATE_CONFIG) scripts/05_train_stage2.py
