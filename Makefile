.EXPORT_ALL_VARIABLES:
SHELL = bash

NOTEBOOK ?= gmic

.DEFAULT_GOAL := help
.PHONY: help sync run notebook test docker-build docker-build-gpu docker-run docker-run-gpu docker-test

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

sync:  ## Installe / met à jour les dépendances Python via uv
	uv sync

run:  ## Sert un notebook Quarto en live preview (NOTEBOOK=gmic|resnet18_training|rsna_comparison)
	uv run quarto preview docs/script_notebook/$(NOTEBOOK).qmd

notebook:  ## Rend un notebook Quarto en HTML (NOTEBOOK=gmic|resnet18_training|rsna_comparison)
	uv run quarto render docs/script_notebook/$(NOTEBOOK).qmd --to html

test:  ## Lance les tests unitaires
	uv run pytest scripts/test_validate_input.py -v

DOCKERFILE := docker/inference/Dockerfile

docker-build:  ## Build l'image GMIC tout-en-un (CPU, portable)
	DOCKER_BUILDKIT=1 docker build -f $(DOCKERFILE) -t gmic-inference:cpu .

docker-build-gpu:  ## Build l'image GMIC tout-en-un (GPU NVIDIA cu121)
	DOCKER_BUILDKIT=1 docker build -f $(DOCKERFILE) --build-arg TORCH_VARIANT=cu121 -t gmic-inference:gpu .

docker-run:  ## Lance la démo (preprocess + inférence) dans l'image CPU
	docker run --rm gmic-inference:cpu

docker-run-gpu:  ## Lance la démo dans l'image GPU
	docker run --rm --gpus all gmic-inference:gpu

docker-test:  ## Build l'image CPU puis lance le smoke-test complet de bout en bout
	$(MAKE) docker-build
	docker run --rm gmic-inference:cpu smoke
