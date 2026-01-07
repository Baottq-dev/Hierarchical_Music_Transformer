.PHONY: install install-dev install-mamba train test lint format clean docker-build docker-run

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-mamba:
	pip install causal-conv1d>=1.2.0 mamba-ssm

# Training
train:
	python train.py

train-baseline:
	python train.py model=transformer

train-hybrid:
	python train.py model=hybrid

# Evaluation
evaluate:
	python evaluate.py --checkpoint experiments/checkpoints/best_model.pt

generate:
	python generate.py --checkpoint experiments/checkpoints/best_model.pt

# Testing
test:
	pytest tests/ -v --cov=mamba_music

test-unit:
	pytest tests/ -v -k "not integration"

# Linting
lint:
	ruff check mamba_music/
	black --check mamba_music/

format:
	black mamba_music/ train.py evaluate.py generate.py
	isort mamba_music/ train.py evaluate.py generate.py

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf experiments/logs/*

clean-all: clean
	rm -rf experiments/checkpoints/*
	rm -rf experiments/generated/*

# Docker
docker-build:
	docker build -t mamba-music -f docker/Dockerfile .

docker-run:
	docker run --gpus all -it -v $(PWD):/app mamba-music bash

docker-train:
	docker run --gpus all -v $(PWD):/app mamba-music python train.py

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

# Help
help:
	@echo "Available targets:"
	@echo "  install        - Install package"
	@echo "  install-dev    - Install with dev dependencies"
	@echo "  install-mamba  - Install Mamba dependencies"
	@echo "  train          - Train model"
	@echo "  evaluate       - Evaluate model"
	@echo "  generate       - Generate music"
	@echo "  test           - Run tests"
	@echo "  lint           - Run linters"
	@echo "  format         - Format code"
	@echo "  clean          - Clean cache files"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"
