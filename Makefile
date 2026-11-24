.PHONY: setup lint test test-all

setup:
	uv sync --locked

lint:
	uv run --locked ruff check .
	uv run --locked ruff format --check .

test:
	uv run --locked pytest -m "not slow"

test-all:
	uv run --locked pytest -o addopts=""
