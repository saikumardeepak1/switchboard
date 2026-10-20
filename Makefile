.PHONY: check bench serve

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest

bench:
	uv run python -m benchmarks.run_all

serve:
	uv run --extra api python -m benchmarks.serve
