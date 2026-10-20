.PHONY: check demo bench serve

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest

demo:
	uv run python -m examples.demo

bench:
	uv run python -m benchmarks.run_all

serve:
	uv run --extra api python -m benchmarks.serve
