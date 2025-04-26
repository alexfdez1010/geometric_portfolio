app:
	uv run streamlit run src/geometric_portfolio/Geometric_Mean.py

test:
	uv run pytest

lint:
	uv run ruff check

format:
	uv run ruff format

pre-commit: test lint format

.PHONY: app main