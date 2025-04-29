app:
	uv run streamlit run src/geometric_portfolio/app.py

test:
	uv run pytest

lint:
	uv run ruff check

format:
	uv run ruff format

pre-commit: test lint format

.PHONY: app main