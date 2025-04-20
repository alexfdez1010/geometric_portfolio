app:
	uv run streamlit run src/geometric_portfolio/Geometric_Mean.py

main:
	uv run src/geometric_portfolio/main.py

test:
	uv run pytest

.PHONY: app main