app:
	uv run streamlit run src/geometric_portfolio/app.py

main:
	uv run src/geometric_portfolio/main.py

test:
	uv run pytest

.PHONY: app main