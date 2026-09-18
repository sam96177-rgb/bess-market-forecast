.PHONY: ingest backtest app test all

# Requires GNU Make. If it's not installed (e.g. plain Windows without
# MinGW/choco), run the three commands below directly with `python`
# instead — see README.md.

ingest:
	python scripts/run_ingest.py

backtest:
	python scripts/run_backtest.py

app:
	streamlit run app/streamlit_app.py

test:
	pytest tests/ -q

all: ingest backtest app
