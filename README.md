# FinSight AI

A Streamlit portfolio app by Hajar Mrifag for historical banking disengagement research.

Explore reproduced model results, monthly cohorts, source-backed search, and an explicit campaign scenario calculator. Public exports contain aggregate historical results only; no individual account records or fitted risk model are included.

## Run locally

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Semantic search downloads MiniLM on first use and caches it. If loading fails, the interface explicitly falls back to word matching. Quick evidence answers and all analytics work without a model download. No external LLM API key is needed.

## Deploy on Streamlit Community Cloud

1. Push this directory to a repository owned by your GitHub account.
2. Sign in at https://share.streamlit.io and connect that repository.
3. Choose the repository, `main` branch, and `app.py` entrypoint.
4. In advanced settings choose Python 3.11, supported by these dependency pins.
5. Choose an available app subdomain and deploy.

The resulting link ends in `streamlit.app`. Dependency loading and semantic model initialization can take a few minutes on a cold start.

## Research evidence

The fixed original model reproduces 0.9227 ROC-AUC, 0.1983 average precision, and 7.08× top-decile lift. It captures 141 of 199 observed disengagement cases (70.85%) in the top 10% of 13,177 held-out account-months.

The target is a severe relative decline in next-90-day customer-initiated transaction activity, not observed account closure or credit default. Historical cohort results are not current banking predictions. Adjacent outcome windows overlap temporal boundaries; strict purged validation would add a robustness check. Campaign economics are assumptions, not measured treatment effects.

Source analysis: https://github.com/hajarmrifag/churn-reactivation-engine
Dataset: https://relational.fel.cvut.cz/dataset/Financial

## Tests

```sh
python -m pip install pytest
python -m pytest -q
```

Tests simulate navigation, evidence answers, validation, and scenario input through Streamlit's AppTest framework. Core calculation and retrieval logic is tested separately.
