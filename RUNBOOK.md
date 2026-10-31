# Runbook

## Dashboard (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/run_pipeline.py            # validates that everything the dashboard needs exists and works
python -m streamlit run dashboard/app.py
```

## Tests
```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Regenerate artifacts
| What | Command | Output |
|---|---|---|
| Clusters, evaluation, ablation, data-quality experiment | `python -m src.build_artifacts` | `artifacts/recommendation/` |
| Check that published artifacts match the code | `python -m src.build_artifacts --check` | exit code 0 / 1 |
| Graph (edge list, centrality, PageRank) | run notebook `05_graph_analytics_pagerank.ipynb` | `artifacts/graph/` |
| EDA, PCA and K-means figures | notebooks `01`, `02`, `03` | `reports/` |

The notebooks need `python -m pip install -r requirements-notebooks.txt` and are run from the `notebooks/` folder.

## New data from GitHub
```powershell
copy .env.example .env     # then put your token in .env
python src/ingest.py
```
By default `ingest.py` writes to `data/snapshots/<date>/` and does **not** touch `data/processed/repos_clean.csv`
(the official snapshot all metrics were computed on). Use `--promote` only if you want to replace it, then run
`python -m src.build_artifacts` and notebook 05.
