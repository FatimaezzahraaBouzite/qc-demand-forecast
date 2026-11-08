# Québec Electricity Demand Forecasting: end-to-end MLOps pipeline

Day-ahead forecast of Québec's average daily electricity demand (MW) from Hydro-Québec open
data and population-weighted weather. The model is selected by rolling-origin backtest,
tracked in MLflow, versioned with DVC, tested in CI, deployed as a scheduled AWS Lambda,
and monitored for accuracy decay and input drift.

```
 Hydro-Québec XLSX ─┐                                ┌─> MLflow (runs, metrics, model)
 Open-Meteo ERA5  ──┼─> ingest ─> features ─> backtest ─> train (select + refit) ─> model.joblib
                    │   (quality checks)     (4 rolling folds, 3 models)              │
                    │                                                      DVC  ──> S3
 EventBridge (daily) ──> Lambda container ──> predict ──> forecasts.csv ──> monitor ──> SNS alert
                                                          (S3, versioned)   (MAPE + PSI drift)
 GitHub Actions: lint + tests + smoke pipeline + Docker build on every push; monthly retrain
 Streamlit: stakeholder dashboard (forecast vs actual, error trend, model comparison)
```

## What's in the repo

| Path | What it does |
|---|---|
| `src/qc_demand/ingest.py` | Downloads HQ hourly XLSX (2019-2023), fixes hour-ending timestamps, aggregates to daily, joins weighted weather (Montréal 60%, Québec 20%, Gatineau 10%, Sherbrooke 10%), runs data-quality checks |
| `src/qc_demand/features.py` | Calendar (QC holidays, Christmas break), HDD/CDD, lags 1/2/7/14, rolling stats. Tested for target leakage |
| `src/qc_demand/models.py` | Seasonal naive (baseline), SARIMAX with weather regressors (statistical), LightGBM (ML), all behind one `fit` / `predict` interface |
| `src/qc_demand/backtest.py` | Rolling-origin, expanding-window backtest: 4 folds of 91 days, one-step-ahead |
| `src/qc_demand/train.py` | Backtest → MLflow nested runs → select best on MAPE → refit on all data → save model + metadata |
| `src/qc_demand/predict.py` | `replay` (simulate production on held-out data) and `live` (tomorrow, with forecast weather) |
| `src/qc_demand/monitor.py` | Rolling MAPE vs backtest MAPE, season-aware PSI on temperature, alert status |
| `tests/` | 10 tests incl. leakage check, fold integrity, drift detection, full end-to-end run on synthetic data |
| `dvc.yaml`, `params.yaml` | Reproducible pipeline: `dvc repro` reruns only what changed |
| `.github/workflows/` | `ci.yml` (every push), `retrain.yml` (monthly, OIDC to AWS) |
| `Dockerfile`, `deploy/` | Lambda container image, SAM template (S3 + Lambda + daily schedule + SNS) |
| `app/dashboard.py` | Streamlit dashboard for non-technical stakeholders |

Everything already runs end-to-end on synthetic data (that's what CI uses), so you are never
debugging pipeline plumbing and real data at the same time.
