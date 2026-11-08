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

---

# Weekend plan (≈ 14 h)

| Block | Time | Outcome |
|---|---|---|
| **Saturday** | | |
| S1 Setup + green tests | 0:45 | Repo on GitHub, tests pass locally |
| S2 Real data ingest + quick EDA | 1:30 | `daily.parquet` from real HQ data, 3 EDA plots |
| S3 Backtest + model iteration in MLflow | 2:30 | Selected model, results table, 2-3 documented experiments |
| S4 DVC with S3 remote | 1:00 | `dvc repro` / `dvc push` work |
| S5 CI on GitHub Actions | 0:45 | Green badge |
| **Sunday** | | |
| U1 AWS deployment (SAM) | 2:30 | Lambda runs daily, writes forecasts to S3, alert email confirmed |
| U2 Replay + monitoring + dashboard | 1:30 | 6 months of "production" forecasts, monitoring report, dashboard screenshot |
| U3 Documentation + resume | 1:30 | Results in README, limitations, resume bullets with real numbers |
| U4 Buffer | 1:00 | For whatever broke |

Prerequisites (do these Friday night, 20 min): GitHub account, AWS account with the free
tier, [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
and [SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
installed, Docker Desktop running, Python 3.11.

## Saturday

### S1 Setup (45 min)

```bash
cd qc-demand-forecast
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app,aws]" "dvc[s3]"
pytest -q                       # 10 passed
git init && git add . && git commit -m "Initial pipeline"
gh repo create qc-demand-forecast --public --source . --push   # or create it on github.com
```

### S2 Real data (1 h 30)

```bash
qc-demand ingest
```

This downloads the five yearly XLSX files and four cities of weather, then prints the
data-quality report. Check:

- ~1,826 days, 2019-01-01 → 2023-12-31, a handful of missing days at most;
- if parsing fails, open one XLSX: `parse_hq_xlsx` finds the header row by the "Date" /
  "Demande" labels, so only an unexpected label change needs a fix.

Quick EDA in a notebook (`notebooks/01_eda.ipynb`, 30-40 min). Make exactly these three plots;
they become the "why" section of your README and talking points in the interview:

1. Daily demand vs temperature (scatter): a hockey stick, strongly rising below ~15 °C. That
   is why HDD is a feature and why trees may beat a linear model.
2. Average demand by weekday: the weekend drop.
3. Demand over 2019-2023: the 2020 COVID dip, winter peaks.

### S3 Backtest and models (2 h 30)

```bash
qc-demand train --until 2023-06-30     # holds back H2 2023 as the "production" period
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

Training takes about a minute. In MLflow, open the parent run: one nested run per model,
with MAPE per fold as a step chart.

Then run 2-3 deliberate experiments, one change each, and keep a note of the result
(this is your "rigour" story):

- HDD base temperature 18 → 15 °C (`params.yaml: features.hdd_base_c`);
- add a 365-day lag or remove `temp_mean_lag1`;
- LightGBM `num_leaves` 31 → 15 and `n_estimators` 600 → 1,200 at lr 0.02.

Pick the final config, rerun, and copy the summary table from
`reports/backtest_metrics.json` into the Results section below.

Don't over-tune: four folds is a small test set, and a 0.1-point MAPE difference is noise.
Saying that in an interview scores better than a lower number.

### S4 DVC (1 h)

```bash
aws configure                                  # region ca-central-1
aws s3 mb s3://qc-demand-<your-initials>       # remote for data/models
dvc init
dvc remote add -d storage s3://qc-demand-<your-initials>/dvc
dvc repro            # runs ingest → train → replay → monitor, writes dvc.lock
dvc push
git add dvc.yaml dvc.lock .dvc params.yaml && git commit -m "DVC pipeline" && git push
```

Prove reproducibility: change `backtest.test_days` to 60, run `dvc repro` (only train and
downstream rerun), then `dvc params diff` and `dvc metrics diff`. Revert.

Note: this bucket is only for DVC. The SAM stack creates its own bucket on Sunday.

### S5 CI (45 min)

The push already triggered `.github/workflows/ci.yml`. In the Actions tab, confirm that lint,
tests, the CLI smoke run and the Docker build pass. Add the badge to the top of this README:

```markdown
![CI](https://github.com/<you>/qc-demand-forecast/actions/workflows/ci.yml/badge.svg)
```

Leave `retrain.yml` for later: it needs an AWS OIDC role (stretch goal below).

## Sunday

### U1 Deploy to AWS (2 h 30)

```bash
cd deploy
sam build --template template.yaml
sam deploy --guided          # stack: qc-demand, region: ca-central-1, AlertEmail: yours
                             # say yes to "create managed ECR repo"
```

Confirm the SNS subscription email you receive. Then seed the bucket (name is in the
stack outputs):

```bash
B=$(aws cloudformation describe-stacks --stack-name qc-demand \
     --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text)
aws s3 cp ../models/model.joblib     s3://$B/models/model.joblib
aws s3 cp ../models/model_meta.json  s3://$B/models/model_meta.json
aws s3 cp ../data/processed/daily.parquet s3://$B/data/daily.parquet
sam remote invoke ForecastFunction --stack-name qc-demand --event '{"mode":"replay-next"}'
aws s3 cp s3://$B/forecasts/forecasts.csv - | tail -3
```

How the daily job works: at 06:23 Montréal time EventBridge invokes the Lambda, which pulls
the model and history from S3, forecasts the next day, appends to `forecasts.csv`, recomputes
monitoring, pushes everything back, and emails you if monitoring raises an alert. In
`replay-next` mode it walks through H2 2023 one day per run, so you have a genuinely running
production pipeline whose forecasts can be scored against real actuals.

If something fails: `sam logs -n ForecastFunction --stack-name qc-demand --tail`. The most
likely causes are a missing S3 object, or a timeout on the first cold start of a large
image (raise `Timeout` in `template.yaml`).

Cost: Lambda, S3 and SNS at this volume stay inside the free tier. Run
`sam delete --stack-name qc-demand` when you no longer need it.

Optional: test the image locally before deploying with
`docker build -t qc-demand . && docker run -p 9000:8080 -e BUCKET=$B -e AWS_REGION=ca-central-1 -v ~/.aws:/root/.aws qc-demand`,
then `curl -XPOST localhost:9000/2015-03-31/functions/function/invocations -d '{}'`.

### U2 Monitoring and dashboard (1 h 30)

```bash
qc-demand predict --mode replay --start 2023-07-01 --end 2023-12-31
qc-demand monitor
streamlit run app/dashboard.py
```

Look at the error in December. If an alert fires, investigate it rather than hiding it: a
cold snap or the holidays are a great interview story ("monitoring caught X; here is what
I would do").

Take a screenshot of the dashboard and save it to `docs/dashboard.png` for the README.

### U3 Documentation and resume (1 h 30)

Fill in the Results and Decisions sections below with your real numbers. Spend most of the
time on the "Limitations" paragraph: it is what a consulting interviewer will probe.

### U4 Buffer (1 h)

If everything worked, pick one stretch goal.

## Stretch goals (later, one evening each)

- Prediction intervals: LightGBM quantile objective (0.1/0.9) or SARIMAX
  `get_prediction().conf_int()`, and evaluate coverage. Very relevant for capacity planning.
- Hourly model and daily peak forecasting (`peak_mw` is already in the table).
- Monthly retrain via `retrain.yml`: create a GitHub OIDC role in IAM, add the `AWS_ROLE_ARN`
  and `BUCKET` secrets.
- `live` mode: open the HQ real-time JSON in a browser, check `parse_hq_realtime` against its
  structure, then switch the schedule input to `{"mode": "live"}`.
- SQL: load `forecasts.csv` into Athena or DuckDB and write the monitoring query in SQL.

---

# Results

*(fill in after S3; replace the placeholders)*

| Model | Backtest MAPE (mean ± sd) | MAE (MW) | Bias (MW) |
|---|---|---|---|
| Seasonal naive | x.xx ± x.xx | | |
| SARIMAX + weather | x.xx ± x.xx | | |
| LightGBM | x.xx ± x.xx | | |

Selected: **___**, ___ % lower MAPE than the seasonal naive baseline. Production replay on
H2 2023 (never seen in training): MAPE ___ %.

# Decisions and limitations

- **Daily mean, day-ahead.** Simplest target that is useful for energy planning; hourly and
  peak forecasting are natural extensions.
- **Rolling-origin backtest** instead of a random split: time series must be validated
  forward in time, and four quarterly folds show performance in every season.
- **Seasonal naive baseline** is mandatory: a model that cannot beat "same day last week"
  adds no value.
- **Replay uses observed temperature**, while live forecasts use forecast weather. Replay
  accuracy is therefore optimistic; weather forecast error adds to real-world error,
  especially in winter.
- **Drift check is season-aware:** production temperatures are compared with the same time
  of year in training, otherwise every winter would raise a false alarm.
- **Data.** HQ open data is real-time operational data, not the official audited figures,
  and its history currently ends in 2023. Licence CC-BY-NC 4.0: fine for a portfolio, not
  for commercial use.

---

# Resume bullets (update numbers after the weekend)

> **Québec Electricity Demand Forecasting Pipeline** | Python, LightGBM, statsmodels, MLflow, DVC, AWS, GitHub Actions — 2026
> • Forecast day-ahead Québec electricity demand from Hydro-Québec open data and weighted
>   weather, benchmarking SARIMAX and LightGBM against a seasonal-naive baseline with
>   rolling-origin backtesting (__ % MAPE, __ % better than baseline).
> • Built a reproducible MLOps pipeline (MLflow tracking, DVC versioning, CI/CD with tests),
>   deployed as a scheduled AWS Lambda with accuracy and drift monitoring and email alerts.

# Interview talking points

1. Why the baseline matters, and how much each model beat it.
2. How you prevented leakage (lags ≥ 1, the leakage unit test, holding out H2 2023).
3. Why SARIMAX needed standardized inputs (demand ≈ 20,000 MW led to degenerate fits where
   AR and MA terms cancelled). Debugging story with a clear fix.
4. Why the drift check compares the same season, and what you would do when it fires.
5. What you would change for a client: hourly granularity, intervals, a feature store, model
   registry stages in MLflow, Terraform instead of SAM if they already use it.

Data: Hydro-Québec, *Historique de la demande d'électricité au Québec*, CC-BY-NC 4.0;
Open-Meteo historical weather API (ERA5), CC BY 4.0.
