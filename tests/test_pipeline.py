import json

import numpy as np
import pandas as pd
import pytest

from qc_demand import config
from qc_demand.backtest import make_folds
from qc_demand.features import TARGET, build_features, feature_columns, model_ready
from qc_demand.ingest import hourly_to_daily, parse_hq_xlsx, validate_daily
from qc_demand.models import metrics
from qc_demand.monitor import psi, seasonal_reference
from qc_demand.synthetic import make_daily


@pytest.fixture
def params():
    return config.load_params()


# ------------------------------------------------------------------ ingestion
def test_parse_hq_xlsx_handles_header_offset_and_hour_ending(tmp_path):
    ts = pd.date_range("2023-01-01 01:00", periods=48, freq="h")
    body = pd.DataFrame({"Date": ts, "Demande (MW)": np.arange(48) + 20000.0})
    path = tmp_path / "hq.xlsx"
    with pd.ExcelWriter(path) as xw:  # two junk title rows above the header
        pd.DataFrame([["Historique de la demande"], [""]]).to_excel(
            xw, index=False, header=False)
        body.to_excel(xw, index=False, startrow=2)
    hourly = parse_hq_xlsx(path)
    assert list(hourly.columns) == ["timestamp", "demand_mw"]
    assert hourly["timestamp"].iloc[0] == pd.Timestamp("2023-01-01 00:00")  # shifted
    daily = hourly_to_daily(hourly)
    assert len(daily) == 2 and daily["demand_mw"].notna().all()


def test_hourly_to_daily_masks_incomplete_days():
    ts = pd.date_range("2023-01-01", periods=30, freq="h")  # day 2 has 6 hours
    daily = hourly_to_daily(pd.DataFrame({"timestamp": ts, "demand_mw": 20000.0}))
    assert daily["demand_mw"].isna().sum() == 1


def test_validate_rejects_bad_units():
    df = make_daily("2022-01-01", "2022-03-01")
    df["demand_mw"] /= 1000  # GW instead of MW
    with pytest.raises(ValueError):
        validate_daily(df)


# ------------------------------------------------------------------ features
def test_features_have_no_target_leakage(params):
    df = make_daily("2021-01-01", "2021-12-31")
    feats = build_features(df, params)
    # changing today's demand must not change any of today's features
    df2 = df.copy()
    day = pd.Timestamp("2021-06-15")
    df2.loc[day, TARGET] += 10_000
    feats2 = build_features(df2, params)
    cols = feature_columns(params)
    pd.testing.assert_series_equal(feats.loc[day, cols], feats2.loc[day, cols])


def test_model_ready_drops_warmup(params):
    feats = model_ready(build_features(make_daily("2021-01-01", "2021-12-31"), params), params)
    assert feats[feature_columns(params)].notna().all().all()
    assert feats.index.min() >= pd.Timestamp("2021-01-15")


# ------------------------------------------------------------------ evaluation
def test_folds_are_ordered_and_disjoint():
    idx = pd.date_range("2019-01-01", periods=1000, freq="D")
    folds = make_folds(idx, n_folds=3, test_days=50, min_train_days=500)
    for train, test in folds:
        assert train.max() < test.min()
    assert folds[-1][1].max() == idx.max()


def test_metrics_and_psi():
    y = pd.Series([100.0, 200.0])
    m = metrics(y, pd.Series([110.0, 180.0]))
    assert m["mape"] == pytest.approx(10.0)
    assert m["bias"] == pytest.approx(-5.0)
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 500)) < 0.1
    assert psi(ref, rng.normal(3, 1, 500)) > 0.5


# ------------------------------------------------------------------ end to end
def test_end_to_end_train_replay_monitor(workspace):
    _, params = workspace
    from qc_demand import ingest, monitor, predict, train

    ingest.run(params, synthetic=True)
    meta = train.run(params, until="2023-06-30")
    assert meta["model"] in {"seasonal_naive", "sarimax", "lightgbm"}
    # the selected model must beat the naive baseline on this signal
    assert meta["backtest_mean"][meta["model"]] < meta["backtest_mean"]["seasonal_naive"]
    assert meta["trained_until"] == "2023-06-30"

    log = predict.replay(params, "2023-07-01", "2023-07-31")
    assert len(log) == 31 and log["pred_mw"].notna().all()

    report = monitor.run(params)
    assert report["n_evaluated"] == 31
    assert report["status"] in {"ok", "alert"}
    assert json.loads(config.MONITOR_FILE.read_text())["model"] == meta["model"]


def test_seasonal_reference_wraps_year_end():
    idx = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    ref = seasonal_reference(pd.Series(1.0, idx), pd.date_range("2023-12-25", "2023-12-31"))
    months = set(ref.index.month)
    assert months <= {12, 1} and 1 in months


def test_monitor_flags_cold_snap(workspace):
    _, params = workspace
    from qc_demand import ingest, monitor, predict, train

    df = ingest.run(params, synthetic=True)
    snap = pd.date_range("2023-07-01", "2023-07-14")
    df.loc[snap, ["temp_mean", "temp_max", "temp_min"]] -= 25  # absurd July cold snap
    df.to_parquet(config.DAILY_FILE)
    train.run(params, until="2023-06-30")
    predict.replay(params, "2023-07-01", "2023-07-14")
    report = monitor.run(params)
    assert report["status"] == "alert"
    assert any("PSI" in a for a in report["alerts"])


@pytest.mark.parametrize("layout", ["odd_labels", "date_plus_hour", "text_french_numbers",
                                    "second_sheet"])
def test_parse_hq_xlsx_finds_columns_by_content(tmp_path, layout):
    ts = pd.date_range("2023-01-01 01:00", periods=48, freq="h")
    mw = np.arange(48) * 10 + 20000.5
    if layout == "odd_labels":
        body = pd.DataFrame({"Date et heure": ts, "Moyenne horaire (MW)": mw})
    elif layout == "date_plus_hour":
        body = pd.DataFrame({"Jour": ts.normalize() - pd.to_timedelta((ts.hour == 0) * 24, "h"),
                             "Heure": np.where(ts.hour == 0, 24, ts.hour),
                             "Demande totale": mw})
    elif layout == "text_french_numbers":
        body = pd.DataFrame({"Date": ts.strftime("%Y-%m-%d %H:%M"),
                             "Demande (MW)": [f"{v:,.1f}".replace(",", " ").replace(".", ",")
                                              for v in mw]})
    else:
        body = pd.DataFrame({"Date": ts, "Demande (MW)": mw})
    path = tmp_path / "hq.xlsx"
    with pd.ExcelWriter(path) as xw:
        if layout == "second_sheet":
            pd.DataFrame([["Notes"], ["Source: Hydro-Québec"]]).to_excel(
                xw, sheet_name="Lisez-moi", index=False, header=False)
        pd.DataFrame([["Historique de la demande"]]).to_excel(
            xw, sheet_name="Données", index=False, header=False)
        body.to_excel(xw, sheet_name="Données", index=False, startrow=2)
    hourly = parse_hq_xlsx(path)
    assert len(hourly) == 48
    assert hourly["timestamp"].iloc[0] == pd.Timestamp("2023-01-01 00:00")
    assert hourly["demand_mw"].iloc[0] == pytest.approx(20000.5)


def test_parse_hq_real_layout_text_dates_are_day_first(tmp_path):
    """Real 2019 file: 'Date' | 'Moyenne (MW)', dates like 05/02/2019 01:00 (5 February)."""
    ts = pd.date_range("2019-02-05 01:00", periods=48, freq="h")
    body = pd.DataFrame({"Date": ts.strftime("%d/%m/%Y %H:%M"),
                         "Moyenne (MW)": np.linspace(23762.55, 24190.08, 48)})
    path = tmp_path / "hq.xlsx"
    body.to_excel(path, index=False)
    hourly = parse_hq_xlsx(path)
    assert hourly["timestamp"].iloc[0] == pd.Timestamp("2019-02-05 00:00")
    assert hourly["timestamp"].dt.month.eq(2).all()
    assert hourly["demand_mw"].iloc[0] == pytest.approx(23762.55)


def test_parse_hq_real_layout_excel_datetimes(tmp_path):
    ts = pd.date_range("2019-01-01 01:00", periods=48, freq="h")
    path = tmp_path / "hq.xlsx"
    pd.DataFrame({"Date": ts, "Moyenne (MW)": 23762.55}).to_excel(path, index=False)
    hourly = parse_hq_xlsx(path)
    assert len(hourly) == 48 and hourly["timestamp"].iloc[0] == pd.Timestamp("2019-01-01")


def test_lightgbm_change_target_follows_level_shift(params):
    """Trained before a +3 GW step, the change-target model must follow it; level model can't."""
    from qc_demand.models import LightGBMModel

    df = make_daily("2019-01-01", "2022-12-31")
    df.loc["2022-01-01":, TARGET] += 3000
    frame = model_ready(build_features(df, params), params)
    train, test_idx = frame.loc[:"2021-12-31"], frame.loc["2022-03-01":"2022-04-30"].index
    hyper = dict(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=0)
    level = LightGBMModel(params, **hyper).fit(train).predict(frame, test_idx)
    change = LightGBMModel(params, target_base="lag_1", **hyper).fit(train).predict(frame, test_idx)
    actual = frame.loc[test_idx, TARGET]
    assert abs((change - actual).mean()) < abs((level - actual).mean()) / 5