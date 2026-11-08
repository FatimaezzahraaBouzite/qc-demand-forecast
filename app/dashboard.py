"""Stakeholder dashboard: `streamlit run app/dashboard.py`.

Written for a non-technical reader: what we forecast, how good it is, is it still healthy.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Québec electricity demand forecast", layout="wide")

meta = json.loads((ROOT / "models/model_meta.json").read_text())
bt = json.loads((ROOT / "reports/backtest_metrics.json").read_text())
mon_path = ROOT / "reports/monitoring.json"
mon = json.loads(mon_path.read_text()) if mon_path.exists() else {}

st.title("Tomorrow's electricity demand in Québec")
st.caption("Day-ahead forecast of average daily demand (MW). Data: Hydro-Québec open data "
           "(CC-BY-NC 4.0) and Open-Meteo.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model in production", meta["model"])
c2.metric("Typical error (backtest)", f"{meta['backtest_mean'][meta['model']]:.1f} %")
c3.metric("Better than 'same day last week'", f"{meta['skill_vs_naive_pct']:.0f} %")
c4.metric("Health", "✅ OK" if mon.get("status") == "ok" else f"⚠️ {mon.get('status', 'n/a')}")
for a in mon.get("alerts", []):
    st.warning(a)

st.subheader("Forecast vs actual")
ts_path = ROOT / "reports/monitoring_timeseries.csv"
if ts_path.exists():
    ts = pd.read_csv(ts_path, parse_dates=["date"])
    fig = go.Figure()
    fig.add_scatter(x=ts["date"], y=ts["demand_mw"], name="Actual", mode="lines")
    fig.add_scatter(x=ts["date"], y=ts["pred_mw"], name="Forecast", mode="lines",
                    line={"dash": "dot"})
    fig.update_layout(yaxis_title="MW", height=380, margin={"t": 10})
    st.plotly_chart(fig, use_container_width=True)
    st.line_chart(ts.set_index("date")["rolling_mape"], height=200)
    st.caption("Rolling 14-day error (%). An alert fires when it exceeds 1.5x the backtest error.")

st.subheader("How the candidate models compared")
st.caption("Each model forecast four held-out quarters, one day ahead at a time.")
st.dataframe(pd.DataFrame(bt["summary"]).T.rename(columns=str.upper).round(2))

imp = ROOT / "reports/feature_importance.csv"
if imp.exists():
    st.subheader("What drives the forecast")
    st.bar_chart(pd.read_csv(imp, index_col=0).head(10))
