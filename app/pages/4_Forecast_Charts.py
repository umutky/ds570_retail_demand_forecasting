"""Actual vs predicted time-series charts across aggregation levels on the test set."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from retail_forecast.config import REPORTS_DIR

st.title("Actual vs Predicted — Test Set")
st.caption(
    "Test period: 2015-12-18 → 2016-05-22  ·  "
    "Sales and predictions aggregated at four levels."
)

# Helpers

ACTUAL_COLOR = "#1f77b4"
PRED_COLOR   = "#d62728"

DEPT_ORDER = ["FOODS_1", "FOODS_2", "FOODS_3", "HOBBIES_1", "HOBBIES_2",
              "HOUSEHOLD_1", "HOUSEHOLD_2"]
CAT_ORDER  = ["FOODS", "HOBBIES", "HOUSEHOLD"]

# Default items: highest avg-sales representative per category
DEFAULT_ITEMS = {
    "FOODS":     "FOODS_3_090_CA_1_evaluation",
    "HOBBIES":   "HOBBIES_1_048_CA_1_evaluation",
    "HOUSEHOLD": "HOUSEHOLD_1_418_CA_1_evaluation",
}


@st.cache_data
def load_preds(model: str) -> pd.DataFrame:
    path = REPORTS_DIR / f"test_predictions_{model}.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _traces(fig, x, y_actual, y_pred, row, col, label, showlegend):
    fig.add_trace(
        go.Scatter(
            x=x, y=y_actual,
            mode="lines", name="Actual",
            line=dict(color=ACTUAL_COLOR, width=1.5),
            legendgroup="actual", showlegend=showlegend,
        ),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y_pred,
            mode="lines", name="Predicted",
            line=dict(color=PRED_COLOR, width=1.5, dash="dot"),
            legendgroup="pred", showlegend=showlegend,
        ),
        row=row, col=col,
    )


def _wmape_str(actual, pred) -> str:
    denom = actual.abs().sum()
    if denom == 0:
        return "n/a"
    return f"WMAPE {(actual - pred).abs().sum() / denom:.1%}"


# Model selector

model = st.radio("Model", ["tweedie", "gaussian"], horizontal=True, index=0,
                 format_func=str.capitalize)

df = load_preds(model)

# Tabs

tab_total, tab_cat, tab_dept, tab_item = st.tabs(
    ["Total", "Category (3)", "Department (7)", "Item (3 examples)"]
)


# Total
with tab_total:
    agg = df.groupby("date")[["sales", "y_pred"]].sum().reset_index()

    wmape_str = _wmape_str(agg["sales"], agg["y_pred"])
    st.caption(f"All 3,049 items summed daily  ·  {wmape_str}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["date"], y=agg["sales"],
        mode="lines", name="Actual",
        line=dict(color=ACTUAL_COLOR, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=agg["date"], y=agg["y_pred"],
        mode="lines", name="Predicted",
        line=dict(color=PRED_COLOR, width=2, dash="dot"),
    ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Total units sold",
        legend=dict(orientation="h", y=1.08),
        height=380, margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# Category
with tab_cat:
    cats = [c for c in CAT_ORDER if c in df["cat_id"].unique()]
    st.caption(f"{len(cats)} categories — daily total units per category")

    fig = make_subplots(
        rows=len(cats), cols=1,
        shared_xaxes=True,
        subplot_titles=cats,
        vertical_spacing=0.08,
    )
    for i, cat in enumerate(cats, 1):
        agg = (
            df[df["cat_id"] == cat]
            .groupby("date")[["sales", "y_pred"]].sum().reset_index()
        )
        wmape = _wmape_str(agg["sales"], agg["y_pred"])
        fig.layout.annotations[i - 1].text = f"{cat}  ({wmape})"
        _traces(fig, agg["date"], agg["sales"], agg["y_pred"],
                row=i, col=1, label=cat, showlegend=(i == 1))

    fig.update_layout(
        height=300 * len(cats),
        legend=dict(orientation="h", y=1.02),
        margin=dict(t=50, b=40),
    )
    fig.update_xaxes(title_text="Date", row=len(cats), col=1)
    st.plotly_chart(fig, use_container_width=True)


# Department
with tab_dept:
    depts = [d for d in DEPT_ORDER if d in df["dept_id"].unique()]
    st.caption(f"{len(depts)} departments — daily total units per department")

    fig = make_subplots(
        rows=len(depts), cols=1,
        shared_xaxes=True,
        subplot_titles=depts,
        vertical_spacing=0.05,
    )
    for i, dept in enumerate(depts, 1):
        agg = (
            df[df["dept_id"] == dept]
            .groupby("date")[["sales", "y_pred"]].sum().reset_index()
        )
        wmape = _wmape_str(agg["sales"], agg["y_pred"])
        fig.layout.annotations[i - 1].text = f"{dept}  ({wmape})"
        _traces(fig, agg["date"], agg["sales"], agg["y_pred"],
                row=i, col=1, label=dept, showlegend=(i == 1))

    fig.update_layout(
        height=240 * len(depts),
        legend=dict(orientation="h", y=1.015),
        margin=dict(t=50, b=40),
    )
    fig.update_xaxes(title_text="Date", row=len(depts), col=1)
    st.plotly_chart(fig, use_container_width=True)


#Item
with tab_item:
    st.caption("One item per category — daily actual vs predicted on the test set.")

    all_items = sorted(df["id"].unique().tolist())

    col1, col2, col3 = st.columns(3)
    selected = []
    for col_widget, cat in zip([col1, col2, col3], CAT_ORDER):
        default_item = DEFAULT_ITEMS.get(cat, all_items[0])
        idx = all_items.index(default_item) if default_item in all_items else 0
        with col_widget:
            chosen = st.selectbox(f"Item ({cat})", all_items, index=idx, key=f"item_{cat}")
            selected.append(chosen)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=selected,
        vertical_spacing=0.08,
    )
    for i, item_id in enumerate(selected, 1):
        item_df = df[df["id"] == item_id].sort_values("date")
        wmape = _wmape_str(item_df["sales"], item_df["y_pred"])
        fig.layout.annotations[i - 1].text = f"{item_id}  ({wmape})"
        _traces(fig, item_df["date"], item_df["sales"], item_df["y_pred"],
                row=i, col=1, label=item_id, showlegend=(i == 1))

    fig.update_layout(
        height=320 * 3,
        legend=dict(orientation="h", y=1.02),
        margin=dict(t=50, b=40),
    )
    fig.update_xaxes(title_text="Date", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)
