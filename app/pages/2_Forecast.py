import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from retail_forecast.config import PROCESSED_DATA_DIR, REPORTS_DIR

st.title("28-Day Demand Forecast")
st.caption(
    "Recursive forecast using the LightGBM Tweedie model. "
    "Lag features from the training period seed the first steps; "
    "predictions from earlier steps feed back as lags for later steps."
)

FORECAST_PATH = REPORTS_DIR / "forecast_28d.parquet"
SALES_PATH = PROCESSED_DATA_DIR / "sales_long.parquet"

if not FORECAST_PATH.exists():
    st.error("Forecast file not found. Run `rf-predict` first.")
    st.stop()

if not SALES_PATH.exists():
    st.error("Processed sales data not found. Run `rf-process` first.")
    st.stop()


@st.cache_data
def load_forecast() -> pd.DataFrame:
    return pd.read_parquet(FORECAST_PATH)


@st.cache_data
def load_recent_sales(lookback_days: int = 90) -> pd.DataFrame:
    df = pd.read_parquet(SALES_PATH, columns=["id", "date", "sales", "cat_id", "dept_id"])
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
    return df[df["date"] >= cutoff]


forecast = load_forecast()
forecast["date"] = pd.to_datetime(forecast["date"])

history = load_recent_sales()

forecast_start = forecast["date"].min()
forecast_end = forecast["date"].max()

st.info(
    f"Forecast period: **{forecast_start.date()}** to **{forecast_end.date()}**  "
    f"({len(forecast['date'].unique())} days, {forecast['id'].nunique():,} items)"
)


# Category-level summary
st.subheader("Total forecast demand by category")
st.caption("Sum of predicted daily units across all items in each category over the 28-day window.")

cat_summary = (
    forecast.groupby("cat_id")["y_pred"]
    .sum()
    .reset_index()
    .rename(columns={"y_pred": "total_units"})
    .sort_values("total_units", ascending=False)
)
cat_summary["total_units"] = cat_summary["total_units"].round(0).astype(int)

fig_cat = px.bar(
    cat_summary,
    x="cat_id",
    y="total_units",
    labels={"cat_id": "Category", "total_units": "Predicted units (28 days)"},
    text_auto=True,
    color="cat_id",
    color_discrete_map={"FOODS": "#1f77b4", "HOUSEHOLD": "#ff7f0e", "HOBBIES": "#2ca02c"},
)
fig_cat.update_layout(showlegend=False, height=320, margin=dict(t=20, b=40))
st.plotly_chart(fig_cat, use_container_width=True)


# Department-level breakdown                       
st.divider()
st.subheader("Forecast by department")

dept_summary = (
    forecast.groupby(["cat_id", "dept_id"])["y_pred"]
    .sum()
    .reset_index()
    .rename(columns={"y_pred": "total_units"})
    .sort_values(["cat_id", "total_units"], ascending=[True, False])
)
dept_summary["total_units"] = dept_summary["total_units"].round(0).astype(int)

fig_dept = px.bar(
    dept_summary,
    x="dept_id",
    y="total_units",
    color="cat_id",
    labels={"dept_id": "Department", "total_units": "Predicted units (28 days)", "cat_id": "Category"},
    text_auto=True,
    color_discrete_map={"FOODS": "#1f77b4", "HOUSEHOLD": "#ff7f0e", "HOBBIES": "#2ca02c"},
)
fig_dept.update_layout(height=350, margin=dict(t=20, b=40))
st.plotly_chart(fig_dept, use_container_width=True)


# Item-level forecast    
st.divider()
st.subheader("Item-level forecast")
st.caption("Last 90 days of historical sales plus the 28-day forecast.")

col1, col2 = st.columns([1, 3])
with col1:
    categories = sorted(forecast["cat_id"].unique().tolist()) if "cat_id" in forecast.columns else []
    selected_cat = st.selectbox("Category", categories)

    dept_options = sorted(
        forecast[forecast["cat_id"] == selected_cat]["dept_id"].unique().tolist()
    ) if "dept_id" in forecast.columns else []
    selected_dept = st.selectbox("Department", dept_options)

    items = sorted(forecast[forecast["dept_id"] == selected_dept]["id"].unique().tolist())
    selected_item = st.selectbox("Item", items)

with col2:
    item_hist = (
        history[history["id"] == selected_item]
        .sort_values("date")
        [["date", "sales"]]
    )
    item_fc = (
        forecast[forecast["id"] == selected_item]
        .sort_values("date")
        [["date", "y_pred"]]
    )

    fig_item = go.Figure()
    fig_item.add_trace(
        go.Scatter(
            x=item_hist["date"],
            y=item_hist["sales"],
            mode="lines",
            name="Historical sales",
            line=dict(color="#1f77b4", width=1.5),
        )
    )
    fig_item.add_trace(
        go.Scatter(
            x=item_fc["date"],
            y=item_fc["y_pred"],
            mode="lines",
            name="Forecast",
            line=dict(color="#d62728", width=2, dash="dot"),
        )
    )
    fig_item.add_vline(
        x=forecast_start.timestamp() * 1000,
        line_dash="dash",
        line_color="grey",
        annotation_text="Forecast start",
        annotation_position="top right",
    )
    fig_item.update_layout(
        xaxis_title="Date",
        yaxis_title="Units sold",
        legend=dict(orientation="h", y=1.05),
        height=350,
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig_item, use_container_width=True)

    item_total = item_fc["y_pred"].sum()
    st.metric(
        label=f"Total forecast demand — {selected_item} (28 days)",
        value=f"{item_total:.1f} units",
    )


# Top 20 items by forecast demand                     
st.divider()
st.subheader("Top 20 items by 28-day forecast demand")

top20 = (
    forecast.groupby(["id", "cat_id"])["y_pred"]
    .sum()
    .reset_index()
    .rename(columns={"y_pred": "total_units"})
    .sort_values("total_units", ascending=False)
    .head(20)
)

fig_top = px.bar(
    top20,
    x="total_units",
    y="id",
    orientation="h",
    color="cat_id",
    labels={"id": "Item", "total_units": "Predicted units (28 days)", "cat_id": "Category"},
    color_discrete_map={"FOODS": "#1f77b4", "HOUSEHOLD": "#ff7f0e", "HOBBIES": "#2ca02c"},
)
fig_top.update_layout(
    yaxis=dict(autorange="reversed"),
    height=500,
    margin=dict(t=20, b=40),
)
st.plotly_chart(fig_top, use_container_width=True)
