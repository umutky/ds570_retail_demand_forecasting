import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from retail_forecast.config import PROCESSED_DATA_DIR

st.title("Data Explorer")
st.caption("Historical sales from the M5 dataset — CA_1 store, all products.")


@st.cache_data
def load_sales() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DATA_DIR / "sales_long.parquet")


sales_path = PROCESSED_DATA_DIR / "sales_long.parquet"
if not sales_path.exists():
    st.error("Processed data not found. Run `rf-process` first.")
    st.stop()

df = load_sales()
df["date"] = pd.to_datetime(df["date"])


# Filters
st.sidebar.header("Filters")

categories = sorted(df["cat_id"].unique().tolist())
selected_cat = st.sidebar.selectbox("Category", categories)

depts = sorted(df[df["cat_id"] == selected_cat]["dept_id"].unique().tolist())
selected_dept = st.sidebar.selectbox("Department", depts)

items = sorted(df[df["dept_id"] == selected_dept]["id"].unique().tolist())
selected_item = st.sidebar.selectbox("Item", items)


# Item time series
st.subheader(f"Sales time series — {selected_item}")

item_df = (
    df[df["id"] == selected_item]
    .sort_values("date")
    [["date", "sales", "event_name_1", "event_type_1", "sell_price"]]
)

EVENT_COLORS = {
    "Sporting":  "#e377c2",
    "National":  "#d62728",
    "Cultural":  "#ff7f0e",
    "Religious": "#9467bd",
}
EVENT_SYMBOLS = {
    "Sporting":  "star",
    "National":  "x",
    "Cultural":  "diamond",
    "Religious": "cross",
}

event_rows = item_df[item_df["event_type_1"].notna()]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=item_df["date"],
        y=item_df["sales"],
        mode="lines",
        name="Daily sales",
        line=dict(color="#1f77b4", width=1),
    )
)
for etype, color in EVENT_COLORS.items():
    subset = event_rows[event_rows["event_type_1"] == etype]
    if subset.empty:
        continue
    fig.add_trace(
        go.Scatter(
            x=subset["date"],
            y=subset["sales"],
            mode="markers",
            name=f"{etype} event",
            marker=dict(color=color, size=8, symbol=EVENT_SYMBOLS[etype]),
            hovertext=subset["event_name_1"] if "event_name_1" in subset.columns else etype,
            hoverinfo="text+x+y",
        )
    )
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Units sold",
    legend=dict(orientation="h", y=1.08),
    height=340,
    margin=dict(t=20, b=40),
)
st.plotly_chart(fig, use_container_width=True)

zero_pct = (item_df["sales"] == 0).mean() * 100
mean_sales = item_df[item_df["sales"] > 0]["sales"].mean()
col1, col2, col3 = st.columns(3)
col1.metric("Zero-demand days", f"{zero_pct:.1f}%")
col2.metric("Mean sales (non-zero days)", f"{mean_sales:.2f} units")
col3.metric("Total days", f"{len(item_df):,}")


# Zero-inflation by category/department
st.divider()
st.subheader("Zero-inflation rate by category")
st.caption(
    "Fraction of item-days with zero sales. "
    "High intermittency makes Tweedie loss more effective than standard L2."
)

zero_cat = (
    df.groupby("cat_id")["sales"]
    .apply(lambda s: (s == 0).mean() * 100)
    .reset_index()
    .rename(columns={"sales": "zero_pct"})
    .sort_values("zero_pct", ascending=False)
)

fig2 = px.bar(
    zero_cat,
    x="cat_id",
    y="zero_pct",
    labels={"cat_id": "Category", "zero_pct": "Zero-demand rate (%)"},
    text_auto=".1f",
    color="zero_pct",
    color_continuous_scale="Reds",
)
fig2.update_layout(coloraxis_showscale=False, height=300, margin=dict(t=20, b=40))
st.plotly_chart(fig2, use_container_width=True)


# Sales distribution by category
st.divider()
st.subheader("Non-zero sales distribution by category")
st.caption("Distribution of daily sales volumes on days when at least one unit was sold.")

nonzero = df[df["sales"] > 0]
fig3 = px.histogram(
    nonzero,
    x="sales",
    color="cat_id",
    nbins=60,
    barmode="overlay",
    opacity=0.7,
    labels={"sales": "Units sold per item-day", "cat_id": "Category"},
    color_discrete_map={"FOODS": "#1f77b4", "HOUSEHOLD": "#ff7f0e", "HOBBIES": "#2ca02c"},
)
fig3.update_layout(height=320, margin=dict(t=20, b=40))
st.plotly_chart(fig3, use_container_width=True)


# Weekly sales pattern
st.divider()
st.subheader("Average daily sales by weekday — all items")

# M5 wday: 1=Sat, 2=Sun, 3=Mon ... 7=Fri
wday_labels = {1: "Sat", 2: "Sun", 3: "Mon", 4: "Tue", 5: "Wed", 6: "Thu", 7: "Fri"}
wday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

weekly = (
    df.groupby(["cat_id", "wday"])["sales"]
    .mean()
    .reset_index()
    .rename(columns={"sales": "avg_sales"})
)
weekly["day"] = weekly["wday"].map(wday_labels)

fig4 = px.line(
    weekly,
    x="day",
    y="avg_sales",
    color="cat_id",
    markers=True,
    labels={"day": "Day of week", "avg_sales": "Average units sold", "cat_id": "Category"},
    category_orders={"day": wday_order},
    color_discrete_map={"FOODS": "#1f77b4", "HOUSEHOLD": "#ff7f0e", "HOBBIES": "#2ca02c"},
)
fig4.update_layout(height=300, margin=dict(t=20, b=40))
st.plotly_chart(fig4, use_container_width=True)
