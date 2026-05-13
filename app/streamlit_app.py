import streamlit as st

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
)


def home():
    st.title("Retail Demand Forecasting")
    st.markdown(
        """
        End-to-end demand forecasting pipeline built on the
        [M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
        dataset.

        **Scope:** CA_1 store × 3,049 products × 3 categories (HOBBIES, HOUSEHOLD, FOODS)

        **Model:** LightGBM with Tweedie objective, compared against Gaussian (L2) loss

        **Novelty:** Tweedie loss is mathematically suited for zero-inflated demand.
        Retail sales contain many zeros (especially HOBBIES items), and standard
        L2 regression produces biased predictions in this case. The Tweedie
        objective handles the zero-inflation explicitly.
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.page_link("pages/1_Data_Explorer.py", label="Data Explorer", icon="📊")
        st.caption("Explore historical sales, zero-inflation patterns, and event effects.")
    with col2:
        st.page_link("pages/2_Forecast.py", label="28-Day Forecast", icon="🔮")
        st.caption("View the 28-day demand forecast per item and category.")
    with col3:
        st.page_link("pages/3_Model_Insights.py", label="Model Insights", icon="🎯")
        st.caption("Feature importance and Tweedie vs Gaussian metrics by aggregation level.")
    with col4:
        st.page_link("pages/4_Forecast_Charts.py", label="Forecast Charts", icon="📈")
        st.caption("Actual vs predicted time-series across total, category, department, and item levels.")


pg = st.navigation([
    st.Page(home, title="Home", icon="🏠"),
    st.Page("pages/1_Data_Explorer.py", title="Data Explorer", icon="📊"),
    st.Page("pages/2_Forecast.py", title="28-Day Forecast", icon="🔮"),
    st.Page("pages/3_Model_Insights.py", title="Model Insights", icon="🎯"),
    st.Page("pages/4_Forecast_Charts.py", title="Forecast Charts", icon="📈"),
])
pg.run()
