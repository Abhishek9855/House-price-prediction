# app.py

import streamlit as st
import pandas as pd
from recommender import predict_house_price

st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏠",
    layout="centered"
)

st.markdown(
    """
    <h1 style="text-align:center;">🏠 House Price Prediction</h1>
    <p style="text-align:center; color:gray;">
        Machine Learning based house price estimator
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# Load Neighborhoods
# -------------------------------------------------
@st.cache_data
def load_neighborhoods():
    try:
        df = pd.read_csv("train.csv")
        return sorted(df["Neighborhood"].dropna().unique())
    except:
        return ["NAmes", "CollgCr", "OldTown", "Edwards"]

neighborhoods = load_neighborhoods()

# -------------------------------------------------
# Input Form
# -------------------------------------------------
with st.form("house_price_form"):
    col1, col2 = st.columns(2)

    with col1:
        GrLivArea = st.number_input("Above Ground Living Area (sqft)", 300, 10000, 1500)
        OverallQual = st.slider("Overall Quality (1–10)", 1, 10, 5)
        YearBuilt = st.number_input("Year Built", 1800, 2025, 2000)
        FullBath = st.slider("Full Bathrooms", 0, 4, 2)

    with col2:
        GarageCars = st.slider("Garage Capacity (Cars)", 0, 5, 2)
        GarageArea = st.number_input("Garage Area (sqft)", 0, 2000, 400)
        LotArea = st.number_input("Lot Area (sqft)", 500, 50000, 7000)
        Neighborhood = st.selectbox("Neighborhood", neighborhoods)

    submit = st.form_submit_button("Predict Price")

# -------------------------------------------------
# Output
# -------------------------------------------------
if submit:
    try:
        price = predict_house_price(
            GrLivArea,
            OverallQual,
            YearBuilt,
            FullBath,
            GarageCars,
            GarageArea,
            LotArea,
            Neighborhood
        )
        st.success(f"💰 Estimated House Price: ₹ {price:,.2f}")
    except Exception as e:
        st.error("❌ Prediction failed")
        st.code(str(e))
