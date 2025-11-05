
import streamlit as st
import joblib
import numpy as np
import lzma

# Load trained model (compressed)
with lzma.open("model_compressed_p3.pkl.xz", "rb") as f:
    model = joblib.load(f)

# App title and description
st.title("🏠 House Price Prediction App (₹ INR)")

st.divider()

st.write(
    "This app uses a machine learning model to predict **house prices in Indian Rupees (₹)** "
    "based on the features you enter below. Enter the details and click **Predict!**"
)

st.divider()

# Input fields
bedrooms = st.number_input("Number of bedrooms", min_value=0, value=3)
bathrooms = st.number_input("Number of bathrooms", min_value=0, value=2)
livingarea = st.number_input("Living area (sqft)", min_value=200, value=2000)
condition = st.number_input("Condition of the house (1–5)", min_value=1, max_value=5, value=3)
numberofschools = st.number_input("Number of schools nearby", min_value=0, value=2)

st.divider()

# Predict button
if st.button("Predict!"):
    st.balloons()
    x = np.array([[bedrooms, bathrooms, livingarea, condition, numberofschools]])
    prediction = model.predict(x)[0]

    # 💱 Convert USD to INR if needed (optional)
    usd_to_inr = 84.0  # Approximate exchange rate
    price_in_inr = prediction * usd_to_inr

    st.success(f"💰 Estimated house price: **₹{price_in_inr:,.2f}**")
else:
    st.info("👆 Enter values and press **Predict!** to get a price estimate in ₹ INR.")


