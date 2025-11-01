
import streamlit as st
import joblib
import numpy as np
import os

st.title("🏠 House Price Prediction App")
st.divider()
st.write(
    "This app uses a machine learning model to predict house prices "
    "based on the features you enter below. Enter the details and click 'Predict!'"
)
st.divider()

# ----------------------------------------------------------
# Try to load model safely
# ----------------------------------------------------------
model = None

if os.path.exists("model.pkl"):
    try:
        model = joblib.load("model.pkl")
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
else:
    st.warning("⚠️ Model file not found. Running in demo mode (no real predictions).")

# ----------------------------------------------------------
# Input fields
# ----------------------------------------------------------
bedrooms = st.number_input("Number of bedrooms", min_value=0, value=0)
bathrooms = st.number_input("Number of bathrooms", min_value=0, value=0)
livingarea = st.number_input("Living area (sqft)", min_value=0, value=2000)
condition = st.number_input("Condition of the house (1–5)", min_value=1, max_value=5, value=3)
numberofschools = st.number_input("Number of schools nearby", min_value=0, value=0)

st.divider()

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------
x = np.array([[bedrooms, bathrooms, livingarea, condition, numberofschools]])

if st.button("Predict!"):
    st.balloons()
    if model:
        prediction = model.predict(x)[0]
        st.success(f"🏡 Predicted price: ₹{prediction:,.2f}")
    else:
        # Demo output if model isn't available
        fake_price = (bedrooms + bathrooms + condition) * 100000 + livingarea * 150
        st.info(f"💡 Demo price (no model loaded): ₹{fake_price:,.0f}")
else:
    st.write("Please click **Predict!** after entering values.")
