
import streamlit as st
import joblib
import numpy as np
import os

# ----------------------------
# Load model safely
# ----------------------------
MODEL_PATH = "model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        model_loaded = True
    except Exception as e:
        st.warning(f"⚠️ Failed to load model. Running demo mode. ({e})")
        model_loaded = False
else:
    st.warning("⚠️ Model file not found. Running demo mode.")
    model_loaded = False

# ----------------------------
# App title and description
# ----------------------------
st.title("🏠 House Price Prediction App")
st.divider()

st.write(
    "This app predicts house prices based on the features you enter below. "
    "Enter the details and click **Predict!**"
)
st.divider()

# ----------------------------
# Input fields
# ----------------------------
bedrooms = st.number_input("Number of bedrooms", min_value=0, value=3)
bathrooms = st.number_input("Number of bathrooms", min_value=0, value=2)
livingarea = st.number_input("Living area (sqft)", min_value=0, value=2000)
condition = st.number_input("Condition of the house (1-5)", min_value=1, max_value=5, value=3)
numberofschools = st.number_input("Number of schools nearby", min_value=0, value=2)

st.divider()

# ----------------------------
# Prediction logic
# ----------------------------
x = np.array([[bedrooms, bathrooms, livingarea, condition, numberofschools]])

if st.button("Predict!"):
    st.balloons()
    if model_loaded:
        try:
            prediction = model.predict(x)[0]
            st.success(f"🏡 Estimated House Price: ₹{prediction:,.2f}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
    else:
        # Demo calculation
        fake_price = (bedrooms * 150000) + (bathrooms * 120000) + (livingarea * 120) + (condition * 50000)
        st.info(f"💡 Demo Mode — Estimated Price: ₹{fake_price:,.0f}")
else:
    st.info("👉 Fill in the details above and click **Predict!**")

st.divider()

# ----------------------------
# Footer (centered)
# ----------------------------
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "Built with ❤️ using <b>Streamlit</b> | Demo mode enabled if no model available"
    "</p>",
    unsafe_allow_html=True
)
