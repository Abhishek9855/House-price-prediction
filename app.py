
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")  # Make sure model.pkl is in the same folder

# App title and description
st.title("🏠 House Price Prediction App")

st.divider()

st.write(
    "This app uses a machine learning model to predict house prices "
    "based on the features you enter below. Enter the details and click 'Predict!'"
)

st.divider()

# Input fields
bedrooms = st.number_input("Number of bedrooms", min_value=0, value=0)
bathrooms = st.number_input("Number of bathrooms", min_value=0, value=0)
livingarea = st.number_input("Living area", min_value=0, value=2000)
condition = st.number_input("Condition of the house", min_value=0, value=3)
numberofschools = st.number_input("Number of schools nearby", min_value=0, value=0)

st.divider()

# Create input array for prediction
x = [[bedrooms, bathrooms, livingarea, condition, numberofschools]]

# Predict button
predictbutton = st.button("Predict!")

if predictbutton:
    st.balloons()
    x_array = np.array(x)
    prediction = model.predict(x_array)[0]

    st.write(f"Price prediction is{ prediction:,.2f}")
else:
    st.write("please use predict button after entering values")



#'number of bedrooms', 'number of bathrooms', 'living area',
  #'condition of the house', 'Number of schools nearby'