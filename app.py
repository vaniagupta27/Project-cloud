# =========================================
# 🏭 SMART MANUFACTURING STREAMLIT APP
# =========================================

import streamlit as st
import kagglehub
from kagglehub import kaggleDatasetAdapter
import numpy as np
import joblib

# Load saved files
model = joblib.load("smart_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Smart Manufacturing Predictor")
st.title("🏭 Smart Manufacturing Prediction System")

st.write("Enter machine details to predict maintenance requirement")

# =========================================
# 🎯 USER INPUT
# =========================================

machine_id = st.number_input("Machine ID", 1, 100)
temperature = st.number_input("Temperature")
vibration = st.number_input("Vibration")
humidity = st.number_input("Humidity")
pressure = st.number_input("Pressure")
energy = st.number_input("Energy Consumption")
machine_status = st.selectbox("Machine Status", [0,1,2])
anomaly_flag = st.selectbox("Anomaly Flag", [0,1])
predicted_life = st.number_input("Predicted Remaining Life")
downtime_risk = st.number_input("Downtime Risk")

failure_type_input = st.selectbox(
    "Failure Type",
    ["Normal", "Vibration Issue", "Overheating", "Power Failure"]
)

# Encode failure_type using saved encoder
if "failure_type" in encoders:
    failure_type_encoded = encoders["failure_type"].transform([failure_type_input])[0]
else:
    failure_type_encoded = 0

# =========================================
# 🚀 PREDICTION
# =========================================

if st.button("Predict"):

    input_data = np.array([[machine_id, temperature, vibration, humidity,
                            pressure, energy, machine_status, anomaly_flag,
                            predicted_life, failure_type_encoded, downtime_risk]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    # =========================================
    # 📢 OUTPUT
    # =========================================

    if prediction == 1:
        st.error("⚠️ Maintenance Required!")
    else:
        st.success("✅ Machine is Healthy")
