# =========================================
# 🏭 SMART MANUFACTURING STREAMLIT APP
# =========================================

import streamlit as st
import numpy as np
import joblib

# ===============================
# Load saved model files
# ===============================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Smart Manufacturing Predictor")

st.title("🏭 Smart Manufacturing Prediction System")
st.write("Predict whether maintenance is required based on machine data")

# ===============================
# USER INPUT
# ===============================

machine_id = st.number_input("Machine ID", min_value=1, max_value=100, value=1)
temperature = st.number_input("Temperature", value=50.0)
vibration = st.number_input("Vibration", value=20.0)
humidity = st.number_input("Humidity", value=50.0)
pressure = st.number_input("Pressure", value=2.0)
energy = st.number_input("Energy Consumption", value=1.0)

machine_status = st.selectbox("Machine Status", [0, 1, 2])
anomaly_flag = st.selectbox("Anomaly Flag", [0, 1])

predicted_life = st.number_input("Predicted Remaining Life", value=100)
downtime_risk = st.number_input("Downtime Risk", value=0.0)

failure_type_input = st.selectbox(
    "Failure Type",
    ["Normal", "Vibration Issue", "Overheating", "Power Failure"]
)

# ===============================
# Encode categorical input
# ===============================

try:
    failure_type_encoded = encoders["failure_type"].transform([failure_type_input])[0]
except:
    st.error("⚠️ Unknown category in failure_type")
    failure_type_encoded = 0

# ===============================
# PREDICTION
# ===============================

if st.button("Predict"):

    try:
        # Input order MUST match training
        input_data = np.array([[machine_id, temperature, vibration, humidity,
                                pressure, energy, machine_status, anomaly_flag,
                                predicted_life, failure_type_encoded, downtime_risk]])

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Output
        if prediction == 1:
            st.error("⚠️ Maintenance Required!")
        else:
            st.success("✅ Machine is Healthy")

    except Exception as e:
        st.error(f"Error: {e}")
