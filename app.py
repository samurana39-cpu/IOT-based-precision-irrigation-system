# -----------------------------------------------------
# 💧 IoT Smart Irrigation Dashboard (Modern UI Version)
# -----------------------------------------------------

import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="IoT Smart Irrigation", page_icon="💧", layout="wide")

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #fafafa;
        }
        .main-title {
            text-align: center;
            color: #00B4D8;
            font-size: 2.2rem;
            font-weight: bold;
            padding-bottom: 10px;
        }
        .sub-title {
            text-align: center;
            color: #90E0EF;
            font-size: 1.1rem;
            margin-bottom: 20px;
        }
        .card {
            background-color: #1a1c25;
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0px 4px 15px rgba(0, 180, 216, 0.3);
            text-align: center;
            transition: all 0.3s ease;
        }
        .card:hover {
            box-shadow: 0px 6px 25px rgba(0, 180, 216, 0.6);
        }
        .pump-on {
            background: linear-gradient(135deg, #00b4d8, #0077b6);
            color: white;
            padding: 20px;
            border-radius: 20px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: bold;
        }
        .pump-off {
            background: linear-gradient(135deg, #52b788, #1b4332);
            color: white;
            padding: 20px;
            border-radius: 20px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown("<div class='main-title'>💧 IoT Smart Irrigation System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI + IoT-powered Precision Farming Dashboard</div>", unsafe_allow_html=True)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🌿 Enter Environment Data")

temperature = st.sidebar.number_input("🌡️ Temperature (°C)", 0.0, 50.0, 28.0)
soil_moisture = st.sidebar.number_input("🌱 Soil Moisture (%)", 0.0, 1000.0, 400.0)
days = st.sidebar.number_input("📅 Day Number", 1, 31, 1)
time_val = st.sidebar.number_input("⏰ Time (Hour)", 0.0, 24.0, 12.0)
moisture_trend = st.sidebar.slider("📈 Moisture Trend", 0.0, 1000.0, 500.0)
crop = st.sidebar.selectbox("🌾 Crop Type", ["cotton", "wheat", "rice", "maize"])

# -----------------------------
# Load Model & Scaler
# -----------------------------
try:
    model = pickle.load(open("irrigation_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    with open("feature_columns.txt") as f:
        feature_cols = [line.strip() for line in f.readlines()]
except Exception as e:
    st.error(f"❌ Error loading model or files: {e}")
    st.stop()

num_cols = ["temperature", "soil_moisture", "days", "time", "moisture_trend"]

# -----------------------------
# Prepare Input
# -----------------------------
input_df = pd.DataFrame([{
    "temperature": temperature,
    "soil_moisture": soil_moisture,
    "days": days,
    "time": time_val,
    "moisture_trend": moisture_trend,
    "crop": crop
}])

input_df = pd.get_dummies(input_df, columns=["crop"], drop_first=True)
for c in feature_cols:
    if c not in input_df.columns:
        input_df[c] = 0
input_df = input_df[feature_cols]
input_df[num_cols] = scaler.transform(input_df[num_cols])

# -----------------------------
# Prediction + Logic
# -----------------------------
raw_pred = int(model.predict(input_df)[0])
final_prediction = raw_pred
rule_reason = "📊 Using model prediction"

if soil_moisture >= 600:
    final_prediction = 0
    rule_reason = "🌿 Soil Moisture ≥ 600 → Pump OFF"
elif soil_moisture <= 350 and temperature >= 30:
    final_prediction = 1
    rule_reason = "🔥 Dry & Hot → Pump ON"

# -----------------------------
# Display Output in Dashboard Card
# -----------------------------
st.markdown("## 🧠 Prediction Result")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if final_prediction == 1:
        st.markdown("<div class='pump-on'>💧 Pump is ON<br><span style='font-size:0.9rem;'>Soil is dry. Irrigation required.</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='pump-off'>🌤️ Pump is OFF<br><span style='font-size:0.9rem;'>Soil has enough moisture.</span></div>", unsafe_allow_html=True)

# -----------------------------
# Additional Info
# -----------------------------
with st.expander("🔍 Debug Information"):
    st.write(f"Model Raw Prediction: {raw_pred}")
    st.write(f"Temperature: {temperature} °C")
    st.write(f"Soil Moisture: {soil_moisture}")
    st.write(rule_reason)

st.markdown("---")
st.caption("🌱 Developed as part of Deep Learning IoT Precision Irrigation Project")
