import streamlit as st
import requests
from datetime import date

# ==============================
# CONFIG
# ==============================
API_URL = "http://127.0.0.1:8000/patients/predict/"

st.set_page_config(
    page_title="AI Health Risk Predictor",
    layout="centered"
)

st.title("🩺 AI Health Risk Prediction System")

st.markdown("Enter patient details to get ML + AI-based health risk analysis.")

# ==============================
# INPUT FORM
# ==============================
with st.form("health_form"):

    full_name = st.text_input("Full Name")

    dob = st.date_input(
        "Date of Birth",
        min_value=date(1900, 1, 1)
    )

    gender = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    email = st.text_input("Email")

    glucose = st.number_input(
        "Glucose",
        min_value=0.0
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=0.0
    )

    haemoglobin = st.number_input(
        "Haemoglobin",
        min_value=0.0
    )

    current_smoker = st.selectbox(
        "Current Smoker",
        [0, 1]
    )

    diabetes = st.selectbox(
        "Diabetes",
        [0, 1]
    )

    sysBP = st.number_input(
        "Systolic BP",
        min_value=0.0
    )

    submit = st.form_submit_button("Predict Health Risk")

# ==============================
# API CALL
# ==============================
if submit:

    payload = {
        "full_name": full_name,
        "dob": dob.isoformat(),
        "gender": gender,
        "email": email,
        "glucose": glucose,
        "cholesterol": cholesterol,
        "haemoglobin": haemoglobin,
        "currentSmoker": current_smoker,
        "diabetes": diabetes,
        "sysBP": sysBP
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:

            data = response.json()

            st.success("Prediction Completed ✅")

            # ==========================
            # DISPLAY RESULTS
            # ==========================

            st.subheader("📊 Risk Analysis")

            st.metric("Risk Level", data["risk_level"])
            st.metric("Probability", data["probability"])

            st.write("### 🧠 AI Explanation")
            st.write(data["remarks"])

            st.write("### 🩸 Hemoglobin Status")
            st.write(data["haemoglobin_status"])


        else:
            st.error(response.text)

    except Exception as e:
        st.error(f"Error connecting to API: {e}")