import pickle
import os
import pandas as pd

# =====================================
# LOAD MODEL + THRESHOLD
# =====================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "health_model.pkl"
)

THRESHOLD_PATH = os.path.join(
    os.path.dirname(__file__),
    "threshold.pkl"
)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(THRESHOLD_PATH, "rb") as f:
    THRESHOLD = pickle.load(f)


# =====================================
# RISK LEVEL MAPPING
# =====================================

def get_risk_level(prob):

    if prob < 0.30:
        return "Low Risk"

    elif prob < 0.60:
        return "Medium Risk"

    return "High Risk"


# =====================================
# ML PREDICTION
# =====================================

def predict_health_risk(
    age,
    glucose,
    cholesterol,
    gender,
    current_smoker,
    diabetes,
    sysBP
):

    try:

        male = 1 if gender.lower() == "male" else 0

        input_data = pd.DataFrame([{
            "age": age,
            "male": male,
            "glucose": glucose,
            "cholesterol": cholesterol,
            "currentSmoker": current_smoker,
            "diabetes": diabetes,
            "sysBP": sysBP
        }])

        prob = model.predict_proba(input_data)[0][1]

        prediction = int(prob >= THRESHOLD)

        return {
            "probability": round(float(prob), 3),
            "prediction": prediction,
            "risk_level": get_risk_level(prob)
        }

    except Exception as e:
        print("MODEL ERROR:", e)
        raise e


# =====================================
# HAEMOGLOBIN STATUS (KEEP THIS)
# =====================================

def get_haemoglobin_status(haemoglobin, gender, age):

    if age < 18:
        low_limit = 11.0

    elif gender.lower() == "male":
        low_limit = 13.5

    else:
        low_limit = 12.0

    if haemoglobin < low_limit:
        return "Low"

    elif haemoglobin <= low_limit + 2:
        return "Average"

    return "Good"


# =====================================
# HYBRID RULE ENGINE (NO TEXT OUTPUT HERE)
# =====================================

def apply_hybrid_rules(
    result,
    glucose,
    cholesterol,
    haemoglobin,
    sysBP,
    diabetes
):

    prob = result["probability"]
    risk_level = result["risk_level"]

    flags = []

    # -------------------------
    # Critical Overrides
    # -------------------------

    if glucose >= 300:
        risk_level = "High Risk"
        prob = max(prob, 0.90)
        flags.append("high_glucose")

    if cholesterol >= 450:
        risk_level = "High Risk"
        prob = max(prob, 0.90)
        flags.append("high_cholesterol")

    if haemoglobin <= 8:
        risk_level = "High Risk"
        prob = max(prob, 0.85)
        flags.append("low_haemoglobin")

    if sysBP >= 180:
        risk_level = "High Risk"
        prob = max(prob, 0.90)
        flags.append("high_bp")

    # -------------------------
    # Moderate Conditions
    # -------------------------

    if glucose >= 200 and risk_level == "Low Risk":
        risk_level = "Medium Risk"
        prob = max(prob, 0.50)
        flags.append("elevated_glucose")

    if cholesterol >= 300 and risk_level == "Low Risk":
        risk_level = "Medium Risk"
        prob = max(prob, 0.50)
        flags.append("elevated_cholesterol")

    if sysBP >= 160 and risk_level == "Low Risk":
        risk_level = "Medium Risk"
        prob = max(prob, 0.50)
        flags.append("elevated_bp")

    if diabetes == 1 and risk_level == "Low Risk":
        risk_level = "Medium Risk"
        prob = max(prob, 0.45)
        flags.append("diabetes_present")

    return {
        "probability": round(prob, 2),
        "risk_level": risk_level,
        "flags": flags
    }


# =====================================
# FINAL WRAPPER (LLM WILL HANDLE REMARKS)
# =====================================

from .ai import generate_ai_remarks


def get_final_prediction(
    age,
    glucose,
    cholesterol,
    haemoglobin,
    gender,
    current_smoker,
    diabetes,
    sysBP
):

    ml_result = predict_health_risk(
        age,
        glucose,
        cholesterol,
        gender,
        current_smoker,
        diabetes,
        sysBP
    )

    hybrid_result = apply_hybrid_rules(
        ml_result,
        glucose,
        cholesterol,
        haemoglobin,
        sysBP,
        diabetes
    )

    # -------------------------
    # 🔥 THIS IS WHAT YOU ARE MISSING
    # -------------------------
    ai_remarks = generate_ai_remarks(
        age=age,
        gender=gender,
        glucose=glucose,
        cholesterol=cholesterol,
        sysBP=sysBP,
        current_smoker=current_smoker,
        diabetes=diabetes,
        haemoglobin=haemoglobin,
        risk_level=hybrid_result["risk_level"],
        probability=hybrid_result["probability"]
    )

    hybrid_result["remarks"] = ai_remarks

    hybrid_result["haemoglobin_status"] = get_haemoglobin_status(
        haemoglobin,
        gender,
        age
    )

    return hybrid_result