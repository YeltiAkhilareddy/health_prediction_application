import os
from groq import Groq

# =====================================
# GROQ CLIENT SETUP
# =====================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=GROQ_API_KEY)


# =====================================
# MAIN LLM EXPLANATION FUNCTION
# =====================================

def generate_ai_remarks(
    age,
    gender,
    glucose,
    cholesterol,
    sysBP,
    current_smoker,
    diabetes,
    haemoglobin,
    risk_level,
    probability
):
    """
    Generates human-readable medical explanation using Groq LLM.
    """

    try:

        # -----------------------------
        # Prompt Engineering
        # -----------------------------
        prompt = f"""
You are a medical explanation assistant in a health risk prediction system.

STRICT RULES:
- Do NOT diagnose diseases.
- Do NOT give treatment advice.
- Only explain risk factors based on input data.
- Keep response concise (5–8 lines).
- Be clear, structured, and easy to understand.

PATIENT DATA:
- Age: {age}
- Gender: {gender}
- Glucose: {glucose}
- Cholesterol: {cholesterol}
- Systolic BP: {sysBP}
- Smoker: {current_smoker}
- Diabetes: {diabetes}
- Haemoglobin: {haemoglobin}

MODEL OUTPUT:
- Risk Level: {risk_level}
- Probability: {probability}

TASK:
Explain why this patient received this risk level.
Highlight key contributing factors in simple terms.
"""

        # -----------------------------
        # GROQ API CALL
        # -----------------------------

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful medical explanation assistant. "
                        "You only explain risk factors, never diagnose diseases."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=250
        )

        remarks = response.choices[0].message.content
        remarks = remarks.replace("\n", " ").strip()

        return remarks

    except Exception as e:

        print("❌ GROQ API ERROR:", str(e))

        # Safe fallback (important for production stability)
        return (
            "AI explanation temporarily unavailable. "
            "Please refer to risk factors such as glucose, cholesterol, BP, and lifestyle conditions."
        )