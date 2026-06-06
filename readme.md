# 🩺 AI Health Risk Prediction System
---

## 📖 Overview

A full-stack machine learning application that predicts a user's cardiovascular health risk based on basic medical inputs and explains the result in plain language using a large language model (LLM).

It combines a trained ML model, rule-based safety logic, and the Groq LLM (Llama 3) to deliver risk predictions with human-readable explanations — all through a clean Streamlit interface.

---

## 🚀 What This Project Does

- Accepts user health details (age, glucose, cholesterol, blood pressure, etc.)
- Uses a trained ML model to predict cardiovascular risk
- Applies rule-based checks as a medical safety layer
- Generates plain-language explanations via Groq (Llama 3)
- Displays everything in a simple, interactive Streamlit UI

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend UI | Streamlit |
| Backend API | Django REST Framework |
| Machine Learning | Scikit-learn, Pandas, NumPy |
| LLM Explanations | Groq API (Llama 3) |
| Model Serialisation | Joblib |
| Environment Config | python-dotenv |

---

## 🤖 Model Details

| Property | Value |
|---|---|
| **Algorithm** | Random Forest Classifier |
| **Dataset** | Framingham Heart Study (~4,240 records) |
| **Features Used** | Age, glucose, cholesterol, systolic BP, smoking status, diabetes, haemoglobin |
| **Output** | Risk probability (0–1) + risk level (Low / Medium / High) |
| **Accuracy** | ~85% on test split |
| **AUC-ROC** | ~0.88 |
| **Training Script** | `backend/ml/train_model.py` |

The trained model is saved as `backend/ml/health_model.pkl`. To retrain from scratch, run:

```bash
python backend/ml/train_model.py
```

---

## 🏗️ System Flow

```
User Input (Streamlit UI)
        │
        ▼
Django REST API  ──  POST /predict_health/
        │
        ▼
predictor.py  ──►  health_model.pkl  (ML Prediction + Probability)
        │
        ▼
Rule-based Safety Layer  (flag: high BP, high glucose, anaemia, etc.)
        │
        ▼
ai_explainer_llm.py  ──►  Groq API / Llama 3  (Plain-language Explanation)
        │
        ▼
JSON Response  ──►  Streamlit UI  (Risk Level + Score + Remarks + Flags)
```

---

## 📊 Features

- **Risk prediction**: Low / Medium / High cardiovascular risk classification
- **Probability score**: Numeric confidence output (0.0 – 1.0)
- **AI-generated explanation**: Plain-language medical remarks via Llama 3
- **Haemoglobin status analysis**: Classifies as Low / Average / High based on gender
- **Risk flags**: Automatic flagging of abnormal vitals (e.g. `high_bp`, `high_glucose`)
- **Clean Streamlit UI**: Simple form-based interaction, no setup required for end users

### Haemoglobin Classification Thresholds

| Status | Male | Female |
|---|---|---|
| Low | < 13.0 g/dL | < 12.0 g/dL |
| Average | 13.0 – 17.5 g/dL | 12.0 – 15.5 g/dL |
| High | > 17.5 g/dL | > 15.5 g/dL |

---

## 📁 Project Structure

```
project-root/
│
├── backend/
│   ├── manage.py
│   ├── settings.py
│   ├── requirements.txt
│   │
│   ├── ml/
│   │   ├── predictor.py          # Loads model, runs prediction + rule checks
│   │   ├── ai_explainer_llm.py   # Calls Groq API for LLM explanation
│   │   ├── health_model.pkl      # Trained Random Forest model (binary)
│   │   └── train_model.py        # Script to retrain the model from scratch
│   │
│   └── api/
│       ├── views.py              # /predict_health/ endpoint logic
│       ├── models.py             # Django DB models (optional persistence)
│       ├── serializers.py        # DRF serializers for request/response validation
│       └── urls.py               # URL routing
│
├── frontend/
│   └── app.py                    # Streamlit UI
│
├── .env.example                  # Environment variable template
├── .env                          # Your local secrets (never commit this)
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ⚙️ Local Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd project-folder
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Core packages included:

```
django
djangorestframework
scikit-learn
pandas
numpy
streamlit
groq
joblib
python-dotenv
```

### 3. Configure Environment Variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

**.env.example:**

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Django
DJANGO_SECRET_KEY=your_django_secret_key_here
DEBUG=True

# Streamlit → Django connection
BACKEND_URL=http://127.0.0.1:8000
```

> Get your Groq API key at [console.groq.com](https://console.groq.com)

### 4. Run Django Backend

```bash
cd backend
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

The API will be available at: `http://127.0.0.1:8000`

### 5. Run Streamlit Frontend

Open a second terminal:

```bash
cd frontend
streamlit run app.py
```

The UI will open automatically at: `http://localhost:8501`

---

## 📌 API Reference

### Endpoint

```
POST http://127.0.0.1:8000/predict_health/
Content-Type: application/json
```

### Request Body

```json
{
  "full_name": "John Doe",
  "dob": "1995-05-10",
  "gender": "male",
  "email": "john@example.com",
  "glucose": 180,
  "cholesterol": 220,
  "haemoglobin": 13.5,
  "currentSmoker": 1,
  "diabetes": 0,
  "sysBP": 140
}
```

### Input Field Reference

| Field | Type | Valid Range / Values | Notes |
|---|---|---|---|
| `full_name` | string | Any | Patient's full name |
| `dob` | string | `YYYY-MM-DD` | Date of birth |
| `gender` | string | `"male"` / `"female"` | Used for haemoglobin classification |
| `email` | string | Valid email | Contact identifier |
| `glucose` | int | 50 – 500 mg/dL | Fasting blood glucose level |
| `cholesterol` | int | 100 – 400 mg/dL | Total serum cholesterol |
| `haemoglobin` | float | 5.0 – 20.0 g/dL | Blood haemoglobin level |
| `sysBP` | int | 60 – 250 mmHg | Systolic blood pressure |
| `currentSmoker` | int | `0` or `1` | 1 = currently smokes |
| `diabetes` | int | `0` or `1` | 1 = diagnosed diabetic |

### Response Body

```json
{
  "probability": 0.82,
  "risk_level": "High Risk",
  "haemoglobin_status": "Average",
  "flags": ["high_bp", "high_glucose"],
  "remarks": "The patient presents with elevated systolic blood pressure (140 mmHg) and high fasting glucose (180 mg/dL), both significant risk factors for cardiovascular disease. Combined with active smoking, the overall cardiovascular risk is assessed as high. Immediate lifestyle modifications and medical consultation are strongly recommended."
}
```

### Risk Flag Reference

| Flag | Trigger Condition |
|---|---|
| `high_bp` | `sysBP` > 130 mmHg |
| `high_glucose` | `glucose` > 125 mg/dL |
| `high_cholesterol` | `cholesterol` > 200 mg/dL |
| `low_haemoglobin` | Below gender-specific threshold |
| `smoker_risk` | `currentSmoker` = 1 |
| `diabetic_risk` | `diabetes` = 1 |

---

## 📦 Dependencies

Install all with:

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `django` | Backend web framework |
| `djangorestframework` | REST API support |
| `scikit-learn` | ML model training and prediction |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `streamlit` | Frontend UI |
| `groq` | Groq API client for Llama 3 |
| `joblib` | Model serialisation (.pkl) |
| `python-dotenv` | Environment variable loading |

---

## 💡 What I Learned

- Building an end-to-end ML pipeline (data → training → serialisation → inference)
- Integrating ML models into a Django REST API
- Using the Groq API for real-time LLM-powered explanations
- Streamlit frontend development and state management
- Real-world system design: separating concerns across ML, rules, LLM, and UI layers

---

## 📈 Future Improvements

- **Patient history dashboard** — track risk scores over time per user
- **PDF report generation** — downloadable health summary per patient
- **Model accuracy improvements** — larger dataset, hyperparameter tuning, cross-validation
- **Authentication system** — secure patient login and data isolation
- **Cloud deployment** — AWS / GCP / Render with CI/CD pipeline
- **Additional risk models** — diabetes onset, stroke risk, kidney disease
- **Multilingual support** — LLM explanations in regional languages

---


> **⚠️ Medical Disclaimer:** This tool is for **educational and portfolio demonstration purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Never disregard professional medical advice or delay seeking it because of something generated by this application.
