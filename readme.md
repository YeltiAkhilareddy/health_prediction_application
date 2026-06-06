# 🩺 AI Health Risk Prediction System

This project is a full-stack machine learning application that predicts a user’s health risk based on basic medical inputs and provides a clear explanation using an AI model.

It combines Machine Learning, rule-based logic, and a large language model to make predictions and explain them in simple terms.

---

## 🚀 What this project does

- Takes user health details like age, glucose, cholesterol, blood pressure, etc.
- Uses a trained ML model to predict cardiovascular risk
- Applies basic medical rule checks for safety adjustments
- Generates human-readable explanations using Groq LLM
- Displays everything in a simple Streamlit interface

---

## 🧠 Tech Stack

- Python
- Django (Backend API)
- Scikit-learn (Machine Learning)
- Pandas / NumPy
- Streamlit (Frontend UI)
- Groq API (Llama 3 for explanations)

---

## 🏗️ System Flow

User Input (Streamlit)
        ↓
Django API
        ↓
ML Model Prediction
        ↓
Rule-based Adjustments (Safety Layer)
        ↓
Groq LLM Explanation
        ↓
Final Response to UI

---

## 📊 Features

- Health risk prediction (Low / Medium / High)
- Probability score output
- AI-generated medical explanation
- Hemoglobin status analysis
- Risk flags for abnormal conditions
- Clean Streamlit UI for interaction

---

## 📁 Project Structure

backend/
│
├── ml/
│   ├── predictor.py
│   ├── ai_explainer_llm.py
│   ├── health_model.pkl
│
├── api/
│   ├── views.py
│   ├── models.py
│   ├── urls.py
│
frontend/
│   ├── app.py (Streamlit UI)

---

## ⚙️ How to run locally

### 1. Clone the project
git clone <your-repo-url>
cd project-folder

### 2. Install dependencies
pip install -r requirements.txt

### 3. Setup environment variables
GROQ_API_KEY=your_api_key_here

### 4. Run Django backend
python manage.py runserver

### 5. Run Streamlit frontend
streamlit run app.py

---

## 📌 API Endpoint

POST /predict_health/

### Sample Request
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

---

## 📤 Output Example
{
  "probability": 0.82,
  "risk_level": "High Risk",
  "haemoglobin_status": "Average",
  "flags": ["high_bp"],
  "remarks": "Patient shows elevated blood pressure and glucose levels indicating cardiovascular risk..."
}

---

## 💡 What I learned from this project

- Building ML pipeline end-to-end
- Django API integration with ML
- LLM integration using Groq
- Streamlit frontend development
- Real-world system design thinking

---

## 📈 Future improvements

- Patient history dashboard
- PDF report generation
- Model accuracy improvements
- Authentication system
- Cloud deployment

---

## ⚠️ Disclaimer

This is for educational purposes only, not for real medical diagnosis.
