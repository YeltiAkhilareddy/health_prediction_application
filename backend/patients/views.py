from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import viewsets
from datetime import date
import json

from .models import Patient
from .serializers import PatientSerializer
from ml.predictor import get_final_prediction


# -----------------------------
# OPTIONAL: CRUD API (DRF)
# -----------------------------
class PatientViewSet(viewsets.ModelViewSet):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer


# -----------------------------
# AGE CALCULATION FUNCTION
# -----------------------------
def calculate_age(dob):
    today = date.today()
    return (
        today.year
        - dob.year
        - ((today.month, today.day) < (dob.month, dob.day))
    )


# -----------------------------
# ML PREDICTION API
# -----------------------------
@csrf_exempt
def predict_health(request):

    if request.method != "POST":
        return JsonResponse(
            {"error": "POST request required"},
            status=405
        )

    try:

        data = json.loads(request.body)

        # -------------------------
        # INPUT DATA
        # -------------------------

        full_name = data.get("full_name")
        dob = date.fromisoformat(data.get("dob"))
        email = data.get("email")
        gender = data.get("gender")

        glucose = float(data.get("glucose"))
        haemoglobin = float(data.get("haemoglobin"))
        cholesterol = float(data.get("cholesterol"))

        current_smoker = int(
            data.get("currentSmoker", 0)
        )

        diabetes = int(
            data.get("diabetes", 0)
        )

        sysBP = float(
            data.get("sysBP")
        )

        # -------------------------
        # BASIC VALIDATION
        # -------------------------

        if glucose < 0:
            return JsonResponse(
                {"error": "Invalid glucose value"},
                status=400
            )

        if cholesterol < 0:
            return JsonResponse(
                {"error": "Invalid cholesterol value"},
                status=400
            )

        if haemoglobin < 0:
            return JsonResponse(
                {"error": "Invalid haemoglobin value"},
                status=400
            )

        if sysBP < 50:
            return JsonResponse(
                {"error": "Invalid systolic BP value"},
                status=400
            )

        # -------------------------
        # PROCESSING
        # -------------------------

        age = calculate_age(dob)

        result = get_final_prediction(
            age=age,
            glucose=glucose,
            cholesterol=cholesterol,
            haemoglobin=haemoglobin,
            gender=gender,
            current_smoker=current_smoker,
            diabetes=diabetes,
            sysBP=sysBP
        )

        risk = result.get(
            "risk_level",
            "Unknown"
        )

        probability = result.get(
            "probability",
            0.0
        )

        # -------------------------
        # SAVE TO DATABASE
        # -------------------------

        patient = Patient.objects.create(
            full_name=full_name,
            dob=dob,
            age=age,
            gender=gender,
            email=email,

            glucose=glucose,
            cholesterol=cholesterol,
            haemoglobin=haemoglobin,

            current_smoker=bool(current_smoker),
            diabetes=bool(diabetes),
            sysBP=sysBP,

            risk_prediction=risk,
            remarks=result.get(
                "remarks",
                ""
            ),

            prediction_probability=probability
        )

        result["patient_id"] = patient.id

        return JsonResponse(result)

    except Exception as e:

        return JsonResponse(
            {"error": str(e)},
            status=500
        )