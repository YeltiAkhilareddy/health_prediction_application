from django.db import models


class Patient(models.Model):

    # -------------------------
    # Patient Information
    # -------------------------

    full_name = models.CharField(max_length=100)

    dob = models.DateField()

    age = models.IntegerField()

    gender = models.CharField(max_length=10)

    email = models.EmailField(unique=True)

    # -------------------------
    # Health Parameters
    # -------------------------

    glucose = models.FloatField()

    haemoglobin = models.FloatField()

    cholesterol = models.FloatField()

    current_smoker = models.BooleanField(
        default=False
    )

    diabetes = models.BooleanField(
        default=False
    )

    sysBP = models.FloatField(
        null=True,
        blank=True
    )

    # -------------------------
    # ML Output
    # -------------------------

    risk_prediction = models.CharField(
        max_length=50,
        blank=True
    )

    prediction_probability = models.FloatField(
        null=True,
        blank=True
    )

    # -------------------------
    # AI Explanation
    # -------------------------

    remarks = models.TextField(
        blank=True
    )

    # -------------------------
    # Tracking
    # -------------------------

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    def __str__(self):
        return self.full_name