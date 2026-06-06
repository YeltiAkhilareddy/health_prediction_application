from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import PatientViewSet, predict_health

# DRF Router
router = DefaultRouter()
router.register(r'patients', PatientViewSet)

urlpatterns = [
    path("predict/", predict_health, name="predict_health"),
]