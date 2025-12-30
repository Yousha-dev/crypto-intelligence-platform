# indicators/urls.py
from django.urls import path
from .api import GetAvailableIndicatorsAPI, GetTechnicalSignalsAPI

urlpatterns = [
    path('available/', GetAvailableIndicatorsAPI.as_view(), name='available_indicators'),
    path('signals/', GetTechnicalSignalsAPI.as_view(), name='technical_signals'),
] 