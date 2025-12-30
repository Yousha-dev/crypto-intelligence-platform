# predictions/urls.py
from django.urls import path
from .api import (
    GenerateAIPredictionAPI, ListUserPredictionsAPI, 
    GetUserPredictionStatsAPI, GetUserPredictionLimitsAPI
)

urlpatterns = [
    path('generate/', GenerateAIPredictionAPI.as_view(), name='generate_prediction'),
    path('list/', ListUserPredictionsAPI.as_view(), name='list_predictions'),
    path('stats/', GetUserPredictionStatsAPI.as_view(), name='predictions_stats'),
    path('limits/', GetUserPredictionLimitsAPI.as_view(), name='prediction_limits'),  
] 