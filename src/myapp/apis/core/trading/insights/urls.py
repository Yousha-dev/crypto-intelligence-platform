from django.urls import path
from .api import (
    MarketSummaryAPI,
    CoinInsightsAPI,
    MarketAlertsAPI,
    InfluencerTrackingAPI,
    NarrativeAnalysisAPI,
)

urlpatterns = [
    # Market Summary
    path('summary/', MarketSummaryAPI.as_view(), name='market-summary'),
    
    # Coin-specific Insights
    path('coin/<str:symbol>/', CoinInsightsAPI.as_view(), name='coin-insights'),
    
    # Alerts
    path('alerts/', MarketAlertsAPI.as_view(), name='market-alerts'),
    
    # Influencer Tracking
    path('influencers/', InfluencerTrackingAPI.as_view(), name='influencer-tracking'),
    
    # Narrative Analysis
    path('narratives/', NarrativeAnalysisAPI.as_view(), name='narrative-analysis'),
]