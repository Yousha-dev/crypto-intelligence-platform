# news/urls.py
from django.urls import path
from .api import (
    # News Feed APIs
    GetCuratedFeedAPI,
    GetAllArticlesAPI,
    GetArticleDetailAPI,
    SearchArticlesAPI,
    
    # Social Feed APIs
    GetSocialFeedAPI,
    GetSocialPostDetailAPI,
    GetSocialStatsAPI,
    SearchSocialPostAPI,
    
    # Combined Feed API
    GetCombinedFeedAPI,
    
    # Credibility APIs
    AnalyzeContentAPI,
    GetSourceHistoryAPI,
    GetCredibilityStatsAPI,
     
    # Trending & Topics APIs
    GetTrendingAPI,
    GetTopicsAPI,
    GetTrendingHistoryAPI,
    GetSentimentOverviewAPI,
    
)

urlpatterns = [
    # News Feed
    path('feed/', GetCuratedFeedAPI.as_view(), name='news-feed'),
    path('articles/', GetAllArticlesAPI.as_view(), name='news-articles'),
    path('articles/<str:article_id>/', GetArticleDetailAPI.as_view(), name='news-article-detail'),
    path('search/', SearchArticlesAPI.as_view(), name='news-search'),
     
    # Social Feed (NEW)
    path('social/', GetSocialFeedAPI.as_view(), name='social-feed'),
    path('social/<str:post_id>/', GetSocialPostDetailAPI.as_view(), name='social-post-detail'),
    path('social-stats/', GetSocialStatsAPI.as_view(), name='social-stats'),
    path('social-search/', SearchSocialPostAPI.as_view(), name='social-search'),
    
    # 🔀 Combined Feed (NEW)
    path('combined/', GetCombinedFeedAPI.as_view(), name='combined-feed'),
    
    # Credibility
    path('analyze/', AnalyzeContentAPI.as_view(), name='news-analyze'),
    path('source-history/', GetSourceHistoryAPI.as_view(), name='source-history'),
    path('credibility-stats/', GetCredibilityStatsAPI.as_view(), name='credibility-stats'),
    path('sentiment-overview/', GetSentimentOverviewAPI.as_view(), name='sentiment-overview'),
    
    # Trending & Topics
    path('trending/', GetTrendingAPI.as_view(), name='news-trending'),
    path('topics/', GetTopicsAPI.as_view(), name='news-topics'),
    path('trending-history/', GetTrendingHistoryAPI.as_view(), name='trending-history'),

]