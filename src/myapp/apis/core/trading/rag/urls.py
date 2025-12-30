# rag/urls.py
from django.urls import path
from .api import (
    SemanticSearchAPI,
    AskQuestionAPI,
    ChatWithNewsAPI,
    GenerateSummaryAPI,
    AnalyzeEntityAPI,
    MarketSentimentAnalysisAPI,
    ExecuteQueryChainAPI,
    ChatWithContextAPI,
    RAGPerformanceStatsAPI,
)

urlpatterns = [
    # Semantic Search
    path('semantic-search/', SemanticSearchAPI.as_view(), name='semantic-search'),
    
    # 💬 Q&A
    path('ask/', AskQuestionAPI.as_view(), name='ask-question'),
    path('chat/', ChatWithNewsAPI.as_view(), name='chat-with-news'),
    
    # Summary & Analysis
    path('summary/', GenerateSummaryAPI.as_view(), name='generate-summary'),
    path('analyze-entity/', AnalyzeEntityAPI.as_view(), name='analyze-entity'),
    path('market-sentiment/', MarketSentimentAnalysisAPI.as_view(), name='market-sentiment-analysis'),
      
    # 🔗 Advanced RAG
    path('query-chain/', ExecuteQueryChainAPI.as_view(), name='execute-query-chain'),
    path('chat-context/', ChatWithContextAPI.as_view(), name='chat-with-context'),
    path('performance/', RAGPerformanceStatsAPI.as_view(), name='rag-performance-stats'),
]