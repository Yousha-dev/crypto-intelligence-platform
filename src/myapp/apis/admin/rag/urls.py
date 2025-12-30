# rag/urls.py
from django.urls import path
from .api import (
    IndexDocumentsAPI,
    RAGStatsAPI,
    RebuildIndexAPI,
)
from .kg_api import (
    EntityContextAPI,
    EntityPathAPI,
    TrendingEntitiesAPI,
    KnowledgeGraphStatsAPI,
    BuildGraphFromContentAPI,
)
from .llm_api import (
    LLMStatusAPI,
    SwitchLLMProviderAPI,
    TestLLMProviderAPI,
)
 
urlpatterns = [
    # 🔧 Index Management
    path('index-documents/', IndexDocumentsAPI.as_view(), name='index-documents'),
    path('rag-stats/', RAGStatsAPI.as_view(), name='rag-stats'),
    path('rebuild-index/', RebuildIndexAPI.as_view(), name='rebuild-index'),
     
    # 🕸️ Knowledge Graph
    path('kg/entity/', EntityContextAPI.as_view(), name='kg-entity-context'),
    path('kg/path/', EntityPathAPI.as_view(), name='kg-entity-path'),
    path('kg/trending/', TrendingEntitiesAPI.as_view(), name='kg-trending'),
    path('kg/stats/', KnowledgeGraphStatsAPI.as_view(), name='kg-stats'),
    path('kg/build/', BuildGraphFromContentAPI.as_view(), name='kg-build'),
    
    # LLM Management
    path('llm/status/', LLMStatusAPI.as_view(), name='llm-status'),
    path('llm/switch/', SwitchLLMProviderAPI.as_view(), name='llm-switch'),
    path('llm/test/', TestLLMProviderAPI.as_view(), name='llm-test'),
]