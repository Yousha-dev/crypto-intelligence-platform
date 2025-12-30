from django.urls import path
from .api import (
    GetModerationQueueAPI,
    ApproveContentAPI,
    RejectContentAPI,
    FlagContentAPI,
    GetSystemHealthAPI,
    GetDatabaseStatsAPI,
    UpdateThresholdsAPI,
)
 
urlpatterns = [
    # 👮 Moderation
    path('moderation/queue/', GetModerationQueueAPI.as_view(), name='moderation-queue'),
    path('moderation/approve/<str:article_id>/', ApproveContentAPI.as_view(), name='moderation-approve'),
    path('moderation/reject/<str:article_id>/', RejectContentAPI.as_view(), name='moderation-reject'),
    path('moderation/flag/<str:article_id>/', FlagContentAPI.as_view(), name='moderation-flag'),
    
    # ⚙️ System
    path('health/', GetSystemHealthAPI.as_view(), name='system-health'),
    path('statistics/', GetDatabaseStatsAPI.as_view(), name='database-stats'),
    path('thresholds/', UpdateThresholdsAPI.as_view(), name='update-thresholds'),
]
