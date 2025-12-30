# routing.py
from django.urls import re_path
from myapp import consumers

websocket_urlpatterns = [
    re_path(r'^ws/prices/(?P<exchange>\w+)/(?P<symbol>[A-Z0-9\-_/]+)/(?P<timeframe>\w+)/?$',
            consumers.PriceConsumer.as_asgi()),

    # Series management WebSocket
    re_path(r'^ws/series/status/(?P<exchange>\w+)/(?P<symbol>[A-Z0-9\-_/]+)/(?P<timeframe>\w+)/?$',
            consumers.SeriesStatusConsumer.as_asgi()),
     
    # Orchestrator monitoring
    re_path(r'^ws/orchestrator/monitor/(?P<worker_id>\w+)/?$',
            consumers.OrchestratorMonitorConsumer.as_asgi()),

]