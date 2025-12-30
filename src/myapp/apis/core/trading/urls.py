from django.urls import include, path
from .trading_api import (
    ListExchangesAPI
)

urlpatterns = [
    # Include indicator URLs
    # path('indicators/', include('myapp.apis.core.trading.indicators.urls')),
    # # Include prediction URLs
    # path('predictions/', include('myapp.apis.core.trading.predictions.urls')),
    
    # Include Content URLs
    path('content/', include('myapp.apis.core.trading.content.urls')), 
    path('rag/', include('myapp.apis.core.trading.rag.urls')), 
    path('insights/', include('myapp.apis.core.trading.insights.urls')), 
    # Exchange Management
    path('exchanges/', ListExchangesAPI.as_view(), name='list_exchanges'),
    # path('exchanges/credentials/', AddUserExchangeCredentialsAPI.as_view(), name='add_exchange_credentials'),
    
    # Account & Balance
    # path('balance/', GetUserAccountBalanceAPI.as_view(), name='get_balance'),
    # path('portfolio/', GetUserPortfolioAPI.as_view(), name='get_portfolio'),
    
    # # Trading
    # path('trade/', PlaceTradeOrderAPI.as_view(), name='place_order'),
    # path('trades/', GetUserTradingHistoryAPI.as_view(), name='trading_history'),
]