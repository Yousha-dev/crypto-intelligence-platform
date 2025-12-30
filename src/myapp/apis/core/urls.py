from django.urls import path, include
from myapp.apis.core.core_api import GetAvailableSubscriptionPlansAPI, GetUserAPI, GetUserPaymentsAPI, ListBillingHistoryAPI, UpdateUserAPI, GetUserSubscriptionAPI, ChangeSubscriptionPlanAPI, GetUserSubscriptionStatsAPI, SubscriptionHealthCheckAPI, CheckSubscriptionLimitsAPI, RecordAPIUsageAPI

urlpatterns = [
    path('trading/', include('myapp.apis.core.trading.urls')),
    #path('events/', include('myapp.apis.core.events.urls')),
    #path('notifications/', include('myapp.apis.core.notifications.urls')),
    #path('reminders/', include('myapp.apis.core.reminders.urls')),
    # User Management
    path('user/', GetUserAPI.as_view(), name='get_user'),
    path('user/update/', UpdateUserAPI.as_view(), name='update_user'),
    
    # Payment Management
    path('payments/', GetUserPaymentsAPI.as_view(), name='get_user_payments'),
    path('payments/billing/history/', ListBillingHistoryAPI.as_view(), name='list_billing_history'),
    
    # Subscription Management
    path('subscription/', GetUserSubscriptionAPI.as_view(), name='get_subscription'),
    path('subscription/stats/', GetUserSubscriptionStatsAPI.as_view(), name='get_subscription_stats'),
    path('subscription/health/', SubscriptionHealthCheckAPI.as_view(), name='subscription_health_check'),
    path('subscription/limits/', CheckSubscriptionLimitsAPI.as_view(), name='check_subscription_limits'),
    path('subscription/plans/', GetAvailableSubscriptionPlansAPI.as_view(), name='get_available_subscription_plans'),
    path('subscription/change/', ChangeSubscriptionPlanAPI.as_view(), name='change_subscription_plan'),
    
    # API Usage Tracking
    path('usage/record/', RecordAPIUsageAPI.as_view(), name='record_api_usage'),
]