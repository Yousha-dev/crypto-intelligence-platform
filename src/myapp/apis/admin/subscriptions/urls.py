from django.urls import path
from .api import (
    CreateSubscriptionAPI, ListSubscriptionsAPI,
    UpdateSubscriptionAPI, DeleteSubscriptionAPI,
    SubscriptionAnalyticsAPI, SubscriptionDashboardOverviewAPI, RenewSubscriptionAPI, AutoRenewSubscriptionsAPI,
)

urlpatterns = [
    path('create/', CreateSubscriptionAPI.as_view(), name='create_subscription'),
    path('list/', ListSubscriptionsAPI.as_view(), name='list_subscriptions'),
    path('<int:subscription_id>/update/', UpdateSubscriptionAPI.as_view(), name='update_subscription'),
    path('<int:subscription_id>/delete/', DeleteSubscriptionAPI.as_view(), name='delete_subscription'),
    path('analytics/', SubscriptionAnalyticsAPI.as_view(), name='subscription_analytics'),
    path('dashboard/', SubscriptionDashboardOverviewAPI.as_view(), name='subscription_dashboard_overview'),
    path('<int:subscription_id>/renew/', RenewSubscriptionAPI.as_view(), name='renew_subscription'),
    path('sync/auto-renew/', AutoRenewSubscriptionsAPI.as_view(), name='auto_renew_subscriptions'),
]
