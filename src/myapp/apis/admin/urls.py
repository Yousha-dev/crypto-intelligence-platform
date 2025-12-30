from django.urls import include, path
from myapp.apis.admin.admin_api import (
    EditUserAPI, GetAllUsersAPI, GetPaymentsAPI, ListUsersAPI, GetDashboardStatsAPI
)

urlpatterns = [
    path('subscriptions/', include('myapp.apis.admin.subscriptions.urls')),
    path('subscriptionplans/', include('myapp.apis.admin.subscriptionplans.urls')),
    path('content/', include('myapp.apis.admin.content.urls')),
    path('rag/', include('myapp.apis.admin.rag.urls')),
    path('users/', ListUsersAPI.as_view(), name='list_users'),
    path('users/<int:user_id>/edit-user', EditUserAPI.as_view(), name='edit_user'),
    path('all-users/', GetAllUsersAPI.as_view(), name='get_all_users'),
    path('payments/', GetPaymentsAPI.as_view(), name='get_payments'),
    path('dashboard/stats/', GetDashboardStatsAPI.as_view(), name='dashboard_stats'),
    path('', lambda request: None, name='placeholder'),
]