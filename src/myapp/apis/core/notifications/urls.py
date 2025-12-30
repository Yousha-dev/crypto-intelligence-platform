from django.urls import path
from .apis import (
    AutoSendEmailNotificationAPI, ClearAllNotificationsAPI, CreateNotificationAPI, ListNotificationsAPI, DeleteNotificationAPI, MarkNotificationsAsReadAPI
)

urlpatterns = [
    path('create/', CreateNotificationAPI.as_view(), name='create_notification'),
    path('list/', ListNotificationsAPI.as_view(), name='list_notifications'),
    path('<int:notification_id>/delete/', DeleteNotificationAPI.as_view(), name='delete_notification'),
    path('clear-all/', ClearAllNotificationsAPI.as_view(), name='clear_notifications'),
    path('<int:notification_id>/mark-as-read/', MarkNotificationsAsReadAPI.as_view(), name='marks_as_read_notification'),
    path('auto-send-email-notification/', AutoSendEmailNotificationAPI.as_view(), name='auto_send_email_notification'),
]

    
