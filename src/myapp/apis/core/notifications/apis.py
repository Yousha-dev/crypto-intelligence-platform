from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from datetime import timedelta
from myapp.emailhelper import EmailHelper
from drf_yasg import openapi
from myapp.serializers.core_serializers import NotificationSerializer
from myapp.models import Notifications, Users
from myapp.permissions import IsUserAccess
from drf_yasg.utils import swagger_auto_schema
import logging
logger = logging.getLogger(__name__)

class CreateNotificationAPI(APIView):
    """Create a new notification for the authenticated user."""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Create a new notification for the authenticated user.",
        request_body=NotificationSerializer,
        responses={
            201: openapi.Response(
                description="Notification created successfully",
                schema=NotificationSerializer
            ),
            400: openapi.Response(
                description="Validation errors.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            ),
            500: openapi.Response(
                description="Internal server error.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            )
        }
    )
    def post(self, request):
        try:
            user_id = getattr(request, "user_id", None)
            
            if not user_id:
                return Response({"error": "User ID is missing in the token."}, status=400)

            data = request.data
            data["isactive"] = 1
            data["isdeleted"] = 0
            data["userid"] = user_id
            data["createdby"] = user_id
            
            serializer = NotificationSerializer(data=data)
            if serializer.is_valid():
                notification = serializer.save()
                return Response({
                    "message": "Notification created successfully",
                    "data": serializer.data
                }, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response(
                {"error": f"Error creating notification: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ListNotificationsAPI(APIView):
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="List all active notifications for the authenticated user.",
        responses={
            200: NotificationSerializer(many=True),
            400: openapi.Response(
                description="Bad request",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            ),
            500: openapi.Response(
                description="Internal server error",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            )
        }
    )
    def get(self, request):
        try:
            user_id = getattr(request, "user_id", None)
            
            if not user_id:
                return Response({"error": "User ID is missing in the token."}, status=400)

            notifications = Notifications.objects.filter(
                userid=user_id,
                isactive=1,
                isdeleted=0
            ).select_related('extensionsubscriptionid', 'contractid').order_by('-createdat')
            
            serializer = NotificationSerializer(notifications, many=True)
            return Response({
                "message": "Notifications retrieved successfully",
                "count": notifications.count(),
                "data": serializer.data
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"Error retrieving notifications: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
class MarkNotificationsAsReadAPI(APIView):
    """Mark notification as read for the authenticated user."""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Mark notification as read for the authenticated user.",
        responses={
            200: openapi.Response(
                description="Notification marked as read successfully.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'message': openapi.Schema(type=openapi.TYPE_STRING),
                        'data': openapi.Schema(
                            type=openapi.TYPE_OBJECT,
                            properties={field: NotificationSerializer().fields[field] for field in NotificationSerializer.Meta.fields}
                        )
                    }
                )
            ),
            400: openapi.Response(
                description="Bad request.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            ),
            404: openapi.Response(
                description="Notification not found.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            ),
            500: openapi.Response(
                description="Internal server error.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            )
        }
    )
    def put(self, request, notification_id):
        try:
            user_id = getattr(request, "user_id", None)
            
            if not user_id:
                return Response({"error": "User ID is missing in the token."}, status=400)

            notification = Notifications.objects.filter(
                notificationid=notification_id,
                userid=user_id,
                isactive=1,
                isdeleted=0
            ).first()

            if not notification:
                return Response(
                    {"error": "Notification not found."},
                    status=status.HTTP_404_NOT_FOUND
                )

            notification.isread = 1
            notification.updatedby = user_id
            notification.updatedat = timezone.now()
            notification.save()
            
            return Response({
                "message": "Notification marked as read successfully.",
                "data": NotificationSerializer(notification).data
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"Error marking notification as read: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class DeleteNotificationAPI(APIView):
    """Delete a specific notification for the authenticated user."""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Delete a specific notification.",
        responses={
            200: openapi.Response(
                description="Notification deleted successfully.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'message': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            ),
            404: openapi.Response(
                description="Notification not found.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            ),
            400: openapi.Response(
                description="Bad request.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            ),
            500: openapi.Response(
                description="Internal server error.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            )
        }
    )
    def delete(self, request, notification_id):
        try:
            user_id = getattr(request, "user_id", None)
            
            if not user_id:
                return Response(
                    {"error": "User ID is missing in the token."}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            notification = Notifications.objects.filter(
                notificationid=notification_id,
                userid=user_id,
                isdeleted=0
            ).first()

            if not notification:
                return Response(
                    {"error": "Notification not found."}, 
                    status=status.HTTP_404_NOT_FOUND
                )

            # Soft delete
            notification.isdeleted = 1
            notification.updatedby = user_id
            notification.updatedat = timezone.now()
            notification.save()

            return Response({
                "message": "Notification deleted successfully."
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error deleting notification: {str(e)}")
            return Response(
                {"error": f"Error deleting notification: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ClearAllNotificationsAPI(APIView):
    """Clear all notifications for the authenticated user."""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Clear all notifications for the authenticated user.",
        responses={
            200: openapi.Response(
                description="All notifications cleared successfully.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'message': openapi.Schema(type=openapi.TYPE_STRING),
                        'count': openapi.Schema(type=openapi.TYPE_INTEGER)
                    }
                )
            ),
            400: openapi.Response(
                description="Bad request.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            ),
            500: openapi.Response(
                description="Internal server error.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            )
        }
    )
    def delete(self, request):
        try:
            user_id = getattr(request, "user_id", None)
            
            if not user_id:
                return Response(
                    {"error": "User ID is missing in the token."}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Get all active notifications for the user
            notifications = Notifications.objects.filter(
                userid=user_id,
                isactive=1,
                isdeleted=0
            )

            count = notifications.count()

            # Bulk update to soft delete all notifications
            notifications.update(
                isdeleted=1,
                updatedby=user_id,
                updatedat=timezone.now()
            )

            return Response({
                "message": "All notifications cleared successfully.",
                "count": count
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error clearing notifications: {str(e)}")
            return Response(
                {"error": f"Error clearing notifications: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class MarkNotificationsAsReadAPI(APIView):
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Mark notification as read for the authenticated user.",
        responses={
            200: NotificationSerializer,
            400: openapi.Response(
                description="Bad request",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            ),
            404: openapi.Response(
                description="Notification not found",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            ),
            500: openapi.Response(
                description="Internal server error",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            )
        }
    )

    def put(self, request, notification_id):
        try:
            user_id = getattr(request, "user_id", None)
            
            if not user_id:
                return Response({"error": "User ID is missing in the token."}, status=400)

            notification = Notifications.objects.filter(
                notificationid=notification_id,
                userid=user_id,
                isactive=1,
                isdeleted=0
            ).first()

            if not notification:
                return Response(
                    {"error": "Notification not found."},
                    status=status.HTTP_404_NOT_FOUND
                )

            notification.isread = 1
            notification.updatedby = user_id
            notification.updatedat = timezone.now()
            notification.save()
            
            return Response({
                "message": "Notification marked as read successfully.",
                "data": NotificationSerializer(notification).data
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"Error marking notification as read: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class AutoSendEmailNotificationAPI(APIView):
    """
    Automatically check and send email notifications for subscriptions and contracts near expiry.
    """
    permission_classes = []  # No permission required for automated task
    authentication_classes = []  # No authentication required

    def _create_notification(self, user_id, title, message, type_name, related_id, days_remaining, is_contract=False):
        """Helper method to create a notification using serializer"""
        try:
            notification_data = {
                'userid': user_id,
                'title': title,
                'message': message,
                'type': type_name,
                'daysuntilexpiry': days_remaining,
                'isread': 0,
                'isactive': 1,
                'isdeleted': 0
            }

            # Set the appropriate related field based on type
            if is_contract:
                notification_data['contractid'] = related_id
            else:
                notification_data['extensionsubscriptionid'] = related_id
            
            serializer = NotificationSerializer(data=notification_data)
            if serializer.is_valid():
                serializer.save()
            else:
                logger.error(f"Notification validation error: {serializer.errors}")
                
        except Exception as e:
            logger.error(f"Error creating notification: {str(e)}")

    @swagger_auto_schema(
        operation_description="Automatically send email notifications for subscriptions and contracts near expiry.",
        responses={
            200: openapi.Response(
                description="Email notifications sent successfully.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'message': openapi.Schema(type=openapi.TYPE_STRING),
                        'seven_day_notifications': openapi.Schema(
                            type=openapi.TYPE_OBJECT,
                            properties={
                                'subscriptions': openapi.Schema(type=openapi.TYPE_INTEGER),
                                'contracts': openapi.Schema(type=openapi.TYPE_INTEGER)
                            }
                        ),
                        'one_day_notifications': openapi.Schema(
                            type=openapi.TYPE_OBJECT,
                            properties={
                                'subscriptions': openapi.Schema(type=openapi.TYPE_INTEGER),
                                'contracts': openapi.Schema(type=openapi.TYPE_INTEGER)
                            }
                        )
                    }
                )
            ),
            500: openapi.Response(
                description="Internal server error.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            )
        }
    )
    def get(self, request):
        try:
            today = timezone.now().date()
            email_helper = EmailHelper()
            
            # Check items expiring in 7 days and tomorrow
            seven_days_future = today + timedelta(days=7)
            tomorrow = today + timedelta(days=1)

            # Extension Subscriptions
            seven_day_subs = Extensionsubscriptions.objects.filter(
                enddate=seven_days_future,
                isactive=1,
                isdeleted=0
            )
            one_day_subs = Extensionsubscriptions.objects.filter(
                enddate=tomorrow,
                isactive=1,
                isdeleted=0
            )

            # Contracts
            seven_day_contracts = Contracts.objects.filter(
                duedate=seven_days_future,
                isactive=1,
                isdeleted=0
            )
            one_day_contracts = Contracts.objects.filter(
                duedate=tomorrow,
                isactive=1,
                isdeleted=0
            )
            
            # Process Extension Subscriptions
            for subscription in seven_day_subs:
               # recipient_list = list(self._get_user_users(subscription.userid))
                if recipient_list:
                    message = f"Your subscription for {subscription.servicename} will expire in 7 days."
                    html_message = get_notification_template(
                        title="Subscription Expiry Notice",
                        item_name=subscription.servicename,
                        days_remaining=7,
                        end_date=subscription.enddate,
                        item_type="subscription"
                    )
                    email_helper.send_email_async(
                        subject="Subscription Expiry Notice - 7 Days Remaining",
                        message=message,
                        recipient_list=recipient_list,
                        html_message=html_message
                    )
                    self._create_notification(
                        user_id=subscription.userid.userid,
                        title="Subscription Expiry Notice",
                        message=message,
                        type_name="Expiry",
                        related_id=subscription.extensionsubscriptionid,
                        days_remaining=7
                    )

            # Process Contracts
            for contract in seven_day_contracts:
               # recipient_list = list(self._get_user_users(contract.userid))
                if recipient_list:
                    message = f"Your contract '{contract.contractname}' will expire in 7 days."
                    html_message = get_notification_template(
                        title="Contract Expiry Notice",
                        item_name=contract.contractname,
                        days_remaining=7,
                        end_date=contract.duedate,
                        item_type="contract"
                    )
                    email_helper.send_email_async(
                        subject="Contract Expiry Notice - 7 Days Remaining",
                        message=message,
                        recipient_list=recipient_list,
                        html_message=html_message
                    )
                    self._create_notification(
                        user_id=contract.userid.userid,
                        title="Contract Expiry Notice",
                        message=message,
                        type_name="Expiry",
                        related_id=contract.contractid,
                        days_remaining=7,
                        is_contract=True
                    )

            # Similar process for one day notifications
            # ... (implement similar logic for one_day_subs and one_day_contracts)

            # Process Extension Subscriptions
            for subscription in one_day_subs:
               # recipient_list = list(self._get_user_users(subscription.userid))
                if recipient_list:
                    message = f"Your subscription for {subscription.servicename} will expire in 1 day."
                    html_message = get_notification_template(
                        title="Subscription Expiry Notice",
                        item_name=subscription.servicename,
                        days_remaining=1,
                        end_date=subscription.enddate,
                        item_type="subscription"
                    )
                    email_helper.send_email_async(
                        subject="Subscription Expiry Notice - 1 Day Remaining",
                        message=message,
                        recipient_list=recipient_list,
                        html_message=html_message
                    )
                    self._create_notification(
                        user_id=subscription.userid.userid,
                        title="Subscription Expiry Notice",
                        message=message,
                        type_name="Expiry",
                        related_id=subscription.extensionsubscriptionid,
                        days_remaining=1
                    )

            # Process Contracts
            for contract in one_day_contracts:
               # recipient_list = list(self._get_user_users(contract.userid))
                if recipient_list:
                    message = f"Your contract '{contract.contractname}' will expire in 1 day."
                    html_message = get_notification_template(
                        title="Contract Expiry Notice",
                        item_name=contract.contractname,
                        days_remaining=1,
                        end_date=contract.duedate,
                        item_type="contract"
                    )
                    email_helper.send_email_async(
                        subject="Contract Expiry Notice - 1 Day Remaining",
                        message=message,
                        recipient_list=recipient_list,
                        html_message=html_message
                    )
                    self._create_notification(
                        user_id=contract.userid.userid,
                        title="Contract Expiry Notice",
                        message=message,
                        type_name="Expiry",
                        related_id=contract.contractid,
                        days_remaining=1,
                        is_contract=True
                    )
            
            return Response({
                "message": "Email notifications sent successfully.",
                "seven_day_notifications": {
                    "subscriptions": seven_day_subs.count(),
                    "contracts": seven_day_contracts.count()
                },
                "one_day_notifications": {
                    "subscriptions": one_day_subs.count(),
                    "contracts": one_day_contracts.count()
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in AutoSendEmailNotificationAPI: {str(e)}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
def get_notification_template(title, item_name, days_remaining, end_date, item_type):
    """
    HTML template for expiry notifications
    """
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
    </head>
    <body style="margin: 0; padding: 0; font-family: Arial, sans-serif; line-height: 1.6; background-color: #f5f5f5;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background-color: #00796B; padding: 30px; border-radius: 10px 10px 0 0; text-align: center;">
                <h1 style="color: #ffffff; margin: 0; font-size: 24px;">{title}</h1>
            </div>
            <div style="background-color: #ffffff; padding: 30px; border-radius: 0 0 10px 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin-bottom: 20px; font-size: 16px;">Your {item_type} <strong>{item_name}</strong> will expire in <strong>{days_remaining} days</strong>.</p>
                
                <div style="background-color: #f8f9fa; border-left: 4px solid #00796B; padding: 15px; margin-bottom: 20px;">
                    <p style="margin: 0; color: #444444;">
                        <strong>{item_type.title()}:</strong> {item_name}<br>
                        <strong>Expiry Date:</strong> {end_date.strftime('%B %d, %Y')}<br>
                        <strong>Days Remaining:</strong> {days_remaining}
                    </p>
                </div>
                
                <p style="margin-bottom: 20px; color: #666666;">Please take action to ensure uninterrupted service:</p>
                <ul style="color: #666666; margin-bottom: 25px;">
                    <li>Review your {item_type} details</li>
                    <li>Contact support if you need assistance</li>
                </ul>
                
                <div style="text-align: center;">
                    <a href="#" style="display: inline-block; background-color: #00796B; color: #ffffff; padding: 12px 25px; text-decoration: none; border-radius: 5px; font-weight: bold;">View {item_type.title()}</a>
                </div>
            </div>
            <div style="text-align: center; padding-top: 20px; color: #666666; font-size: 12px;">
                <p>This is an automated message, please do not reply directly to this email.</p>
            </div>
        </div>
    </body>
    </html>
    """