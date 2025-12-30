from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth.hashers import make_password
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from myapp.permissions import IsUserAccess
from myapp.serializers.admin_serializers import (
    PaymentSerializer, SubscriptionSerializer, UserDetailAdminSerializer,
    AdminDashboardStatsSerializer
)
from myapp.models import (
    Payments, Subscriptions, Users, UserTrade, 
    APIUsage, UserPortfolio, Subscriptionplans, UserExchangeCredentials
)
from myapp.serializers.auth_serializers import UserSerializer
from django.db import transaction
from django.utils import timezone
from django.db.models import Count, Sum, Q
from datetime import datetime, timedelta
from decimal import Decimal

class ListUsersAPI(APIView):
    """
    List all users with their organization details.
    """
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="List all users with their organization details.",
        manual_parameters=[
            openapi.Parameter(
                'user_id', 
                openapi.IN_QUERY,
                description="Filter by user ID",
                type=openapi.TYPE_INTEGER,
                required=False
            ),
            openapi.Parameter(
                'subscriptionid', 
                openapi.IN_QUERY,
                description="Filter by subscription ID",
                type=openapi.TYPE_INTEGER,
                required=False
            ),
            openapi.Parameter(
                'paymentid', 
                openapi.IN_QUERY,
                description="Filter by payment ID",
                type=openapi.TYPE_INTEGER,
                required=False
            ),
            openapi.Parameter(
                'subscriptionplanid', 
                openapi.IN_QUERY,
                description="Filter by subscription plan ID",
                type=openapi.TYPE_INTEGER,
                required=False
            ),
            openapi.Parameter(
                'role', 
                openapi.IN_QUERY,
                description="Filter by user role",
                type=openapi.TYPE_STRING,
                required=False
            ),
            openapi.Parameter(
                'trading_experience', 
                openapi.IN_QUERY,
                description="Filter by trading experience",
                type=openapi.TYPE_STRING,
                enum=['beginner', 'intermediate', 'advanced'],
                required=False
            ),
            openapi.Parameter(
                'limit', 
                openapi.IN_QUERY,
                description="Limit number of results",
                type=openapi.TYPE_INTEGER,
                required=False
            ),
            openapi.Parameter(
                'offset', 
                openapi.IN_QUERY,
                description="Offset for pagination",
                type=openapi.TYPE_INTEGER,
                required=False
            )
        ],
        responses={
            200: UserDetailAdminSerializer(many=True),
            500: "Internal server error"
        }
    )
    def get(self, request):
        try:
            # Get query parameters
            user_id = request.query_params.get('user_id')
            subscriptionid = request.query_params.get('subscriptionid')
            paymentid = request.query_params.get('paymentid')
            subscriptionplanid = request.query_params.get('subscriptionplanid')
            role = request.query_params.get('role')
            trading_experience = request.query_params.get('trading_experience')
            limit = request.query_params.get('limit', 50)
            offset = request.query_params.get('offset', 0)

            # Build filters
            filters = {'isdeleted': 0}
            if user_id:
                filters['userid'] = user_id
            if role:
                filters['role'] = role
            if trading_experience:
                filters['trading_experience'] = trading_experience

            # Start with base queryset
            users = Users.objects.filter(**filters)

            # Apply relationship filters
            if subscriptionid:
                users = users.filter(
                    subscriptions__subscriptionid=subscriptionid,
                    subscriptions__isdeleted=0
                )
            if paymentid:
                users = users.filter(
                    subscriptions__payments__paymentid=paymentid,
                    subscriptions__payments__isdeleted=0
                )
            if subscriptionplanid:
                users = users.filter(
                    subscriptions__subscriptionplanid=subscriptionplanid,
                    subscriptions__isdeleted=0
                )

            # Apply pagination
            users = users.distinct().order_by('-createdat')
            total_count = users.count()
            users = users[int(offset):int(offset) + int(limit)]

            serializer = UserDetailAdminSerializer(users, many=True)
            
            return Response({
                'users': serializer.data,
                'total_count': total_count,
                'limit': int(limit),
                'offset': int(offset)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {"error": f"Error fetching users: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class EditUserAPI(APIView):
    """
    Edit user details including subscription information.
    """
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Edit user and subscription details.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'user': openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'fullname': openapi.Schema(type=openapi.TYPE_STRING, description="User full name"),
                        'email': openapi.Schema(type=openapi.TYPE_STRING, description="User email"),
                        'role': openapi.Schema(type=openapi.TYPE_STRING, description="User role"),
                        'organization': openapi.Schema(type=openapi.TYPE_STRING, description="Organization name"),
                        'phone': openapi.Schema(type=openapi.TYPE_STRING, description="Phone number"),
                        'address': openapi.Schema(type=openapi.TYPE_STRING, description="Address"),
                        'state': openapi.Schema(type=openapi.TYPE_STRING, description="State"),
                        'zipcode': openapi.Schema(type=openapi.TYPE_STRING, description="Zip code"),
                        'country': openapi.Schema(type=openapi.TYPE_STRING, description="Country"),
                        'trading_experience': openapi.Schema(
                            type=openapi.TYPE_STRING, 
                            enum=['beginner', 'intermediate', 'advanced'],
                            description="Trading experience"
                        ),
                        'risk_tolerance': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            enum=['low', 'medium', 'high'], 
                            description="Risk tolerance"
                        ),
                        'password': openapi.Schema(type=openapi.TYPE_STRING, description="User password (optional)"),
                        'isactive': openapi.Schema(type=openapi.TYPE_INTEGER, description="Is active flag"),
                        'isdeleted': openapi.Schema(type=openapi.TYPE_INTEGER, description="Is deleted flag"),
                    },
                ),
                'subscription': openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'subscriptionplanid': openapi.Schema(type=openapi.TYPE_INTEGER, description="Subscription plan ID"),
                        'billingfrequency': openapi.Schema(type=openapi.TYPE_STRING, description="Billing frequency"),
                        'startdate': openapi.Schema(type=openapi.TYPE_STRING, description="Subscription start date"),
                        'enddate': openapi.Schema(type=openapi.TYPE_STRING, description="Subscription end date"),
                        'autorenew': openapi.Schema(type=openapi.TYPE_INTEGER, description="Auto-renewal flag"),
                        'status': openapi.Schema(type=openapi.TYPE_STRING, description="Subscription status"),
                        'isactive': openapi.Schema(type=openapi.TYPE_INTEGER, description="Is active flag"),
                        'isdeleted': openapi.Schema(type=openapi.TYPE_INTEGER, description="Is deleted flag"),
                    },
                )
            },
        ),
        responses={
            200: "Update successful",
            400: "Validation error",
            404: "User not found",
            500: "Internal server error"
        }
    )
    @transaction.atomic
    def put(self, request, user_id):
        try:
            # Fetch the user
            user = Users.objects.filter(
                userid=user_id,
                isdeleted=0
            ).first()
            
            if not user:
                return Response(
                    {"error": "User not found"},
                    status=status.HTTP_404_NOT_FOUND
                )
    
            current_time = timezone.now()
    
            # Update user if user data is provided
            if 'user' in request.data and request.data['user']:
                user_data = {
                    'updatedat': current_time,
                    'updatedby': getattr(request, 'user_id', user.userid)
                }
                
                # Update user fields
                allowed_fields = [
                    'fullname', 'email', 'role', 'organization', 'phone', 'address', 
                    'state', 'zipcode', 'country', 'trading_experience', 'risk_tolerance',
                    'isactive', 'isdeleted'
                ]
                
                for field in allowed_fields:
                    if field in request.data['user']:
                        user_data[field] = request.data['user'][field]
    
                # Handle password separately
                if 'password' in request.data['user']:
                    user_data['passwordhash'] = make_password(request.data['user']['password'])
    
                user_serializer = UserSerializer(user, data=user_data, partial=True)
                if not user_serializer.is_valid():
                    return Response(user_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
                user_serializer.save()
            else:
                user_serializer = UserDetailAdminSerializer(user)
    
            # Update subscription if subscription data is provided
            subscription_serializer = None
            if 'subscription' in request.data and request.data['subscription']:
                # Get the active subscription for this user
                subscription = Subscriptions.objects.filter(
                    userid=user_id,
                    isactive=1,
                    isdeleted=0
                ).first()
    
                if subscription:
                    subscription_data = {
                        'updatedat': current_time,
                        'updatedby': getattr(request, 'user_id', user.userid)
                    }
                    
                    # Update subscription fields
                    allowed_sub_fields = [
                        'subscriptionplanid', 'billingfrequency', 'startdate', 'enddate',
                        'autorenew', 'status', 'isactive', 'isdeleted'
                    ]
                    
                    for field in allowed_sub_fields:
                        if field in request.data['subscription']:
                            subscription_data[field] = request.data['subscription'][field]
          
                    subscription_serializer = SubscriptionSerializer(
                        subscription, 
                        data=subscription_data, 
                        partial=True
                    )
                    if not subscription_serializer.is_valid():
                        return Response(subscription_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
                    subscription_serializer.save()
    
            # Prepare response data
            response_data = {
                "message": "Update successful",
                "data": {
                    "user": user_serializer.data if hasattr(user_serializer, 'data') else UserDetailAdminSerializer(user).data
                }
            }
            
            # Only include updated data in response
            if subscription_serializer:
                response_data["data"]["subscription"] = subscription_serializer.data
    
            return Response(response_data, status=status.HTTP_200_OK)
    
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class GetAllUsersAPI(APIView):
    """
    Get all users with their organization and subscription information.
    """
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter(
                'search', 
                openapi.IN_QUERY,
                description="Search by name or email",
                type=openapi.TYPE_STRING,
                required=False
            ),
            openapi.Parameter(
                'role', 
                openapi.IN_QUERY,
                description="Filter by user role",
                type=openapi.TYPE_STRING,
                required=False
            ),
            openapi.Parameter(
                'status', 
                openapi.IN_QUERY,
                description="Filter by active status (active/inactive)",
                type=openapi.TYPE_STRING,
                required=False
            ),
            openapi.Parameter(
                'limit', 
                openapi.IN_QUERY,
                description="Limit number of results",
                type=openapi.TYPE_INTEGER,
                required=False
            )
        ],
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    "users": openapi.Schema(
                        type=openapi.TYPE_ARRAY,
                        items=openapi.Schema(
                            type=openapi.TYPE_OBJECT,
                            properties={
                                "user_id": openapi.Schema(type=openapi.TYPE_INTEGER, description="User ID"),
                                "fullname": openapi.Schema(type=openapi.TYPE_STRING, description="Full name"),
                                "email": openapi.Schema(type=openapi.TYPE_STRING, description="Email"),
                                "role": openapi.Schema(type=openapi.TYPE_STRING, description="Role"),
                                "organization": openapi.Schema(type=openapi.TYPE_STRING, description="Organization"),
                                "phone": openapi.Schema(type=openapi.TYPE_STRING, description="Phone"),
                                "trading_experience": openapi.Schema(type=openapi.TYPE_STRING, description="Trading Experience"),
                                "isactive": openapi.Schema(type=openapi.TYPE_BOOLEAN, description="Is active"),
                                "subscription_status": openapi.Schema(type=openapi.TYPE_STRING, description="Subscription status"),
                                "created_at": openapi.Schema(type=openapi.TYPE_STRING, description="Created date"),
                            },
                        ),
                    ),
                    "total_count": openapi.Schema(type=openapi.TYPE_INTEGER, description="Total count")
                },
            )
        },
        operation_summary="Get all users",
        operation_description="API to fetch all users with their organization and subscription information.",
    )
    def get(self, request):
        try:
            # Get query parameters
            search = request.query_params.get('search', '')
            role = request.query_params.get('role')
            status_filter = request.query_params.get('status')
            limit = int(request.query_params.get('limit', 100))

            # Build base queryset
            queryset = Users.objects.filter(isdeleted=0)

            # Apply filters
            if search:
                queryset = queryset.filter(
                    Q(fullname__icontains=search) | Q(email__icontains=search)
                )

            if role:
                queryset = queryset.filter(role=role)

            if status_filter:
                if status_filter.lower() == 'active':
                    queryset = queryset.filter(isactive=1)
                elif status_filter.lower() == 'inactive':
                    queryset = queryset.filter(isactive=0)

            # Order and limit results
            queryset = queryset.order_by('-createdat')[:limit]

            user_data = []
            for user in queryset:
                # Get subscription status
                subscription = Subscriptions.objects.filter(
                    userid=user.userid, 
                    isactive=1, 
                    isdeleted=0
                ).first()

                user_data.append({
                    "user_id": user.userid,
                    "fullname": user.fullname,
                    "email": user.email,
                    "role": user.role,
                    "organization": user.organization,
                    "phone": user.phone,
                    "trading_experience": user.trading_experience,
                    "risk_tolerance": user.risk_tolerance,
                    "isactive": user.isactive == 1,
                    "subscription_status": subscription.status if subscription else "No Subscription",
                    "created_at": user.createdat.isoformat() if user.createdat else None,
                })

            return Response({
                "users": user_data,
                "total_count": len(user_data)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                "error": f"An unexpected error occurred: {e}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class GetPaymentsAPI(APIView):
    """
    Get all payments with optional filters.
    """
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Get all payments with optional filters.",
        manual_parameters=[
            openapi.Parameter(
                'user_id', 
                openapi.IN_QUERY,
                description="Filter by user ID",
                type=openapi.TYPE_INTEGER,
                required=False
            ),
            openapi.Parameter(
                'subscription_id', 
                openapi.IN_QUERY,
                description="Filter by subscription ID",
                type=openapi.TYPE_INTEGER,
                required=False
            ),
            openapi.Parameter(
                'paymentid', 
                openapi.IN_QUERY,
                description="Filter by payment ID",
                type=openapi.TYPE_INTEGER,
                required=False
            ),
            openapi.Parameter(
                'subscriptionplanid', 
                openapi.IN_QUERY,
                description="Filter by subscription plan ID",
                type=openapi.TYPE_INTEGER,
                required=False
            ),
            openapi.Parameter(
                'status', 
                openapi.IN_QUERY,
                description="Filter by payment status",
                type=openapi.TYPE_STRING,
                required=False
            ),
            openapi.Parameter(
                'payment_method', 
                openapi.IN_QUERY,
                description="Filter by payment method",
                type=openapi.TYPE_STRING,
                required=False
            ),
            openapi.Parameter(
                'from_date', 
                openapi.IN_QUERY,
                description="Filter payments from date (YYYY-MM-DD)",
                type=openapi.TYPE_STRING,
                format=openapi.FORMAT_DATE,
                required=False
            ),
            openapi.Parameter(
                'to_date', 
                openapi.IN_QUERY,
                description="Filter payments to date (YYYY-MM-DD)",
                type=openapi.TYPE_STRING,
                format=openapi.FORMAT_DATE,
                required=False
            ),
            openapi.Parameter(
                'limit', 
                openapi.IN_QUERY,
                description="Limit number of results",
                type=openapi.TYPE_INTEGER,
                required=False
            )
        ],
        responses={
            200: PaymentSerializer(many=True),
            400: "Bad Request",
            500: "Internal Server Error"
        }
    )
    def get(self, request):
        try:
            # Start with all active and non-deleted payments
            queryset = Payments.objects.filter(isactive=1, isdeleted=0)

            # Apply filters if provided
            user_id = request.query_params.get('user_id')
            if user_id:
                queryset = queryset.filter(subscriptionid__userid=user_id)

            subscription_id = request.query_params.get('subscription_id')
            if subscription_id:
                queryset = queryset.filter(subscriptionid=subscription_id)

            paymentid = request.query_params.get('paymentid')
            if paymentid:
                queryset = queryset.filter(paymentid=paymentid)

            subscriptionplanid = request.query_params.get('subscriptionplanid')
            if subscriptionplanid:
                queryset = queryset.filter(subscriptionid__subscriptionplanid=subscriptionplanid)

            payment_status = request.query_params.get('status')
            if payment_status:
                queryset = queryset.filter(status=payment_status)

            payment_method = request.query_params.get('payment_method')
            if payment_method:
                queryset = queryset.filter(paymentmethod=payment_method)

            from_date = request.query_params.get('from_date')
            if from_date:
                queryset = queryset.filter(paymentdate__gte=from_date)

            to_date = request.query_params.get('to_date')
            if to_date:
                queryset = queryset.filter(paymentdate__lte=to_date)

            # Apply limit
            limit = request.query_params.get('limit', 100)
            queryset = queryset.order_by('-paymentdate')[:int(limit)]

            # Serialize and return the data
            serializer = PaymentSerializer(queryset, many=True)
            
            return Response({
                "message": "Payments retrieved successfully",
                "count": queryset.count(),
                "data": serializer.data
            }, status=status.HTTP_200_OK)

        except ValueError as e:
            return Response({
                "error": f"Invalid parameter value: {str(e)}"
            }, status=status.HTTP_400_BAD_REQUEST)
            
        except Exception as e:
            return Response({
                "error": f"An unexpected error occurred: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class GetDashboardStatsAPI(APIView):
    """
    Get admin dashboard statistics.
    """
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Get admin dashboard statistics including users, subscriptions, trades, etc.",
        responses={
            200: AdminDashboardStatsSerializer,
            500: "Internal Server Error"
        }
    )
    def get(self, request):
        try:
            # Calculate date ranges
            now = timezone.now()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # Basic counts
            total_users = Users.objects.filter(isdeleted=0).count()
            active_subscriptions = Subscriptions.objects.filter(
                status='Active', isactive=1, isdeleted=0
            ).count()
            total_trades = UserTrade.objects.filter(isactive=1, isdeleted=0).count()
            
            # API usage this month
            total_api_calls = APIUsage.objects.filter(
                timestamp__gte=month_start, isactive=1, isdeleted=0
            ).aggregate(total=Sum('request_count'))['total'] or 0

            # Revenue this month
            revenue_this_month = Payments.objects.filter(
                paymentdate__gte=month_start.date(),
                status='Completed',
                isactive=1,
                isdeleted=0
            ).aggregate(total=Sum('amount'))['total'] or Decimal('0.00')

            # Top exchanges by user count
            top_exchanges = list(
                UserExchangeCredentials.objects.filter(isactive=1, isdeleted=0)
                .values('exchange__display_name')
                .annotate(user_count=Count('user', distinct=True))
                .order_by('-user_count')[:5]
            )

            # Subscription distribution
            subscription_distribution = {}
            plans = Subscriptionplans.objects.filter(isactive=1, isdeleted=0)
            for plan in plans:
                count = Subscriptions.objects.filter(
                    subscriptionplanid=plan,
                    status='Active',
                    isactive=1,
                    isdeleted=0
                ).count()
                subscription_distribution[plan.name] = count

            # Recent activities (last 10 trades)
            recent_activities = []
            recent_trades = UserTrade.objects.filter(
                isactive=1, isdeleted=0
            ).order_by('-created_at')[:10]
            
            for trade in recent_trades:
                recent_activities.append({
                    'type': 'trade',
                    'user': trade.user.fullname,
                    'description': f"{trade.side.upper()} {trade.amount} {trade.symbol}",
                    'timestamp': trade.created_at.isoformat(),
                    'status': trade.status
                })
            
            # Total portfolio value (simplified)
            total_portfolio_value = UserPortfolio.objects.filter(
                isactive=1, isdeleted=0
            ).aggregate(
                total=Sum('total_amount')
            )['total'] or Decimal('0.00')

            stats_data = {
                'total_users': total_users,
                'active_subscriptions': active_subscriptions,
                'total_trades': total_trades,
                'total_api_calls': total_api_calls,
                'revenue_this_month': revenue_this_month,
                'top_exchanges': top_exchanges,
                'subscription_distribution': subscription_distribution,
                'recent_activities': recent_activities,
                'total_portfolio_value': total_portfolio_value
            }

            serializer = AdminDashboardStatsSerializer(stats_data)
            
            return Response({
                "message": "Dashboard statistics retrieved successfully",
                "data": serializer.data
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({
                "error": f"An unexpected error occurred: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)