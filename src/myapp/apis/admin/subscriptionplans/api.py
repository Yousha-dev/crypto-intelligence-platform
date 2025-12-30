from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from myapp.models import Subscriptionplans, Subscriptions, Payments
from myapp.serializers.admin_serializers import SubscriptionPlanSerializer
from myapp.permissions import IsUserAccess
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.utils import timezone
from django.db.models import Avg, Sum, Count, Q, Min, Max
from decimal import Decimal

### 1. Create SubscriptionPlan API ###
class CreateSubscriptionPlanAPI(APIView):
    """
    Create a new subscription plan.
    """
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Create a new subscription plan.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'name': openapi.Schema(type=openapi.TYPE_STRING, description="Plan name"),
                'description': openapi.Schema(type=openapi.TYPE_STRING, description="Plan description"),
                'monthlyprice': openapi.Schema(type=openapi.TYPE_NUMBER, description="Monthly price"),
                'yearlyprice': openapi.Schema(type=openapi.TYPE_NUMBER, description="Yearly price"),
                'max_exchanges': openapi.Schema(type=openapi.TYPE_INTEGER, description="Maximum exchanges allowed"),
                'max_api_calls_per_hour': openapi.Schema(type=openapi.TYPE_INTEGER, description="Max API calls per hour"),
                'ai_predictions_enabled': openapi.Schema(type=openapi.TYPE_BOOLEAN, description="AI predictions enabled"),
                'advanced_indicators_enabled': openapi.Schema(type=openapi.TYPE_BOOLEAN, description="Advanced indicators"),
                'portfolio_tracking': openapi.Schema(type=openapi.TYPE_BOOLEAN, description="Portfolio tracking"),
                'trade_automation': openapi.Schema(type=openapi.TYPE_BOOLEAN, description="Trade automation"),
                'featuredetails': openapi.Schema(type=openapi.TYPE_STRING, description="Detailed features")
            },
            required=['name', 'monthlyprice', 'max_exchanges', 'max_api_calls_per_hour']
        ),
        responses={
            201: SubscriptionPlanSerializer,
            400: "Validation errors"
        }
    )
    def post(self, request):
        data = request.data.copy()
        data["isactive"] = 1
        data["isdeleted"] = 0
        data["createdby"] = getattr(request, "user_id", None)

        serializer = SubscriptionPlanSerializer(data=data)
        if serializer.is_valid():
            plan = serializer.save()
            return Response(
                {
                    "message": "Subscription plan created successfully.",
                    "data": serializer.data
                },
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ListSubscriptionPlanAPI(APIView):
    """
    List all subscription plans with optional filters.
    """
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="List all subscription plans with optional filters.",
        manual_parameters=[
            openapi.Parameter(
                'subscriptionplanid', 
                openapi.IN_QUERY,
                description="Filter by subscription plan ID",
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
                'userid', 
                openapi.IN_QUERY,
                description="Filter by User ID",
                type=openapi.TYPE_INTEGER,
                required=False
            ),
            openapi.Parameter(
                'active_only', 
                openapi.IN_QUERY,
                description="Show only active plans",
                type=openapi.TYPE_BOOLEAN,
                required=False
            ),
            openapi.Parameter(
                'min_price', 
                openapi.IN_QUERY,
                description="Filter by minimum monthly price",
                type=openapi.TYPE_NUMBER,
                required=False
            ),
            openapi.Parameter(
                'max_price', 
                openapi.IN_QUERY,
                description="Filter by maximum monthly price",
                type=openapi.TYPE_NUMBER,
                required=False
            )
        ],
        responses={200: SubscriptionPlanSerializer(many=True), 400: "Bad Request"},
    )
    def get(self, request):
        subscriptionplanid = request.query_params.get('subscriptionplanid')
        subscriptionid = request.query_params.get('subscriptionid')
        paymentid = request.query_params.get('paymentid')
        userid = request.query_params.get('userid')
        active_only = request.query_params.get('active_only', '').lower() == 'true'
        min_price = request.query_params.get('min_price')
        max_price = request.query_params.get('max_price')

        filters = {'isdeleted': 0}

        if active_only:
            filters['isactive'] = 1

        if subscriptionplanid:
            filters['subscriptionplanid'] = subscriptionplanid

        if min_price:
            filters['monthlyprice__gte'] = Decimal(min_price)

        if max_price:
            filters['monthlyprice__lte'] = Decimal(max_price)

        plans = Subscriptionplans.objects.filter(**filters)

        if subscriptionid:
            plans = plans.filter(subscriptions__subscriptionid=subscriptionid)

        if paymentid:
            plans = plans.filter(
                subscriptions__payments__paymentid=paymentid,
                subscriptions__payments__isdeleted=0
            )

        if userid:
            plans = plans.filter(
                subscriptions__userid=userid,
                subscriptions__isdeleted=0
            )

        plans = plans.distinct().order_by('monthlyprice')

        serializer = SubscriptionPlanSerializer(plans, many=True)
        return Response({
            "message": "Subscription plans retrieved successfully",
            "count": plans.count(),
            "data": serializer.data
        }, status=status.HTTP_200_OK)

### 3. Update SubscriptionPlan API ###
class UpdateSubscriptionPlanAPI(APIView):
    """
    Update an existing subscription plan.
    """
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Update an existing subscription plan.",
        request_body=SubscriptionPlanSerializer,
        responses={
            200: "SubscriptionPlan updated successfully", 
            404: "Plan not found", 
            400: "Validation errors"
        }
    )
    def put(self, request, subscriptionplan_id):
        try:
            plan = Subscriptionplans.objects.get(
                subscriptionplanid=subscriptionplan_id, 
                isdeleted=0
            )
            
            data = request.data.copy()
            data["updatedby"] = getattr(request, "user_id", None)
            
            serializer = SubscriptionPlanSerializer(plan, data=data, partial=True)
            if serializer.is_valid():
                updated_plan = serializer.save()
                return Response(
                    {
                        "message": "Subscription plan updated successfully.",
                        "data": serializer.data
                    },
                    status=status.HTTP_200_OK
                )
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Subscriptionplans.DoesNotExist:
            return Response(
                {"error": "Subscription plan not found."}, 
                status=status.HTTP_404_NOT_FOUND
            )

### 4. Delete SubscriptionPlan API ###
class DeleteSubscriptionPlanAPI(APIView):
    """
    Soft delete a subscription plan.
    """
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Soft delete a subscription plan.",
        responses={
            200: "SubscriptionPlan deleted successfully", 
            404: "SubscriptionPlan not found",
            400: "Cannot delete plan with active subscriptions"
        }
    )
    def delete(self, request, subscriptionplan_id):
        try:
            plan = Subscriptionplans.objects.get(
                subscriptionplanid=subscriptionplan_id, 
                isdeleted=0
            )
            
            # Check if there are active subscriptions using this plan
            active_subscriptions = Subscriptions.objects.filter(
                subscriptionplanid=plan,
                status='Active',
                isactive=1,
                isdeleted=0
            ).count()
            
            if active_subscriptions > 0:
                return Response({
                    "error": f"Cannot delete plan. {active_subscriptions} active subscription(s) are using this plan."
                }, status=status.HTTP_400_BAD_REQUEST)
            
            plan.isdeleted = 1
            plan.isactive = 0  # Also mark as inactive
            plan.updatedby = getattr(request, "user_id", None)
            plan.updatedat = timezone.now()
            plan.save()
            
            return Response({
                "message": "Subscription plan deleted successfully."
            }, status=status.HTTP_200_OK)
        except Subscriptionplans.DoesNotExist:
            return Response(
                {"error": "Subscription plan not found."}, 
                status=status.HTTP_404_NOT_FOUND
            )

### 5. SubscriptionPlan Analytics API ###
class SubscriptionPlanAnalyticsAPI(APIView):
    """
    Fetch analytics data for subscription plans.
    """
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Fetch comprehensive analytics data for subscription plans.",
        responses={200: "Analytics data retrieved successfully", 400: "Bad Request"}
    )
    def get(self, request):
        plans = Subscriptionplans.objects.filter(isdeleted=0)
        active_plans = plans.filter(isactive=1)
        
        # Basic counts
        total_plans = plans.count()
        active_plans_count = active_plans.count()
        
        # Price statistics
        price_stats = active_plans.aggregate(
            avg_monthly_price=Avg('monthlyprice'),
            min_monthly_price=plans.aggregate(min_price=Min('monthlyprice'))['min_price'],
            max_monthly_price=plans.aggregate(max_price=Max('monthlyprice'))['max_price'],
            avg_yearly_price=Avg('yearlyprice'),
            total_monthly_revenue_potential=Sum('monthlyprice')
        )

        # Feature distribution
        feature_stats = active_plans.aggregate(
            ai_enabled_plans=Count('subscriptionplanid', filter=Q(ai_predictions_enabled=True)),
            trade_automation_plans=Count('subscriptionplanid', filter=Q(trade_automation=True)),
            portfolio_tracking_plans=Count('subscriptionplanid', filter=Q(portfolio_tracking=True)),
            advanced_indicators_plans=Count('subscriptionplanid', filter=Q(advanced_indicators_enabled=True))
        )

        # Exchange limits distribution
        exchange_limits = active_plans.values('max_exchanges').annotate(
            plan_count=Count('subscriptionplanid')
        ).order_by('max_exchanges')

        # API rate limits distribution
        api_limits = active_plans.values('max_api_calls_per_hour').annotate(
            plan_count=Count('subscriptionplanid')
        ).order_by('max_api_calls_per_hour')

        # Subscription usage per plan
        plan_usage = []
        for plan in active_plans:
            active_subscriptions = Subscriptions.objects.filter(
                subscriptionplanid=plan,
                status='Active',
                isactive=1,
                isdeleted=0
            ).count()
            
            total_subscriptions = Subscriptions.objects.filter(
                subscriptionplanid=plan,
                isdeleted=0
            ).count()
            
            plan_usage.append({
                'plan_id': plan.subscriptionplanid,
                'plan_name': plan.name,
                'monthly_price': float(plan.monthlyprice),
                'active_subscriptions': active_subscriptions,
                'total_subscriptions': total_subscriptions,
                'monthly_revenue': float(plan.monthlyprice * active_subscriptions)
            })

        # Revenue analysis
        total_monthly_revenue = sum(item['monthly_revenue'] for item in plan_usage)
        
        return Response({
            "analytics": {
                "overview": {
                    "total_plans": total_plans,
                    "active_plans": active_plans_count,
                    "inactive_plans": total_plans - active_plans_count
                },
                "pricing": {
                    "average_monthly_price": float(price_stats['avg_monthly_price'] or 0),
                    "minimum_monthly_price": float(price_stats['min_monthly_price'] or 0),
                    "maximum_monthly_price": float(price_stats['max_monthly_price'] or 0),
                    "average_yearly_price": float(price_stats['avg_yearly_price'] or 0),
                    "total_monthly_revenue": total_monthly_revenue
                },
                "features": {
                    "ai_predictions_enabled": feature_stats['ai_enabled_plans'],
                    "trade_automation_enabled": feature_stats['trade_automation_plans'],
                    "portfolio_tracking_enabled": feature_stats['portfolio_tracking_plans'],
                    "advanced_indicators_enabled": feature_stats['advanced_indicators_plans']
                },
                "limits": {
                    "exchange_limits_distribution": list(exchange_limits),
                    "api_limits_distribution": list(api_limits)
                },
                "usage": plan_usage
            }
        }, status=status.HTTP_200_OK)

### 6. SubscriptionPlan Dashboard Overview API ###
class SubscriptionPlanDashboardOverviewAPI(APIView):
    """
    Get dashboard overview for subscription plans.
    """
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Get dashboard overview for subscription plans with key metrics.",
        responses={200: "Dashboard data retrieved successfully", 400: "Bad Request"}
    )
    def get(self, request):
        plans = Subscriptionplans.objects.filter(isdeleted=0)
        active_plans = plans.filter(isactive=1)
        
        total_plans = plans.count()
        active_plans_count = active_plans.count()
        
        # Get subscription counts and revenue for each plan
        plan_performance = []
        total_active_subscriptions = 0
        total_monthly_revenue = Decimal('0.00')
        
        for plan in active_plans:
            active_subs = Subscriptions.objects.filter(
                subscriptionplanid=plan,
                status='Active',
                isactive=1,
                isdeleted=0
            ).count()
            
            monthly_revenue = plan.monthlyprice * active_subs
            total_active_subscriptions += active_subs
            total_monthly_revenue += monthly_revenue
            
            plan_performance.append({
                'plan_id': plan.subscriptionplanid,
                'plan_name': plan.name,
                'monthly_price': float(plan.monthlyprice),
                'yearly_price': float(plan.yearlyprice) if plan.yearlyprice else 0,
                'active_subscriptions': active_subs,
                'monthly_revenue': float(monthly_revenue),
                'max_exchanges': plan.max_exchanges,
                'max_api_calls': plan.max_api_calls_per_hour,
                'features': {
                    'ai_predictions': plan.ai_predictions_enabled,
                    'trade_automation': plan.trade_automation,
                    'portfolio_tracking': plan.portfolio_tracking,
                    'advanced_indicators': plan.advanced_indicators_enabled
                }
            })
        
        # Sort by revenue (highest first)
        plan_performance.sort(key=lambda x: x['monthly_revenue'], reverse=True)
        
        # Get most popular plan
        most_popular_plan = max(plan_performance, key=lambda x: x['active_subscriptions']) if plan_performance else None
        
        # Get highest revenue plan
        highest_revenue_plan = plan_performance[0] if plan_performance else None
        
        # Calculate average metrics
        avg_price = sum(p['monthly_price'] for p in plan_performance) / len(plan_performance) if plan_performance else 0
        
        return Response({
            "overview": {
                "total_plans": total_plans,
                "active_plans": active_plans_count,
                "inactive_plans": total_plans - active_plans_count,
                "total_active_subscriptions": total_active_subscriptions,
                "total_monthly_revenue": float(total_monthly_revenue),
                "average_plan_price": round(avg_price, 2)
            },
            "top_performers": {
                "most_popular_plan": most_popular_plan,
                "highest_revenue_plan": highest_revenue_plan
            },
            "plan_performance": plan_performance[:10]  # Top 10 plans
        }, status=status.HTTP_200_OK)