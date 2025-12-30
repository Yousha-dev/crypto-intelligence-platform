from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import logging
from myapp.permissions import IsUserAccess
from myapp.models import Users
from myapp.serializers.core_serializers import (
    PricePredictionSerializer
)
from myapp.services.subscription_service import SubscriptionService
from myapp.services.ai_predictor import PredictorService
 

logger = logging.getLogger(__name__)

class GenerateAIPredictionAPI(APIView):
    """Generate AI price prediction"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Generate AI price prediction for specified symbol.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'symbol': openapi.Schema(type=openapi.TYPE_STRING, description="Trading pair symbol"),
                'exchange': openapi.Schema(type=openapi.TYPE_STRING, description="Exchange name"),
                'timeframe': openapi.Schema(type=openapi.TYPE_STRING, description="Prediction timeframe"),
                'horizon_minutes': openapi.Schema(type=openapi.TYPE_INTEGER, description="Prediction horizon in minutes"),
            },
            required=['symbol']
        ),
        responses={200: "Prediction generated", 400: "Invalid request", 403: "Feature not available"},
    )
    def post(self, request):
        try:
            user_id = getattr(request, "user_id", None)
            if not user_id:
                return Response({"error": "Authentication required"}, status=status.HTTP_401_UNAUTHORIZED)
            
            user = Users.objects.get(userid=user_id)
            can_use_ai, message = SubscriptionService.can_use_ai_predictions(user)
            if not can_use_ai:
                return Response({
                    "error": "AI predictions not available in your current plan.",
                    "message": message
                }, status=status.HTTP_403_FORBIDDEN)

            data = request.data
            predictor = PredictorService(user)
            
            prediction = predictor.generate_prediction(
                symbol=data.get('symbol', 'BTC/USDT'),
                exchange=data.get('exchange', 'binance'),
                timeframe=data.get('timeframe', '1h'),
                horizon_minutes=data.get('horizon_minutes', 60)
            )
            
            if not prediction.get('success', False):
                return Response(prediction, status=status.HTTP_400_BAD_REQUEST)
            
            return Response({
                "message": "Prediction generated successfully",
                "data": prediction
            }, status=status.HTTP_200_OK)
        
        except Users.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error generating prediction for user {user_id}: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ListUserPredictionsAPI(APIView):
    """List user's AI predictions"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="List authenticated user's AI price predictions.",
        manual_parameters=[
            openapi.Parameter('limit', openapi.IN_QUERY, description="Number of predictions to return", type=openapi.TYPE_INTEGER, default=20),
            openapi.Parameter('symbol', openapi.IN_QUERY, description="Filter by symbol", type=openapi.TYPE_STRING, required=False)
        ],
        responses={200: PricePredictionSerializer(many=True), 404: "No predictions found"},
    )
    def get(self, request):
        try:
            user_id = getattr(request, "user_id", None)
            if not user_id:
                return Response({"error": "Authentication required"}, status=status.HTTP_401_UNAUTHORIZED)
            
            user = Users.objects.get(userid=user_id)
            can_use_ai, message = SubscriptionService.can_use_ai_predictions(user)
            if not can_use_ai:
                return Response({
                    "error": "AI predictions not available in your current plan.",
                    "message": message
                }, status=status.HTTP_403_FORBIDDEN)

            limit = int(request.GET.get('limit', 20))
            symbol = request.GET.get('symbol')
            
            predictor = PredictorService(user)
            result = predictor.get_user_predictions(limit=limit, symbol=symbol)
            
            if not result.get('success', False):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            
            return Response(result, status=status.HTTP_200_OK)
        
        except Users.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error fetching predictions for user {user_id}: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class GetUserPredictionStatsAPI(APIView):
    """Get user's prediction accuracy statistics"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Get user's AI prediction accuracy statistics.",
        responses={200: "Prediction statistics", 403: "Feature not available"},
    )
    def get(self, request):
        try:
            user_id = getattr(request, "user_id", None)
            if not user_id:
                return Response({"error": "Authentication required"}, status=status.HTTP_401_UNAUTHORIZED)
            
            user = Users.objects.get(userid=user_id)
            can_use_ai, message = SubscriptionService.can_use_ai_predictions(user)
            if not can_use_ai:
                return Response({
                    "error": "AI predictions not available in your current plan.",
                    "message": message
                }, status=status.HTTP_403_FORBIDDEN)

            predictor = PredictorService(user)
            result = predictor.get_prediction_accuracy_stats()
            
            if not result.get('success', False):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            
            return Response(result, status=status.HTTP_200_OK)
        
        except Users.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error fetching prediction stats for user {user_id}: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class GetUserPredictionLimitsAPI(APIView):
    """Get user's prediction limits and usage"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Get user's current prediction limits and usage information.",
        responses={200: "Prediction limits and usage", 403: "Feature not available"},
    )
    def get(self, request):
        try:
            user_id = getattr(request, "user_id", None)
            if not user_id:
                return Response({"error": "Authentication required"}, status=status.HTTP_401_UNAUTHORIZED)

            user = Users.objects.get(userid=user_id)
            predictor = PredictorService(user)
            result = predictor.get_user_prediction_limits()
            
            if not result.get('success', False):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            
            return Response(result, status=status.HTTP_200_OK)
        
        except Users.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error fetching prediction limits for user {user_id}: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)