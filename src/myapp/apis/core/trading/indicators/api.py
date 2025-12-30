# indicators/api.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import logging
from django.conf import settings
from myapp.permissions import IsUserAccess
from myapp.models import Users
from myapp.services.influx_manager import InfluxManager
from myapp.services.technical_analysis import TechnicalAnalysis

 
logger = logging.getLogger(__name__)

class GetAvailableIndicatorsAPI(APIView):
    """Get available technical indicators for user"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Get list of technical indicators available based on user's subscription.",
        responses={200: "Available indicators", 500: "Error"},
    )
    def get(self, request):
        try:
            user_id = getattr(request, "user_id", None)
            if not user_id:
                return Response({"error": "Authentication required"}, status=status.HTTP_401_UNAUTHORIZED)
            
            user = Users.objects.get(userid=user_id)
            result = TechnicalAnalysis.get_available_indicators(user)
            
            if not result.get('success', False):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            
            return Response(result, status=status.HTTP_200_OK)
            
        except Users.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error fetching available indicators for user {user_id}: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class GetTechnicalSignalsAPI(APIView):
    """Get trading signals for a symbol"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Get trading signals based on technical indicators for a symbol.",
        manual_parameters=[
            openapi.Parameter('symbol', openapi.IN_QUERY, description="Trading pair symbol", type=openapi.TYPE_STRING, default='BTC/USDT'),
            openapi.Parameter('exchange', openapi.IN_QUERY, description="Exchange name", type=openapi.TYPE_STRING, default='binance'),
            openapi.Parameter('timeframe', openapi.IN_QUERY, description="Chart timeframe", type=openapi.TYPE_STRING, default='1h')
        ],
        responses={200: "Trading signals", 404: "No data found"},
    )
    def get(self, request):
        user_id = getattr(request, "user_id", None)
        symbol = request.GET.get('symbol', 'BTC/USDT')
        exchange = request.GET.get('exchange', 'binance')
        timeframe = request.GET.get('timeframe', '1h')
        
        try:
            if not user_id:
                return Response({"error": "Authentication required"}, status=status.HTTP_401_UNAUTHORIZED)
            
            if not hasattr(settings, 'INFLUXDB_CONFIG'):
                return Response({'error': 'Technical analysis not available'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

            user = Users.objects.get(userid=user_id)

            with InfluxManager() as influx_manager:
                df = influx_manager.get_ohlcv_data(exchange, symbol, timeframe, 168)
                
                if df.empty or len(df) < 20:
                    return Response({'error': 'Insufficient data for analysis'}, status=status.HTTP_404_NOT_FOUND)
                
                # Calculate indicators and signals with user object
                indicators = TechnicalAnalysis.calculate_indicators(df, user)
                signals = TechnicalAnalysis.generate_signals(indicators, user)
                
                return Response({
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'signals': signals,
                    'key_indicators': {
                        'rsi': indicators.get('rsi'),
                        'current_price': indicators.get('current_price'),
                        'volatility': indicators.get('volatility')
                    }
                }, status=status.HTTP_200_OK)
        
        except Users.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error generating signals for user {user_id}: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)