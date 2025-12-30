from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from myapp.permissions import IsUserAccess
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import logging
from rest_framework.permissions import AllowAny
from django.db import transaction
from django.conf import settings
from myapp.models import (
    Users, Exchange, UserTrade, UserPortfolio
)
from myapp.serializers.core_serializers import (
    ExchangeSerializer,
    UserExchangeCredentialsSerializer, UserTradeSerializer
)
from myapp.services.subscription_service import SubscriptionService
from myapp.services.influx_manager import InfluxManager


logger = logging.getLogger(__name__)

class ListExchangesAPI(APIView):
    """List all active exchanges"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="List all active exchanges available on the platform.",
        responses={200: ExchangeSerializer(many=True), 400: "Bad Request."},
    )
    def get(self, request):
        exchanges = Exchange.objects.filter(isactive=1, isdeleted=0)
        serializer = ExchangeSerializer(exchanges, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

