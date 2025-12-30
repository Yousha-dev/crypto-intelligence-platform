"""
LLM Provider Management API
Dynamic switching and configuration of LLM providers
"""

import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from myapp.permissions import IsUserAccess
from myapp.services.rag.llm_provider import get_llm_manager, LLMProvider, LLMConfig
 
logger = logging.getLogger(__name__)
 

class LLMStatusAPI(APIView):
    """Get LLM provider status"""
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        operation_description="Get status of all configured LLM providers.",
        responses={
            200: openapi.Response(
                description="LLM provider status",
                examples={
                    "application/json": {
                        "active_provider": "ollama",
                        "providers": {
                            "openai": {"available": True, "model": "gpt-4o-mini"},
                            "anthropic": {"available": True, "model": "claude-3-haiku"},
                            "ollama": {"available": True, "model": "llama3.1"},
                            "groq": {"available": True, "model": "llama-3.1-70b"}
                        }
                    }
                }
            )
        },
        tags=['LLM Management']
    )
    def get(self, request):
        try:
            manager = get_llm_manager()
            status_info = manager.get_status()
            
            return Response(status_info, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting LLM status: {e}")
            return Response({
                'error': 'Failed to get LLM status',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SwitchLLMProviderAPI(APIView):
    """Switch active LLM provider"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Switch to a different LLM provider.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'provider': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="Provider to switch to",
                    enum=['openai', 'anthropic', 'ollama', 'groq']
                )
            },
            required=['provider']
        ),
        responses={
            200: "Provider switched successfully",
            400: "Invalid provider or provider not available"
        },
        tags=['LLM Management']
    )
    def post(self, request):
        try:
            provider = request.data.get('provider')
            
            if not provider:
                return Response({
                    'error': 'Provider is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            manager = get_llm_manager()
            
            try:
                target = LLMProvider(provider)
            except ValueError:
                return Response({
                    'error': f'Invalid provider: {provider}',
                    'valid_providers': ['openai', 'anthropic', 'ollama', 'groq']
                }, status=status.HTTP_400_BAD_REQUEST)
            
            success = manager.switch_provider(target)
            
            if success:
                return Response({
                    'status': 'success',
                    'active_provider': provider,
                    'message': f'Switched to {provider}'
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    'error': f'Provider {provider} is not available'
                }, status=status.HTTP_400_BAD_REQUEST)
            
        except Exception as e:
            logger.error(f"Error switching provider: {e}")
            return Response({
                'error': 'Failed to switch provider',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class TestLLMProviderAPI(APIView):
    """Test LLM provider with a sample prompt"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Test an LLM provider with a sample prompt.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'provider': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="Provider to test (optional, uses active if not specified)",
                    enum=['openai', 'anthropic', 'ollama', 'groq']
                ),
                'prompt': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="Test prompt",
                    default="What is Bitcoin?"
                )
            }
        ),
        responses={200: "Test results"},
        tags=['LLM Management']
    )
    def post(self, request):
        try:
            provider = request.data.get('provider')
            prompt = request.data.get('prompt', 'What is Bitcoin? Answer in one sentence.')
            
            manager = get_llm_manager()
            
            target_provider = None
            if provider:
                try:
                    target_provider = LLMProvider(provider)
                except ValueError:
                    return Response({
                        'error': f'Invalid provider: {provider}'
                    }, status=status.HTTP_400_BAD_REQUEST)
            
            import time
            start_time = time.time()
            
            response, tokens, provider_used = manager.generate(
                prompt,
                system_prompt="You are a helpful assistant. Be concise.",
                provider=target_provider
            )
            
            elapsed = time.time() - start_time
            
            return Response({
                'status': 'success',
                'provider_used': provider_used,
                'response': response,
                'tokens_used': tokens,
                'latency_seconds': round(elapsed, 3)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error testing provider: {e}")
            return Response({
                'error': 'Test failed',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)