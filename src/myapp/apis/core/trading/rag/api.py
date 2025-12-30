"""
RAG System API Endpoints using LlamaIndex
Semantic search, Q&A, and AI-powered analysis
"""

import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from datetime import datetime
from django.utils import timezone

from myapp.services.rag.query_chain import get_query_chain
from myapp.services.rag.context_manager import get_context_manager
from myapp.services.rag.post_processor import get_postprocessor
 
from myapp.permissions import IsUserAccess
from myapp.services.rag.rag_service import get_rag_engine

logger = logging.getLogger(__name__)


class SemanticSearchAPI(APIView):
    """Semantic search across news and social content"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Perform semantic search using LlamaIndex.",
        manual_parameters=[
            openapi.Parameter('q', openapi.IN_QUERY, 
                description="Natural language search query", 
                type=openapi.TYPE_STRING, required=True),
            openapi.Parameter('limit', openapi.IN_QUERY, 
                description="Number of results", 
                type=openapi.TYPE_INTEGER, default=10),
            openapi.Parameter('type', openapi.IN_QUERY, 
                description="Filter by content type", 
                type=openapi.TYPE_STRING,
                enum=['all', 'news', 'social'], default='all'),
            openapi.Parameter('min_trust', openapi.IN_QUERY, 
                description="Minimum trust score", 
                type=openapi.TYPE_NUMBER, default=0),
        ],
        responses={200: "Search results"},
        tags=['RAG - Search']
    )
    def get(self, request):
        try:
            query = request.GET.get('q', '')
            limit = int(request.GET.get('limit', 10))
            content_type = request.GET.get('type', 'all')
            min_trust = float(request.GET.get('min_trust', 0))
            
            if not query:
                return Response({
                    'error': 'Search query is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            rag_engine = get_rag_engine()
            
            # Build filters
            filters = {}
            if content_type != 'all':
                filters['type'] = content_type
            
            # Perform semantic search - returns List[SearchResult]
            results = rag_engine.retrieve(query, top_k=limit * 2, filters=filters)
            
            # Filter by trust score and format results
            formatted_results = []
            for result in results:
                # SearchResult has: id, content, metadata, score, rank
                trust_score = result.metadata.get('trust_score', 0)
                if trust_score < min_trust:
                    continue
                
                formatted_results.append({
                    'id': result.id,
                    'title': result.metadata.get('title', 'Untitled'),
                    'source': result.metadata.get('source', 'Unknown'),
                    'platform': result.metadata.get('platform', 'unknown'),
                    'relevance_score': round(result.score, 3),
                    'trust_score': trust_score,
                    'sentiment': result.metadata.get('sentiment', 'neutral'),
                    'url': result.metadata.get('url', ''),
                    'snippet': result.content[:300] + '...' if len(result.content) > 300 else result.content,
                    'type': result.metadata.get('type', 'news')
                })
                
                if len(formatted_results) >= limit:
                    break
            
            return Response({
                'query': query,
                'results': formatted_results,
                'total': len(formatted_results),
                'filters_applied': {'type': content_type, 'min_trust': min_trust},
                'engine': 'LlamaIndex'
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return Response({
                'error': 'Search failed',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AskQuestionAPI(APIView):
    """Ask questions with AI-powered answers"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Ask questions using LlamaIndex RAG.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'question': openapi.Schema(type=openapi.TYPE_STRING),
                'use_knowledge_graph': openapi.Schema(type=openapi.TYPE_BOOLEAN, default=False)
            },
            required=['question']
        ),
        responses={200: "AI-generated answer"},
        tags=['RAG - Q&A']
    )
    def post(self, request):
        try:
            data = request.data
            question = data.get('question', '')
            use_kg = data.get('use_knowledge_graph', False)
            
            if not question:
                return Response({
                    'error': 'Question is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            rag_engine = get_rag_engine()
            
            # Generate answer (with or without KG)
            if use_kg:
                response = rag_engine.generate_answer_with_kg(question)
            else:
                response = rag_engine.generate_answer(question)
            
            return Response({
                'question': response.query,
                'answer': response.answer,
                'sources': response.sources,
                'confidence': response.confidence,
                'tokens_used': response.tokens_used,
                'processing_time': response.processing_time,
                'model_used': response.model_used,
                'engine': 'LlamaIndex'
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return Response({
                'error': 'Failed to generate answer',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatWithNewsAPI(APIView):
    """Interactive chat about crypto news"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Chat about cryptocurrency news.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'message': openapi.Schema(type=openapi.TYPE_STRING),
                'crypto_focus': openapi.Schema(type=openapi.TYPE_STRING)
            },
            required=['message']
        ),
        responses={200: "Chat response"},
        tags=['RAG - Q&A']
    )
    def post(self, request):
        try:
            data = request.data
            message = data.get('message', '')
            crypto_focus = data.get('crypto_focus')
            
            if not message:
                return Response({
                    'error': 'Message is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            rag_engine = get_rag_engine()
            
            # Build query
            query = f"{crypto_focus}: {message}" if crypto_focus else message
            
            # Custom system prompt
            system_prompt = """You are a helpful cryptocurrency news assistant. 
Be conversational but informative. Always cite sources when making claims.
If you don't know something, say so. Focus on factual information."""
            
            response = rag_engine.generate_answer(query, system_prompt=system_prompt)
            
            return Response({
                'message': message,
                'response': response.answer,
                'sources': response.sources[:3],
                'confidence': response.confidence,
                'crypto_focus': crypto_focus
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return Response({
                'error': 'Chat failed',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GenerateSummaryAPI(APIView):
    """Generate AI summary for a topic"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Generate summary using LlamaIndex.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'topic': openapi.Schema(type=openapi.TYPE_STRING),
                'hours_back': openapi.Schema(type=openapi.TYPE_INTEGER, default=24),
                'min_trust_score': openapi.Schema(type=openapi.TYPE_NUMBER, default=6.0)
            },
            required=['topic']
        ),
        responses={200: "Generated summary"},
        tags=['RAG - Analysis']
    )
    def post(self, request):
        try:
            data = request.data
            topic = data.get('topic', '')
            hours_back = int(data.get('hours_back', 24))
            min_trust = float(data.get('min_trust_score', 6.0))
            
            if not topic:
                return Response({
                    'error': 'Topic is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            rag_engine = get_rag_engine()
            summary = rag_engine.generate_summary(topic, hours_back, min_trust)
            
            return Response(summary, status=status.HTTP_200_OK)
              
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return Response({
                'error': 'Failed to generate summary',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AnalyzeEntityAPI(APIView):
    """Analyze news coverage for an entity"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Analyze entity using LlamaIndex.",
        manual_parameters=[
            openapi.Parameter('entity', openapi.IN_QUERY, type=openapi.TYPE_STRING, required=True),
            openapi.Parameter('entity_type', openapi.IN_QUERY, type=openapi.TYPE_STRING, default='cryptocurrency'),
        ],
        responses={200: "Entity analysis"},
        tags=['RAG - Analysis']
    )
    def get(self, request):
        try:
            entity = request.GET.get('entity', '')
            entity_type = request.GET.get('entity_type', 'cryptocurrency')
            
            if not entity:
                return Response({
                    'error': 'Entity is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            rag_engine = get_rag_engine()
            analysis = rag_engine.analyze_entity(entity, entity_type)
            
            return Response(analysis, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error analyzing entity: {e}")
            return Response({
                'error': 'Analysis failed',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class MarketSentimentAnalysisAPI(APIView):
    """Get AI-powered market sentiment"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get market sentiment analysis.",
        manual_parameters=[
            openapi.Parameter('crypto', openapi.IN_QUERY, type=openapi.TYPE_STRING),
            openapi.Parameter('hours', openapi.IN_QUERY, type=openapi.TYPE_INTEGER, default=24),
        ],
        responses={200: "Sentiment analysis"},
        tags=['RAG - Analysis']
    )
    def get(self, request):
        try:
            crypto = request.GET.get('crypto')
            hours = int(request.GET.get('hours', 24))
            
            rag_engine = get_rag_engine()
            
            query = f"{crypto} market sentiment" if crypto else "cryptocurrency market sentiment"
            results = rag_engine.retrieve(query, top_k=15)
            
            if not results:
                return Response({
                    'sentiment': 'neutral',
                    'confidence': 0,
                    'message': 'No recent news found'
                }, status=status.HTTP_200_OK)
            
            # Calculate sentiment
            sentiment_scores = []
            for r in results:
                sentiment = r.metadata.get('sentiment', 'neutral').lower()
                if sentiment in ['bullish', 'positive']:
                    sentiment_scores.append(1)
                elif sentiment in ['bearish', 'negative']:
                    sentiment_scores.append(-1)
                else:
                    sentiment_scores.append(0)
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            if avg_sentiment > 0.2:
                overall = 'bullish'
            elif avg_sentiment < -0.2:
                overall = 'bearish'
            else:
                overall = 'neutral'
            
            return Response({
                'overall_sentiment': overall,
                'sentiment_score': round(avg_sentiment, 3),
                'confidence': round(abs(avg_sentiment), 3),
                'articles_analyzed': len(results),
                'sentiment_distribution': {
                    'bullish': sentiment_scores.count(1),
                    'neutral': sentiment_scores.count(0),
                    'bearish': sentiment_scores.count(-1)
                },
                'crypto_filter': crypto,
                'time_window_hours': hours,
                'top_sources': [
                    {
                        'title': r.metadata.get('title', ''),
                        'source': r.metadata.get('source', ''),
                        'sentiment': r.metadata.get('sentiment', 'neutral'),
                        'trust_score': r.metadata.get('trust_score', 0)
                    }
                    for r in results[:5]
                ]
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return Response({
                'error': 'Analysis failed',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            

class ExecuteQueryChainAPI(APIView):
    """Execute multi-step query chain for complex questions"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Execute a multi-step query chain for complex analysis.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'query': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="Complex query to analyze"
                ),
                'chain_type': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="Chain type: auto, simple, deep_analysis, comparison",
                    enum=['auto', 'simple', 'deep_analysis', 'comparison'],
                    default='auto'
                ),
                'session_id': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="Session ID for context tracking (optional)"
                )
            },
            required=['query']
        ),
        responses={
            200: openapi.Response(
                description="Chain execution result",
                examples={
                    "application/json": {
                        "query": "Compare Bitcoin vs Ethereum performance",
                        "chain_type": "comparison",
                        "answer": "...",
                        "steps": [
                            {"step_type": "analyze", "duration_ms": 50},
                            {"step_type": "retrieve", "duration_ms": 120},
                            {"step_type": "compare", "duration_ms": 800}
                        ],
                        "total_duration_ms": 970,
                        "success": True
                    }
                }
            )
        },
        tags=['RAG - Advanced']
    )
    def post(self, request):
        try:
            data = request.data
            query = data.get('query', '')
            chain_type = data.get('chain_type', 'auto')
            session_id = data.get('session_id')
            
            if not query:
                return Response({
                    'error': 'Query is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Execute query chain
            query_chain = get_query_chain()
            result = query_chain.execute_chain(query, chain_type=chain_type)
            
            # Update session context if provided
            if session_id and result.success:
                context_manager = get_context_manager()
                context_manager.update_session(
                    session_id=session_id,
                    query=query,
                    answer=result.final_answer,
                    sources=[]
                )
            
            # Format steps for response
            steps = [
                {
                    'step_type': step.step_type.value,
                    'input': str(step.input_data)[:100] if step.input_data else None,
                    'output': step.output_data,
                    'duration_ms': round(step.duration_ms, 2),
                    'success': step.success
                }
                for step in result.steps
            ]
            
            return Response({
                'query': query,
                'chain_type': result.metadata.get('chain_type', chain_type),
                'answer': result.final_answer,
                'steps': steps,
                'step_count': len(steps),
                'total_duration_ms': round(result.total_duration_ms, 2),
                'success': result.success
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error executing query chain: {e}")
            return Response({
                'error': 'Query chain execution failed',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatWithContextAPI(APIView):
    """Chat with session-based context for in-context learning"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Chat with RAG system using session context for follow-up questions.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'query': openapi.Schema(type=openapi.TYPE_STRING, description="User query"),
                'session_id': openapi.Schema(type=openapi.TYPE_STRING, description="Session ID for context"),
                'use_context': openapi.Schema(type=openapi.TYPE_BOOLEAN, default=True),
                'format': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    enum=['detailed', 'brief', 'structured'],
                    default='detailed'
                )
            },
            required=['query', 'session_id']
        ),
        responses={200: "Chat response with context"},
        tags=['RAG - Chat']
    )
    def post(self, request):
        try:
            data = request.data
            query = data.get('query', '')
            session_id = data.get('session_id', '')
            use_context = data.get('use_context', True)
            output_format = data.get('format', 'detailed')
            
            if not query or not session_id:
                return Response({
                    'error': 'Query and session_id are required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            rag_engine = get_rag_engine()
            context_manager = get_context_manager()
            postprocessor = get_postprocessor()
            
            # Check for cached result
            cached = context_manager.get_cached_result(query)
            if cached:
                return Response({
                    **cached,
                    'cached': True
                }, status=status.HTTP_200_OK)
            
            # Build context-aware prompt if enabled
            system_prompt = None
            if use_context:
                base_prompt = """You are a cryptocurrency news analyst assistant."""
                system_prompt = context_manager.build_adaptive_prompt(
                    query=query,
                    session_id=session_id,
                    base_system_prompt=base_prompt
                )
            
            # Generate answer
            import time
            start_time = time.time()
            
            response = rag_engine.generate_answer(
                query=query,
                system_prompt=system_prompt
            )
            
            latency = time.time() - start_time
            
            # Post-process and format output
            formatted = postprocessor.format_output(
                answer=response.answer,
                sources=response.sources,
                format_type=output_format
            )
            
            # Update session
            context_manager.update_session(
                session_id=session_id,
                query=query,
                answer=response.answer,
                sources=response.sources
            )
            
            # Record performance
            context_manager.record_performance(
                query=query,
                latency=latency,
                tokens_used=response.tokens_used,
                success=True
            )
            
            # Cache result
            context_manager.cache_query_result(query, formatted, ttl=1800)
            
            return Response({
                **formatted,
                'session_id': session_id,
                'tokens_used': response.tokens_used,
                'latency_seconds': round(latency, 3),
                'model_used': response.model_used,
                'cached': False
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in contextual chat: {e}")
            return Response({
                'error': 'Chat failed',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RAGPerformanceStatsAPI(APIView):
    """Get RAG system performance statistics"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get RAG system performance statistics.",
        responses={200: "Performance statistics"},
        tags=['RAG - Stats']
    )
    def get(self, request):
        try:
            context_manager = get_context_manager()
            rag_engine = get_rag_engine()
            
            performance_stats = context_manager.get_performance_stats()
            rag_stats = rag_engine.get_stats()
            llm_status = rag_engine.get_llm_status()
            
            return Response({
                'performance': performance_stats,
                'rag_engine': rag_stats,
                'llm': llm_status,
                'generated_at': timezone.now().isoformat()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return Response({
                'error': 'Failed to get stats',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)