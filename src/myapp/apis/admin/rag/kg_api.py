"""
Knowledge Graph API Endpoints
Entity relationships, event tracking, and graph queries

UPDATED VERSION - Now processes both news articles AND social posts
"""

import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from myapp.permissions import IsUserAccess
from myapp.services.rag.knowledge_graph import get_knowledge_graph

logger = logging.getLogger(__name__)


class EntityContextAPI(APIView):
    """Get context for a specific entity"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get entity context including relationships and recent events.",
        manual_parameters=[
            openapi.Parameter('entity', openapi.IN_QUERY, 
                description="Entity ID or name (e.g., bitcoin, ethereum, binance)", 
                type=openapi.TYPE_STRING, required=True),
            openapi.Parameter('depth', openapi.IN_QUERY, 
                description="Relationship depth to include", 
                type=openapi.TYPE_INTEGER, default=2),
        ],
        responses={200: "Entity context with relationships"},
        tags=['Knowledge Graph']
    )
    def get(self, request):
        try:
            entity = request.GET.get('entity', '').lower().strip()
            depth = int(request.GET.get('depth', 2))
            
            if not entity:
                return Response({
                    'error': 'Entity parameter is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            kg = get_knowledge_graph()
            
            # Normalize entity ID
            entity_id = kg._normalize_entity_id(entity)
            
            if not entity_id or entity_id not in kg.entities:
                return Response({
                    'error': f'Entity "{entity}" not found',
                    'suggestions': [
                        e.name for e in list(kg.entities.values())[:10]
                    ]
                }, status=status.HTTP_404_NOT_FOUND)
            
            context = kg.get_entity_context(entity_id, depth=depth)
            
            return Response(context, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting entity context: {e}")
            return Response({
                'error': 'Failed to get entity context',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class EntityPathAPI(APIView):
    """Find paths between entities"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Find relationship paths between two entities.",
        manual_parameters=[
            openapi.Parameter('source', openapi.IN_QUERY, 
                description="Source entity", 
                type=openapi.TYPE_STRING, required=True),
            openapi.Parameter('target', openapi.IN_QUERY, 
                description="Target entity", 
                type=openapi.TYPE_STRING, required=True),
            openapi.Parameter('max_depth', openapi.IN_QUERY, 
                description="Maximum path length", 
                type=openapi.TYPE_INTEGER, default=4),
        ],
        responses={200: "Paths between entities"},
        tags=['Knowledge Graph']
    )
    def get(self, request):
        try:
            source = request.GET.get('source', '').lower().strip()
            target = request.GET.get('target', '').lower().strip()
            max_depth = int(request.GET.get('max_depth', 4))
            
            if not source or not target:
                return Response({
                    'error': 'Both source and target parameters are required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            kg = get_knowledge_graph()
            
            source_id = kg._normalize_entity_id(source)
            target_id = kg._normalize_entity_id(target)
            
            if not source_id or source_id not in kg.entities:
                return Response({
                    'error': f'Source entity "{source}" not found'
                }, status=status.HTTP_404_NOT_FOUND)
            
            if not target_id or target_id not in kg.entities:
                return Response({
                    'error': f'Target entity "{target}" not found'
                }, status=status.HTTP_404_NOT_FOUND)
            
            paths = kg.find_path(source_id, target_id, max_depth=max_depth)
            
            # Format paths with entity names
            formatted_paths = []
            for path in paths[:10]:  # Limit to 10 paths
                formatted_path = []
                for entity_id in path:
                    if entity_id in kg.entities:
                        formatted_path.append({
                            'id': entity_id,
                            'name': kg.entities[entity_id].name,
                            'type': kg.entities[entity_id].entity_type
                        })
                formatted_paths.append(formatted_path)
            
            return Response({
                'source': source_id,
                'target': target_id,
                'paths_found': len(paths),
                'paths': formatted_paths
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error finding paths: {e}")
            return Response({
                'error': 'Failed to find paths',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class TrendingEntitiesAPI(APIView):
    """Get trending entities"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get entities with most recent activity.",
        manual_parameters=[
            openapi.Parameter('hours', openapi.IN_QUERY, 
                description="Time window in hours", 
                type=openapi.TYPE_INTEGER, default=24),
            openapi.Parameter('limit', openapi.IN_QUERY, 
                description="Number of entities to return", 
                type=openapi.TYPE_INTEGER, default=10),
        ],
        responses={200: "Trending entities"},
        tags=['Knowledge Graph']
    )
    def get(self, request):
        try:
            hours = int(request.GET.get('hours', 24))
            limit = int(request.GET.get('limit', 10))
            
            kg = get_knowledge_graph()
            
            trending = kg.get_trending_entities(hours_back=hours, limit=limit)
            
            return Response({
                'trending_entities': trending,
                'time_window_hours': hours
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting trending entities: {e}")
            return Response({
                'error': 'Failed to get trending entities',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class KnowledgeGraphStatsAPI(APIView):
    """Get knowledge graph statistics"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get knowledge graph statistics.",
        responses={200: "Graph statistics"},
        tags=['Knowledge Graph']
    )
    def get(self, request):
        try:
            kg = get_knowledge_graph()
            stats = kg.get_graph_statistics()
            
            return Response(stats, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return Response({
                'error': 'Failed to get statistics',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class BuildGraphFromContentAPI(APIView):
    """Build knowledge graph from Content (Admin) - UPDATED to handle both news and social"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="""
        Extract entities and relationships from content to build knowledge graph.
        
        Processes BOTH news articles AND social media posts to build a comprehensive
        knowledge graph of cryptocurrency entities, relationships, and events.
        
        News sources include: CryptoPanic, CryptoCompare, NewsAPI, etc.
        Social sources include: Reddit, Twitter, YouTube
        """,
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'hours_back': openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="Hours of content to process",
                    default=24
                ),
                'limit': openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="Maximum content items to process (per type)",
                    default=500
                ),
                'min_trust_score': openapi.Schema(
                    type=openapi.TYPE_NUMBER,
                    description="Minimum trust score for content inclusion",
                    default=5.0
                ),
                'include_news': openapi.Schema(
                    type=openapi.TYPE_BOOLEAN,
                    description="Include news articles in graph building",
                    default=True
                ),
                'include_social': openapi.Schema(
                    type=openapi.TYPE_BOOLEAN,
                    description="Include social media posts in graph building",
                    default=True
                ),
                'platforms': openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Schema(type=openapi.TYPE_STRING),
                    description="Specific platforms to include (optional)",
                    example=['cryptopanic', 'reddit', 'twitter']
                )
            }
        ),
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'status': openapi.Schema(type=openapi.TYPE_STRING),
                    'stats': openapi.Schema(
                        type=openapi.TYPE_OBJECT,
                        properties={
                            'articles_processed': openapi.Schema(type=openapi.TYPE_INTEGER),
                            'social_posts_processed': openapi.Schema(type=openapi.TYPE_INTEGER),
                            'total_content_processed': openapi.Schema(type=openapi.TYPE_INTEGER),
                            'entities_found': openapi.Schema(type=openapi.TYPE_INTEGER),
                            'relationships_created': openapi.Schema(type=openapi.TYPE_INTEGER),
                            'events_created': openapi.Schema(type=openapi.TYPE_INTEGER),
                        }
                    ),
                    'graph_stats': openapi.Schema(type=openapi.TYPE_OBJECT),
                    'content_breakdown': openapi.Schema(type=openapi.TYPE_OBJECT)
                }
            )
        },
        tags=['Knowledge Graph']
    )
    def post(self, request):
        try:
            data = request.data
            hours_back = int(data.get('hours_back', 24))
            limit = int(data.get('limit', 500))
            min_trust = float(data.get('min_trust_score', 5.0))
            include_news = data.get('include_news', True)
            include_social = data.get('include_social', True)
            platforms = data.get('platforms', None)  # Optional platform filter
            
            logger.info(f"Building knowledge graph: hours_back={hours_back}, limit={limit}, min_trust={min_trust}")
            logger.info(f"Include news: {include_news}, Include social: {include_social}, Platforms: {platforms}")
            
            from myapp.services.mongo_manager import get_mongo_manager
            
            kg = get_knowledge_graph()
            mongo = get_mongo_manager()
            
            # Initialize statistics
            stats = {
                'articles_processed': 0,
                'social_posts_processed': 0,
                'total_content_processed': 0,
                'entities_found': 0,
                'relationships_created': 0,
                'events_created': 0,
                'errors': 0
            }
            
            content_breakdown = {
                'by_platform': {},
                'by_type': {'news': 0, 'social': 0},
                'high_impact_events': 0
            }
            
            # ===================================================================
            # PROCESS NEWS ARTICLES
            # ===================================================================
            if include_news:
                try:
                    logger.info(f"Fetching news articles (min_trust={min_trust}, limit={limit})...")
                    
                    # ✅ CORRECTED: Use trust_score_threshold instead of min_trust_score
                    articles = mongo.get_high_credibility_articles(
                        trust_score_threshold=min_trust,  # ← FIXED
                        limit=limit,
                        hours_back=hours_back
                    )
                    
                    logger.info(f"Found {len(articles)} news articles to process")
                    
                    # Optional platform filtering
                    if platforms:
                        articles = [a for a in articles if a.get('platform') in platforms]
                        logger.info(f"Filtered to {len(articles)} articles matching platforms: {platforms}")
                    
                    for i, article in enumerate(articles):
                        try:
                            platform = article.get('platform', 'unknown')
                            
                            # Extract and link from article
                            result = kg.extract_and_link_from_content(article)
                            
                            # Update stats
                            stats['articles_processed'] += 1
                            stats['total_content_processed'] += 1
                            stats['entities_found'] += len(result['entities_found'])
                            stats['relationships_created'] += len(result['relationships_created'])
                            
                            if result['event_created']:
                                stats['events_created'] += 1
                                
                                # Track high-impact events
                                event = kg.events.get(result['event_created'])
                                if event and event.impact_score >= 7.0:
                                    content_breakdown['high_impact_events'] += 1
                            
                            # Track by platform
                            if platform not in content_breakdown['by_platform']:
                                content_breakdown['by_platform'][platform] = 0
                            content_breakdown['by_platform'][platform] += 1
                            content_breakdown['by_type']['news'] += 1
                            
                            # Log progress every 50 articles
                            if (i + 1) % 50 == 0:
                                logger.info(f"Processed {i + 1}/{len(articles)} news articles")
                            
                        except Exception as e:
                            logger.error(f"Error processing article {i+1}: {e}")
                            stats['errors'] += 1
                    
                    logger.info(f"Completed news processing: {stats['articles_processed']} articles")
                    
                except Exception as e:
                    logger.error(f"Error fetching/processing news articles: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ===================================================================
            # PROCESS SOCIAL MEDIA POSTS
            # ===================================================================
            if include_social:
                try:
                    logger.info(f"Fetching social media posts (min_trust={min_trust}, limit={limit})...")
                    
                    # ✅ CORRECTED: Use trust_score_threshold instead of min_trust_score
                    social_posts = mongo.get_high_credibility_social_posts(
                        trust_score_threshold=min_trust,  # ← FIXED
                        limit=limit,
                        hours_back=hours_back
                    )
                    
                    logger.info(f"Found {len(social_posts)} social posts to process")
                    
                    # Optional platform filtering
                    if platforms:
                        social_posts = [p for p in social_posts if p.get('platform') in platforms]
                        logger.info(f"Filtered to {len(social_posts)} posts matching platforms: {platforms}")
                    
                    for i, post in enumerate(social_posts):
                        try:
                            platform = post.get('platform', 'unknown')
                            
                            # Extract and link from social post
                            result = kg.extract_and_link_from_content(post)
                            
                            # Update stats
                            stats['social_posts_processed'] += 1
                            stats['total_content_processed'] += 1
                            stats['entities_found'] += len(result['entities_found'])
                            stats['relationships_created'] += len(result['relationships_created'])
                            
                            if result['event_created']:
                                stats['events_created'] += 1
                                
                                # Track high-impact events
                                event = kg.events.get(result['event_created'])
                                if event and event.impact_score >= 7.0:
                                    content_breakdown['high_impact_events'] += 1
                            
                            # Track by platform
                            if platform not in content_breakdown['by_platform']:
                                content_breakdown['by_platform'][platform] = 0
                            content_breakdown['by_platform'][platform] += 1
                            content_breakdown['by_type']['social'] += 1
                            
                            # Log progress every 50 posts
                            if (i + 1) % 50 == 0:
                                logger.info(f"Processed {i + 1}/{len(social_posts)} social posts")
                            
                        except Exception as e:
                            logger.error(f"Error processing social post {i+1}: {e}")
                            stats['errors'] += 1
                    
                    logger.info(f"Completed social processing: {stats['social_posts_processed']} posts")
                    
                except Exception as e:
                    logger.error(f"Error fetching/processing social posts: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ===================================================================
            # SAVE KNOWLEDGE GRAPH
            # ===================================================================
            logger.info("Saving knowledge graph to disk...")
            kg.save_graph()
            logger.info("Knowledge graph saved successfully")
            
            # Get final graph statistics
            graph_stats = kg.get_graph_statistics()
            
            # Build response
            response_data = {
                'status': 'success',
                'stats': stats,
                'graph_stats': graph_stats,
                'content_breakdown': content_breakdown,
                'parameters': {
                    'hours_back': hours_back,
                    'limit': limit,
                    'min_trust_score': min_trust,
                    'include_news': include_news,
                    'include_social': include_social,
                    'platforms': platforms or 'all'
                }
            }
            
            logger.info(f"Knowledge graph build complete: {stats}")
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            import traceback
            traceback.print_exc()
            return Response({
                'error': 'Failed to build graph',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)