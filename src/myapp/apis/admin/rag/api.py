"""
RAG Management APIs - CORRECTED VERSION
Properly integrates with Content Integration Service and handles both news + social
"""
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from myapp.permissions import IsUserAccess
from myapp.services.rag.rag_service import get_rag_engine
from myapp.services.mongo_manager import get_mongo_manager

logger = logging.getLogger(__name__)


# ============================================================================
# 🔧 INDEX MANAGEMENT APIs (CORRECTED)
# ============================================================================

class IndexDocumentsAPI(APIView):
    """Index documents into vector store (Admin) - NOW WITH SOCIAL POSTS SUPPORT"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="""
        Index content into the vector store for RAG.
        
        IMPORTANT: This indexes BOTH news articles AND social media posts.
        All content is fetched from MongoDB (already processed by Content Integration Service).
        
        Content must be:
        - Already processed (sentiment analysis, entity extraction done)
        - Stored in MongoDB (by Content Integration Service)
        - Above minimum trust score threshold
        """,
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'source': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="Source of documents",
                    enum=['mongodb', 'manual'],
                    default='mongodb'
                ),
                'hours_back': openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="Hours of data to index (for mongodb source)",
                    default=24
                ),
                'limit': openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="Maximum documents to index per type (news + social)",
                    default=1000
                ),
                'min_trust_score': openapi.Schema(
                    type=openapi.TYPE_NUMBER,
                    description="Minimum trust score for indexing",
                    default=5.0
                ),
                'include_news': openapi.Schema(
                    type=openapi.TYPE_BOOLEAN,
                    description="Include news articles",
                    default=True
                ),
                'include_social': openapi.Schema(
                    type=openapi.TYPE_BOOLEAN,
                    description="Include social media posts",
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
                    'indexing_stats': openapi.Schema(
                        type=openapi.TYPE_OBJECT,
                        properties={
                            'news_articles': openapi.Schema(
                                type=openapi.TYPE_OBJECT,
                                properties={
                                    'added': openapi.Schema(type=openapi.TYPE_INTEGER),
                                    'duplicates': openapi.Schema(type=openapi.TYPE_INTEGER),
                                    'errors': openapi.Schema(type=openapi.TYPE_INTEGER)
                                }
                            ),
                            'social_posts': openapi.Schema(
                                type=openapi.TYPE_OBJECT,
                                properties={
                                    'added': openapi.Schema(type=openapi.TYPE_INTEGER),
                                    'duplicates': openapi.Schema(type=openapi.TYPE_INTEGER),
                                    'errors': openapi.Schema(type=openapi.TYPE_INTEGER)
                                }
                            ),
                            'total_indexed': openapi.Schema(type=openapi.TYPE_INTEGER)
                        }
                    ),
                    'vector_store_stats': openapi.Schema(type=openapi.TYPE_OBJECT)
                }
            )
        },
        tags=['RAG - Index Management']
    )
    def post(self, request):
        try:
            data = request.data
            source = data.get('source', 'mongodb')
            hours_back = int(data.get('hours_back', 24))
            limit = int(data.get('limit', 1000))
            min_trust = float(data.get('min_trust_score', 5.0))
            include_news = data.get('include_news', True)
            include_social = data.get('include_social', True)
            platforms = data.get('platforms', None)
            
            logger.info(f"RAG Indexing request: hours_back={hours_back}, limit={limit}, min_trust={min_trust}")
            logger.info(f"Include news: {include_news}, Include social: {include_social}, Platforms: {platforms}")
            
            if source != 'mongodb':
                return Response({
                    'error': 'Only mongodb source is supported'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            rag_engine = get_rag_engine()
            mongo_manager = get_mongo_manager()
            
            # ================================================================
            # INITIALIZE STATS
            # ================================================================
            indexing_stats = {
                'news_articles': {'added': 0, 'duplicates': 0, 'errors': 0},
                'social_posts': {'added': 0, 'duplicates': 0, 'errors': 0},
                'total_indexed': 0
            }
            
            # ================================================================
            # INDEX NEWS ARTICLES
            # ================================================================
            if include_news:
                try:
                    logger.info(f"Fetching news articles (min_trust={min_trust}, limit={limit})...")
                    
                    # ✅ CORRECT: Use trust_score_threshold parameter
                    articles = mongo_manager.get_high_credibility_articles(
                        trust_score_threshold=min_trust,
                        limit=limit,
                        hours_back=hours_back
                    )
                    
                    logger.info(f"Found {len(articles)} news articles to index")
                    
                    # Optional platform filtering
                    if platforms:
                        articles = [a for a in articles if a.get('platform') in platforms]
                        logger.info(f"Filtered to {len(articles)} articles matching platforms: {platforms}")
                    
                    if articles:
                        # Bulk index news articles
                        news_stats = rag_engine.bulk_index_articles(articles)
                        indexing_stats['news_articles'] = news_stats
                        indexing_stats['total_indexed'] += news_stats['added']
                        
                        logger.info(f"News indexing complete: {news_stats}")
                    else:
                        logger.warning("No news articles to index")
                        
                except Exception as e:
                    logger.error(f"Error indexing news articles: {e}")
                    import traceback
                    traceback.print_exc()
                    indexing_stats['news_articles']['errors'] += 1
            
            # ================================================================
            # INDEX SOCIAL MEDIA POSTS
            # ================================================================
            if include_social:
                try:
                    logger.info(f"Fetching social media posts (min_trust={min_trust}, limit={limit})...")
                    
                    # ✅ CORRECT: Use trust_score_threshold parameter
                    social_posts = mongo_manager.get_high_credibility_social_posts(
                        trust_score_threshold=min_trust,
                        limit=limit,
                        hours_back=hours_back
                    )
                    
                    logger.info(f"Found {len(social_posts)} social posts to index")
                    
                    # Optional platform filtering
                    if platforms:
                        social_posts = [p for p in social_posts if p.get('platform') in platforms]
                        logger.info(f"Filtered to {len(social_posts)} posts matching platforms: {platforms}")
                    
                    if social_posts:
                        # Bulk index social posts
                        social_stats = rag_engine.bulk_index_social_posts(social_posts)
                        indexing_stats['social_posts'] = social_stats
                        indexing_stats['total_indexed'] += social_stats['added']
                        
                        logger.info(f"Social indexing complete: {social_stats}")
                    else:
                        logger.warning("No social posts to index")
                        
                except Exception as e:
                    logger.error(f"Error indexing social posts: {e}")
                    import traceback
                    traceback.print_exc()
                    indexing_stats['social_posts']['errors'] += 1
            
            # ================================================================
            # SAVE INDEX
            # ================================================================
            if indexing_stats['total_indexed'] > 0:
                logger.info("Saving RAG index to disk...")
                rag_engine.save_index()
                logger.info("RAG index saved successfully")
            else:
                logger.warning("No documents indexed, skipping save")
            
            # ================================================================
            # RESPONSE
            # ================================================================
            response_data = {
                'status': 'success' if indexing_stats['total_indexed'] > 0 else 'no_content',
                'source': 'mongodb',
                'indexing_stats': indexing_stats,
                'vector_store_stats': rag_engine.get_statistics(),
                'parameters': {
                    'hours_back': hours_back,
                    'limit': limit,
                    'min_trust_score': min_trust,
                    'include_news': include_news,
                    'include_social': include_social,
                    'platforms': platforms or 'all'
                }
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            import traceback
            traceback.print_exc()
            return Response({
                'error': 'Indexing failed',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RAGStatsAPI(APIView):
    """Get RAG system statistics"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="""
        Get comprehensive statistics about the RAG system.
        
        Returns:
        - Vector store stats (document counts by type)
        - Embedding model info
        - LLM provider info
        - Index health metrics
        """,
        responses={
            200: openapi.Response(
                description="RAG system statistics",
                examples={
                    "application/json": {
                        "total_documents": 5432,
                        "document_types": {
                            "news": 3456,
                            "social": 1976
                        },
                        "index_size_mb": 234.5,
                        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                        "embedding_dimension": 768,
                        "llm_provider": "ollama",
                        "llm_model": "llama3.1",
                        "llm_available": True,
                        "last_updated": "2025-12-11T10:30:00Z"
                    }
                }
            )
        },
        tags=['RAG - Index Management']
    )
    def get(self, request):
        try:
            rag_engine = get_rag_engine()
            stats = rag_engine.get_statistics()
            
            return Response(stats, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting RAG stats: {e}")
            return Response({
                'error': 'Failed to get stats',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RebuildIndexAPI(APIView):
    """Rebuild vector index (Admin) - NOW WITH SOCIAL POSTS SUPPORT"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="""
        Rebuild the ENTIRE vector index from MongoDB.
        
        WARNING: This deletes the existing index and rebuilds from scratch.
        Use this when:
        - Switching embedding models
        - Fixing corrupted index
        - Major content updates
        
        Indexes BOTH news articles AND social media posts.
        """,
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'days_back': openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="Days of data to include",
                    default=7
                ),
                'min_trust_score': openapi.Schema(
                    type=openapi.TYPE_NUMBER,
                    description="Minimum trust score",
                    default=4.0
                ),
                'include_news': openapi.Schema(
                    type=openapi.TYPE_BOOLEAN,
                    description="Include news articles",
                    default=True
                ),
                'include_social': openapi.Schema(
                    type=openapi.TYPE_BOOLEAN,
                    description="Include social media posts",
                    default=True
                )
            }
        ),
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'status': openapi.Schema(type=openapi.TYPE_STRING),
                    'rebuild_stats': openapi.Schema(
                        type=openapi.TYPE_OBJECT,
                        properties={
                            'news_articles_indexed': openapi.Schema(type=openapi.TYPE_INTEGER),
                            'social_posts_indexed': openapi.Schema(type=openapi.TYPE_INTEGER),
                            'total_indexed': openapi.Schema(type=openapi.TYPE_INTEGER),
                            'duplicates_skipped': openapi.Schema(type=openapi.TYPE_INTEGER),
                            'errors': openapi.Schema(type=openapi.TYPE_INTEGER)
                        }
                    ),
                    'vector_store_stats': openapi.Schema(type=openapi.TYPE_OBJECT)
                }
            )
        },
        tags=['RAG - Index Management']
    )
    def post(self, request):
        try:
            data = request.data
            days_back = int(data.get('days_back', 7))
            min_trust = float(data.get('min_trust_score', 4.0))
            include_news = data.get('include_news', True)
            include_social = data.get('include_social', True)
            
            logger.info(f"RAG Index Rebuild: days_back={days_back}, min_trust={min_trust}")
            logger.info(f"Include news: {include_news}, Include social: {include_social}")
            
            # ================================================================
            # CREATE NEW RAG ENGINE
            # ================================================================
            from myapp.services.rag.rag_service import LlamaIndexRAGEngine
            from myapp.services.mongo_manager import get_mongo_manager
            
            logger.info("Creating new RAG engine instance...")
            new_rag_engine = LlamaIndexRAGEngine()
            
            mongo_manager = get_mongo_manager()
            
            # ================================================================
            # INITIALIZE STATS
            # ================================================================
            rebuild_stats = {
                'news_articles_indexed': 0,
                'social_posts_indexed': 0,
                'total_indexed': 0,
                'duplicates_skipped': 0,
                'errors': 0
            }
            
            # ================================================================
            # INDEX NEWS ARTICLES
            # ================================================================
            if include_news:
                try:
                    logger.info("Fetching news articles for rebuild...")
                    
                    # ✅ CORRECT: Use trust_score_threshold parameter
                    articles = mongo_manager.get_high_credibility_articles(
                        trust_score_threshold=min_trust,
                        limit=10000,  # Large limit for rebuild
                        hours_back=days_back * 24
                    )
                    
                    logger.info(f"Found {len(articles)} news articles for indexing")
                    
                    if articles:
                        news_stats = new_rag_engine.bulk_index_articles(articles)
                        rebuild_stats['news_articles_indexed'] = news_stats['added']
                        rebuild_stats['duplicates_skipped'] += news_stats['duplicates']
                        rebuild_stats['errors'] += news_stats['errors']
                        rebuild_stats['total_indexed'] += news_stats['added']
                        
                        logger.info(f"News articles indexed: {news_stats['added']}")
                    
                except Exception as e:
                    logger.error(f"Error indexing news articles during rebuild: {e}")
                    import traceback
                    traceback.print_exc()
                    rebuild_stats['errors'] += 1
            
            # ================================================================
            # INDEX SOCIAL POSTS
            # ================================================================
            if include_social:
                try:
                    logger.info("Fetching social posts for rebuild...")
                    
                    # ✅ CORRECT: Use trust_score_threshold parameter
                    social_posts = mongo_manager.get_high_credibility_social_posts(
                        trust_score_threshold=min_trust,
                        limit=10000,  # Large limit for rebuild
                        hours_back=days_back * 24
                    )
                    
                    logger.info(f"Found {len(social_posts)} social posts for indexing")
                    
                    if social_posts:
                        social_stats = new_rag_engine.bulk_index_social_posts(social_posts)
                        rebuild_stats['social_posts_indexed'] = social_stats['added']
                        rebuild_stats['duplicates_skipped'] += social_stats['duplicates']
                        rebuild_stats['errors'] += social_stats['errors']
                        rebuild_stats['total_indexed'] += social_stats['added']
                        
                        logger.info(f"Social posts indexed: {social_stats['added']}")
                    
                except Exception as e:
                    logger.error(f"Error indexing social posts during rebuild: {e}")
                    import traceback
                    traceback.print_exc()
                    rebuild_stats['errors'] += 1
            
            # ================================================================
            # SAVE AND ACTIVATE NEW INDEX
            # ================================================================
            if rebuild_stats['total_indexed'] > 0:
                logger.info("Saving rebuilt index...")
                new_rag_engine.save_index()
                
                # Update global instance
                import myapp.services.rag.rag_service as rag_module
                rag_module._rag_engine_instance = new_rag_engine
                
                logger.info("Index rebuild complete and activated")
            else:
                logger.error("No documents indexed during rebuild!")
                return Response({
                    'error': 'Rebuild failed: no documents indexed',
                    'rebuild_stats': rebuild_stats
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # ================================================================
            # RESPONSE
            # ================================================================
            return Response({
                'status': 'success',
                'rebuild_stats': rebuild_stats,
                'vector_store_stats': new_rag_engine.get_statistics(),
                'parameters': {
                    'days_back': days_back,
                    'min_trust_score': min_trust,
                    'include_news': include_news,
                    'include_social': include_social
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            import traceback
            traceback.print_exc()
            return Response({
                'error': 'Rebuild failed',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)