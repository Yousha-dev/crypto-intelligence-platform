import logging
import pprint
from bson import ObjectId
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.utils import timezone
from dateutil.parser import parse as dt_parse
from datetime import datetime, timedelta

from myapp.permissions import IsUserAccess
from myapp.services.content.integrator_service import get_integrator_service, ContentIntegrationService
from myapp.services.content.credibility_engine import get_credibility_engine, get_threshold_manager
from myapp.services.mongo_manager import get_mongo_manager
from myapp.services.content.hashtag_analyzer import get_hashtag_analyzer
from myapp.services.content.topic_modeler import get_topic_modeler
from myapp.services.content.sentiment_analyzer import get_sentiment_analyzer

logger = logging.getLogger(__name__)
 

# ============================================================================
# NEWS FEED APIs
# ============================================================================

class GetCuratedFeedAPI(APIView):
    """Get curated high-credibility news feed"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get curated news feed filtered by credibility score.",
        manual_parameters=[
            openapi.Parameter('threshold', openapi.IN_QUERY, 
                description="Minimum credibility score (0-10)", 
                type=openapi.TYPE_NUMBER, default=6.0),
            openapi.Parameter('limit', openapi.IN_QUERY, 
                description="Number of articles to return", 
                type=openapi.TYPE_INTEGER, default=50),
            openapi.Parameter('hours', openapi.IN_QUERY, 
                description="Hours of historical data", 
                type=openapi.TYPE_INTEGER, default=24),
            openapi.Parameter('category', openapi.IN_QUERY, 
                description="Filter by category (bitcoin, ethereum, defi, regulation, etc.)", 
                type=openapi.TYPE_STRING),
            openapi.Parameter('sentiment', openapi.IN_QUERY, 
                description="Filter by sentiment (bullish, bearish, neutral)", 
                type=openapi.TYPE_STRING),
        ],
        responses={
            200: openapi.Response(
                description="Curated news feed",
                examples={
                    "application/json": {
                        "articles": [
                            {
                                "id": "article_123",
                                "title": "Bitcoin Price Analysis",
                                "description": "...",
                                "url": "https://...",
                                "source": "CoinDesk",
                                "published_at": "2024-12-05T10:00:00Z",
                                "trust_score": 8.5,
                                "sentiment": "bullish",
                                "categories": ["bitcoin", "analysis"]
                            }
                        ],
                        "total": 50,
                        "filters_applied": {"threshold": 6.0, "hours": 24}
                    }
                }
            ),
            500: "Server error"
        },
        tags=['News Feed']
    )
    def get(self, request):
        try:
            # Parse parameters
            threshold = float(request.GET.get('threshold', 6.0))
            limit = int(request.GET.get('limit', 50))
            hours = int(request.GET.get('hours', 24))
            category = request.GET.get('category')
            sentiment_filter = request.GET.get('sentiment')
            
            service = get_integrator_service()
            
            # Get curated feed
            feed = service.get_curated_feed(
                trust_score_threshold=threshold,
                hours_back=hours
            )
            
            articles = feed.get('feed', [])  # Changed from 'articles' to 'feed'
            
            # Apply additional filters
            if category:
                articles = [
                    a for a in articles 
                    # Updated: Check extracted_entities for cryptocurrencies
                    if category.lower() in [
                        c.lower() for c in 
                        a.get('content', {}).get('extracted_entities', {}).get('cryptocurrencies', [])
                    ]
                ]
            
            if sentiment_filter:
                articles = [
                    a for a in articles 
                    if (
                        a.get('content', {}).get('sentiment_analysis', {}).get('label', '').lower() == sentiment_filter.lower() or
                        a.get('content', {}).get('sentiment_analysis', {}).get('sentiment_label', '').lower() == sentiment_filter.lower()
                    )
                ]
            
            # Format response
            formatted_articles = []
            for idx, article in enumerate(articles):
                try:
                    content = article.get('content', {})
                    source_val = content.get('source', {})
                    if isinstance(source_val, dict):
                        source_name = source_val.get('title', source_val.get('name', 'Unknown'))
                    elif isinstance(source_val, str):
                        source_name = source_val
                    else:
                        source_name = 'Unknown'

                    sentiment_data = content.get('sentiment_analysis', {})
                    sentiment_label = sentiment_data.get('label', 'neutral')

                    formatted_articles.append({
                        'id': str(content.get('_id', content.get('id', content.get('source_id')))),
                        'title': content.get('title'),
                        'description': content.get('description', '')[:200],
                        'url': content.get('url'),
                        'source': source_name,
                        'platform': content.get('platform'),
                        'published_at': content.get('published_at'),
                        'trust_score': round(content.get('trust_score', 0), 2),
                        'status': content.get('status', 'pending'),
                        'sentiment': sentiment_label,
                        'flags': content.get('credibility_analysis', {}).get('flags', [])[:3]
                    })
                except Exception as fe:
                    logger.error(f"Error formatting article at index {idx}: {fe}")
                    logger.error(f"Article raw data: {article}")
                    continue
            
            return Response({
                'articles': formatted_articles,
                'total': len(formatted_articles),
                'filters_applied': {
                    'threshold': threshold,
                    'hours': hours,
                    'category': category,
                    'sentiment': sentiment_filter
                },
                'timestamp': timezone.now().isoformat()
            }, status=status.HTTP_200_OK)
            
        except Exception as e: 
            logger.error(f"Error fetching curated feed: {e}")
            return Response({
                'error': 'Failed to fetch news feed',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GetAllArticlesAPI(APIView):
    """Get all articles with pagination and filtering"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get all articles with pagination and advanced filtering.",
        manual_parameters=[
            openapi.Parameter('page', openapi.IN_QUERY, type=openapi.TYPE_INTEGER, default=1),
            openapi.Parameter('page_size', openapi.IN_QUERY, type=openapi.TYPE_INTEGER, default=20),
            openapi.Parameter('sort_by', openapi.IN_QUERY, type=openapi.TYPE_STRING, 
                enum=['published_at', 'trust_score', 'sentiment_score'], default='published_at'),
            openapi.Parameter('sort_order', openapi.IN_QUERY, type=openapi.TYPE_STRING, 
                enum=['asc', 'desc'], default='desc'),
            openapi.Parameter('min_score', openapi.IN_QUERY, type=openapi.TYPE_NUMBER),
            openapi.Parameter('max_score', openapi.IN_QUERY, type=openapi.TYPE_NUMBER),
            openapi.Parameter('source', openapi.IN_QUERY, type=openapi.TYPE_STRING),
            openapi.Parameter('platform', openapi.IN_QUERY, type=openapi.TYPE_STRING),
            openapi.Parameter('status', openapi.IN_QUERY, type=openapi.TYPE_STRING,
                enum=['approved', 'pending', 'flagged', 'rejected']),
        ],
        responses={200: "Paginated articles list"},
        tags=['News Feed']
    )
    def get(self, request):
        try:
            # Parse parameters
            page = int(request.GET.get('page', 1))
            page_size = int(request.GET.get('page_size', 20))
            sort_by = request.GET.get('sort_by', 'published_at')
            sort_order = request.GET.get('sort_order', 'desc')
            min_score = request.GET.get('min_score')
            max_score = request.GET.get('max_score')
            source = request.GET.get('source')
            platform = request.GET.get('platform')
            article_status = request.GET.get('status')
            
            mongo_manager = get_mongo_manager()
            
            # Build query
            query = {'isdeleted': {'$ne': True}}
            
            if min_score:
                query['trust_score'] = {'$gte': float(min_score)}
            if max_score:
                if 'trust_score' in query:
                    query['trust_score']['$lte'] = float(max_score)
                else:
                    query['trust_score'] = {'$lte': float(max_score)}
            if source:
                query['source.title'] = {'$regex': source, '$options': 'i'}
            if platform:
                query['platform'] = platform
            if article_status:
                query['status'] = article_status
            
            # Sort configuration
            sort_direction = -1 if sort_order == 'desc' else 1
            sort_field = sort_by
            if sort_by == 'published_at':
                sort_field = 'published_at'
            elif sort_by == 'trust_score':
                sort_field = 'trust_score'
            
            # Execute query
            skip = (page - 1) * page_size
            
            collection = mongo_manager.collections['news_articles']
            total = collection.count_documents(query)
            articles = list(
                collection.find(query)
                .sort(sort_field, sort_direction)
                .skip(skip)
                .limit(page_size)
            )
            
            # Format articles
            formatted_articles = []
            for idx, article in enumerate(articles):
                try:
                    source_obj = article.get("source", {})
                    source_name = source_obj.get("name") or source_obj.get("title") or article.get("platform") or "Unknown"

                    formatted_articles.append({
                        "id": str(article.get("_id")),
                        "title": article.get("title"),
                        "description": article.get("description", "")[:200],
                        "url": article.get("url"),
                        "source": source_name,
                        "platform": article.get("platform"),
                        "published_at": article.get("published_at"),
                        "trust_score": round(article.get("trust_score", 0), 2),
                        "status": article.get("status", "pending"),
                        "sentiment": article.get("sentiment_analysis", {}).get("label"),
                        "flags": article.get("credibility_analysis", {}).get("flags", [])[:3],
                        "entities": article.get("extracted_entities", {}),
                    })

                except Exception as fe:
                    logger.error(f"Error formatting article at index {idx}: {fe}")
                    logger.error(f"Article raw data: {article}")
                    continue
            
            return Response({
                'articles': formatted_articles,
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total': total,
                    'total_pages': (total + page_size - 1) // page_size,
                    'has_next': page * page_size < total,
                    'has_prev': page > 1
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            import traceback
            logger.error(f"Error fetching articles: {e}")
            logger.error(traceback.format_exc())
            return Response({
                'error': 'Failed to fetch articles',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GetArticleDetailAPI(APIView):
    """Get detailed information about a single article"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get detailed article information including full credibility analysis.",
        responses={
            200: openapi.Response(
                description="Article details",
                examples={
                    "application/json": {
                        "article": {
                            "id": "article_123",
                            "title": "Bitcoin Price Analysis",
                            "content": "Full article content...",
                            "credibility_analysis": {
                                "trust_score": 8.5,
                                "breakdown": {
                                    "source": 9.0,
                                    "sentiment": 7.0,
                                    "cross_check": 8.0,
                                    "history": 8.5,
                                    "recency": 9.0
                                }
                            }
                        }
                    }
                }
            ),
            404: "Article not found"
        },
        tags=['News Feed']
    )
    def get(self, request, article_id):
        try:
            mongo_manager = get_mongo_manager()
            
            query = {
                '$or': [
                    {'id': article_id},
                    {'source_id': article_id}
                ]
            }
            # Try to add ObjectId match if possible
            try:
                query['$or'].append({'_id': ObjectId(article_id)})
            except Exception:
                query['$or'].append({'_id': article_id})

            article = mongo_manager.collections['news_articles'].find_one(query)

            if not article:
                return Response({
                    'error': 'Article not found'
                }, status=status.HTTP_404_NOT_FOUND)
            
            source_obj = article.get("source", {})
            source_name = source_obj.get("name") or source_obj.get("title") or article.get("platform") or "Unknown"

            cred = article.get("credibility_analysis", {})
            
            content = article.get("content")
            if not content:
                content = article.get("description")


            response_data = {
                "article": {
                    "id": str(article.get("_id")),
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "content": content,
                    "url": article.get("url"),
                    "author": article.get("author"),

                    # UPDATED source
                    "source": {
                        "name": source_name,
                        "domain": source_obj.get("domain", "")
                    },

                    "platform": article.get("platform"),
                    "published_at": article.get("published_at"),
                    "fetched_at": article.get("analysis_timestamp"),
                    "status": article.get("status", "pending"),

                    # UPDATED credibility structure
                    "credibility_analysis": {
                        "trust_score": round(article.get("trust_score", 0), 2),
                        "confidence": cred.get("confidence", 0),
                        "breakdown": {
                            "source_score": cred.get("source_score", 0),
                            "content_score": cred.get("content_score", 0),
                            "engagement_score": cred.get("engagement_score", 0),
                            "cross_check_score": cred.get("cross_check_score", 0),
                            "source_history_score": cred.get("source_history_score", 0),
                            "recency_score": cred.get("recency_score", 0),
                        },
                        "flags": cred.get("flags", []),
                        "reasoning": cred.get("reasoning", ""),
                        "cross_reference_matches": cred.get("cross_reference_matches", 0),
                        "corroboration_sources": cred.get("corroboration_sources", []),

                        # UPDATED: recommended_action from action_info
                        "recommended_action": article.get("action_info", {}),
                    },
                    
                    "verification_details": article.get("verification_details", {}),

                    # UPDATED sentiment
                    "sentiment_analysis": article.get("sentiment_analysis", {}),

                    # UPDATED crypto relevance
                    "crypto_analysis": article.get("extracted_entities", {}),
                    
                    "source_credibility": article.get("source_credibility", {}),
                    "entities": article.get("extracted_entities", {}),
                    "text_processing": article.get("text_processing", {}),

                    "raw_metrics": {
                        "votes": article.get("votes", {}),
                        "upvotes": article.get("upvotes", 0),
                        "downvotes": article.get("downvotes", 0),
                    }
                }
            }

            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching article {article_id}: {e}")
            return Response({
                'error': 'Failed to fetch article',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SearchArticlesAPI(APIView):
    """Search articles by keywords, topics, or entities"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Search news articles by natural text, metadata fields, or crypto filters.",
        manual_parameters=[
            openapi.Parameter('q', openapi.IN_QUERY,
                              description="Search query",
                              type=openapi.TYPE_STRING,
                              required=True),
            openapi.Parameter('crypto', openapi.IN_QUERY,
                              description="Filter by cryptocurrency",
                              type=openapi.TYPE_STRING),
            openapi.Parameter('limit', openapi.IN_QUERY,
                              description="Max results",
                              type=openapi.TYPE_INTEGER,
                              default=20),
        ],
        responses={200: "Search results"},
        tags=['News Feed']
    )
    def get(self, request):
        try:
            query = request.GET.get('q', '').strip()
            crypto = request.GET.get('crypto')
            limit = int(request.GET.get('limit', 20))

            if not query:
                return Response({"error": "Search query is required"},
                                status=status.HTTP_400_BAD_REQUEST)

            mongo = get_mongo_manager()
            col = mongo.collections['news_articles']

            # -----------------------------
            # 1) PRIMARY SEARCH (MAIN TEXT)
            # -----------------------------
            primary_query = {
                "$and": [
                    {
                        "$or": [
                            {"title": {"$regex": query, "$options": "i"}},
                            {"description": {"$regex": query, "$options": "i"}},
                            {"content": {"$regex": query, "$options": "i"}},
                        ]
                    },
                    {"isdeleted": {"$ne": True}}
                ]
            }

            # Add crypto filter
            if crypto:
                crypto_filter = {
                    "$or": [
                        {"extracted_entities.cryptocurrencies": {"$regex": crypto, "$options": "i"}},
                        {"crypto_relevance.mentioned_cryptocurrencies": {"$regex": crypto, "$options": "i"}},
                        {"crypto_analysis.mentioned_cryptocurrencies": {"$regex": crypto, "$options": "i"}}
                    ]
                }
                primary_query["$and"].append(crypto_filter)

            articles = list(
                col.find(primary_query)
                  .sort("trust_score", -1)
                  .limit(limit)
            )

            # ---------------------------------
            # 2) FALLBACK SEARCH (METADATA)
            # ---------------------------------
            if len(articles) == 0:
                fallback_query = {
                    "$and": [
                        {
                            "$or": [
                                {"extracted_entities.cryptocurrencies": {"$regex": query, "$options": "i"}},
                                {"extracted_entities.keywords": {"$regex": query, "$options": "i"}},
                                {"extracted_entities.people": {"$regex": query, "$options": "i"}},
                                {"extracted_entities.companies": {"$regex": query, "$options": "i"}},
                                {"crypto_relevance.keywords": {"$regex": query, "$options": "i"}},
                                {"sentiment_analysis.keywords": {"$regex": query, "$options": "i"}},
                                {"text_processing.keywords": {"$regex": query, "$options": "i"}},
                                {"hashtags": {"$regex": query, "$options": "i"}},
                            ]
                        },
                        {"isdeleted": {"$ne": True}}
                    ]
                }

                if crypto:
                    fallback_query["$and"].append(crypto_filter)

                articles = list(
                    col.find(fallback_query)
                      .sort("trust_score", -1)
                      .limit(limit)
                )

            # ---------------------------------
            # FORMAT RESULTS
            # ---------------------------------
            results = []
            for a in articles:
                source_val = a.get("source")

                if isinstance(source_val, dict):
                    source_name = source_val.get("name") or source_val.get("title") or "Unknown"
                elif isinstance(source_val, str):
                    source_name = source_val
                else:
                    source_name = "Unknown"

                results.append({
                    "id": str(a.get("_id")),
                    "title": a.get("title"),
                    "description": (a.get("description") or "")[:200],
                    "url": a.get("url"),
                    "source": source_name,
                    "published_at": a.get("published_at"),
                    "trust_score": round(a.get("trust_score", 0), 2),
                    "cryptocurrencies": a.get("extracted_entities", {}).get("cryptocurrencies", []),
                })

            return Response({
                "query": query,
                "results": results,
                "total": len(results)
            })

        except Exception as e:
            logger.error(f"Error searching articles: {e}")
            return Response({
                "error": "Search failed",
                "detail": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GetSocialFeedAPI(APIView):
    """Get social media posts feed"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get social media posts filtered by platform and credibility.",
        manual_parameters=[
            openapi.Parameter('platform', openapi.IN_QUERY, 
                description="Filter by platform", 
                type=openapi.TYPE_STRING,
                enum=['all', 'reddit', 'twitter', 'youtube'], default='all'),
            openapi.Parameter('threshold', openapi.IN_QUERY, 
                description="Minimum trust score (0-10)", 
                type=openapi.TYPE_NUMBER, default=5.0),
            openapi.Parameter('limit', openapi.IN_QUERY, 
                description="Number of posts to return", 
                type=openapi.TYPE_INTEGER, default=50),
            openapi.Parameter('hours', openapi.IN_QUERY, 
                description="Hours of historical data", 
                type=openapi.TYPE_INTEGER, default=24),
            openapi.Parameter('sentiment', openapi.IN_QUERY, 
                description="Filter by sentiment (bullish, bearish, neutral)", 
                type=openapi.TYPE_STRING),
        ],
        responses={200: "Social posts feed"},
        tags=['Social Feed']
    )
    def get(self, request): 
        try:
            platform = request.GET.get('platform', 'all')
            threshold = float(request.GET.get('threshold', 5.0))
            limit = int(request.GET.get('limit', 50))
            hours = int(request.GET.get('hours', 24))
            sentiment_filter = request.GET.get('sentiment')
            
            mongo_manager = get_mongo_manager()
            
            # Get social posts
            posts = mongo_manager.get_social_posts(
                platform=platform if platform != 'all' else None,
                hours_back=hours
            )
            
            # Filter by trust score
            filtered_posts = [p for p in posts if p.get('trust_score', 0) >= threshold]
            
            # Filter by sentiment if specified
            if sentiment_filter:
                filtered_posts = [
                    p for p in filtered_posts
                    if p.get('sentiment_analysis', {}).get('label', '').lower() == sentiment_filter.lower()
                ]
            
            # Format response
            formatted_posts = []
            for post in filtered_posts[:limit]:
                engagement = post.get('engagement_metrics', {})
                user_cred = post.get('user_credibility', {})
                
                formatted_post = {
                    'id': str(post.get('_id', post.get('source_id'))),
                    'platform': post.get('platform', 'unknown'),
                    'title': post.get('title', ''),
                    'content': post.get('content', '')[:300],
                    'url': post.get('url'),
                    'author': post.get('author_username', 'Unknown'),
                    'published_at': post.get('published_at'),
                    'trust_score': round(post.get('trust_score', 0), 2),
                    'sentiment': post.get('sentiment_analysis', {}).get('label', 'neutral'),
                    'status': post.get('status', 'pending'),
                }

                
                # Add platform-specific metrics
                if post.get('platform') == 'reddit':
                    formatted_post['metrics'] = {
                        'score': engagement.get('score', 0),
                        'upvote_ratio': engagement.get('upvote_ratio', 0),
                        'comments': engagement.get('num_comments', 0),
                        'awards': engagement.get('total_awards_received', 0),

                        # Corrected user credibility fields
                        'author_karma': user_cred.get('total_karma', 0),
                        'link_karma': user_cred.get('link_karma', 0),
                        'comment_karma': user_cred.get('comment_karma', 0),
                        'followers': user_cred.get('followers', 0),
                        'is_verified': user_cred.get('verified', False),
                        'is_mod': user_cred.get('is_mod', False),
                        'is_gold': user_cred.get('is_gold', False),

                        'subreddit': post.get('subreddit', ''),
                        'subreddit_subscribers': user_cred.get('subreddit_subscribers', 0)
                    }

                elif post.get('platform') == 'twitter':
                    formatted_post['metrics'] = {
                        'likes': int(engagement.get('like_count', 0) or 0),
                        'retweets': engagement.get('retweet_count', 0),
                        'replies': engagement.get('reply_count', 0),
                        'quotes': engagement.get('quote_count', 0),

                        # Corrected user credibility fields
                        'followers': user_cred.get('followers_count', 0),
                        'following': user_cred.get('following_count', 0),
                        'tweets': user_cred.get('tweet_count', 0),
                        'listed_count': user_cred.get('listed_count', 0),
                        'verified': user_cred.get('verified', False),
                        'verified_type': user_cred.get('verified_type')
                    }

                elif post.get('platform') == 'youtube':
                    channel_info = post.get('channel_info', {})
                    formatted_post['metrics'] = {
                        'views': engagement.get('view_count', 0),
                        'likes': engagement.get('like_count', 0),
                        'comments': engagement.get('comment_count', 0),
                        'channel_subscribers': channel_info.get('subscriber_count', 0),
                        'duration_seconds': engagement.get('duration_seconds', 0)
                    }
                
                formatted_posts.append(formatted_post)
            
            return Response({
                'posts': formatted_posts,
                'total': len(formatted_posts),
                'filters_applied': {
                    'platform': platform,
                    'threshold': threshold,
                    'hours': hours,
                    'sentiment': sentiment_filter
                },
                'timestamp': timezone.now().isoformat()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching social feed: {e}")
            return Response({
                'error': 'Failed to fetch social feed',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GetSocialPostDetailAPI(APIView):
    """Get detailed information about a social post"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get detailed social post information including full credibility analysis.",
        responses={200: "Post details", 404: "Post not found"},
        tags=['Social Feed']
    )
    def get(self, request, post_id):
        try:
            mongo_manager = get_mongo_manager()
            
            # Try to match by _id (ObjectId), source_id, or id
            query = {
                '$or': [
                    {'id': post_id},
                    {'source_id': post_id}
                ]
            }
            try:
                query['$or'].append({'_id': ObjectId(post_id)})
            except Exception:
                query['$or'].append({'_id': post_id})

            post = mongo_manager.collections['social_posts'].find_one(query)
            
            if not post:
                return Response({
                    'error': 'Post not found'
                }, status=status.HTTP_404_NOT_FOUND)
            
            platform = post.get('platform', 'unknown')
            
            response_data = {
                'post': {
                    'id': str(post.get('_id', post.get('source_id'))),
                    'platform': platform,
                    'title': post.get('title'),
                    'content': post.get('content'),
                    'url': post.get('url'),
                    'author': post.get('author_username'),
                    'published_at': post.get('published_at'),
                    'status': post.get('status', 'pending'),
                    
                    # Trust score and breakdown
                    'trust_score': round(post.get('trust_score', 0), 2),
                    'credibility_analysis': post.get('credibility_analysis', {}),
                    
                    # Engagement metrics (platform-specific)
                    'engagement_metrics': post.get('engagement_metrics', {}),
                    
                    # User/Channel credibility
                    'user_credibility': post.get('user_credibility', {}),
                    
                    # Sentiment analysis
                    'sentiment_analysis': post.get('sentiment_analysis', {}),
                    
                    # Crypto relevance
                    'crypto_analysis': post.get('crypto_relevance', post.get('crypto_analysis', {})),
                    
                    # Extracted entities
                    'extracted_entities': post.get('extracted_entities', {}),
                    
                    # Text processing info
                    'text_processing': post.get('text_processing', {})
                }
            }
            
            # Add platform-specific info
            if platform == 'reddit':
                response_data['post']['subreddit_info'] = post.get('subreddit_info', {})
                response_data['post']['author_info'] = post.get('author_info', {})
            elif platform == 'twitter':
                response_data['post']['user_profile'] = post.get('user_credibility', {})
            elif platform == 'youtube':
                response_data['post']['channel_info'] = post.get('channel_info', {})
                response_data['post']['has_transcript'] = bool(post.get('transcript') or post.get('caption'))
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching post {post_id}: {e}")
            return Response({
                'error': 'Failed to fetch post',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SearchSocialPostAPI(APIView):
    """Search social posts by keywords, entities, or crypto."""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Search social media posts (Twitter, Reddit, etc.)",
        manual_parameters=[
            openapi.Parameter('q', openapi.IN_QUERY, type=openapi.TYPE_STRING, required=True),
            openapi.Parameter('crypto', openapi.IN_QUERY, type=openapi.TYPE_STRING),
            openapi.Parameter('limit', openapi.IN_QUERY, type=openapi.TYPE_INTEGER, default=20),
        ],
        responses={200: "Search results"},
        tags=['Social Feed']
    )
    def get(self, request):
        try:
            query = request.GET.get('q', '').strip()
            crypto = request.GET.get('crypto')
            limit = int(request.GET.get('limit', 20))

            if not query:
                return Response({"error": "Search query is required"},
                                status=status.HTTP_400_BAD_REQUEST)

            mongo = get_mongo_manager()
            col = mongo.collections["social_posts"]

            # -----------------------------
            # PRIMARY SEARCH (text content)
            # -----------------------------
            primary_query = {
                "$and": [
                    {
                        "$or": [
                            {"text": {"$regex": query, "$options": "i"}},
                            {"title": {"$regex": query, "$options": "i"}},
                        ]
                    },
                    {"isdeleted": {"$ne": True}}
                ]
            }

            if crypto:
                crypto_filter = {
                    "$or": [
                        {"extracted_entities.cryptocurrencies": {"$regex": crypto, "$options": "i"}},
                        {"crypto_entities": {"$regex": crypto, "$options": "i"}},
                        {"hashtags": {"$regex": crypto, "$options": "i"}},
                    ]
                }
                primary_query["$and"].append(crypto_filter)

            posts = list(
                col.find(primary_query)
                  .sort("engagement_score", -1)
                  .limit(limit)
            )

            # -----------------------------
            # FALLBACK SEARCH (metadata)
            # -----------------------------
            if len(posts) == 0:
                fallback_query = {
                    "$and": [
                        {
                            "$or": [
                                {"hashtags": {"$regex": query, "$options": "i"}},
                                {"extracted_entities.keywords": {"$regex": query, "$options": "i"}},
                                {"extracted_entities.cryptocurrencies": {"$regex": query, "$options": "i"}},
                                {"sentiment.keywords": {"$regex": query, "$options": "i"}},
                            ]
                        },
                        {"isdeleted": {"$ne": True}}
                    ]
                }

                if crypto:
                    fallback_query["$and"].append(crypto_filter)

                posts = list(
                    col.find(fallback_query)
                      .sort("engagement_score", -1)
                      .limit(limit)
                )

            # FORMAT RESULTS
            results = []
            for p in posts:
                results.append({
                    "id": str(p.get("_id")),
                    "text": p.get("text"),
                    "platform": p.get("platform"),
                    "published_at": p.get("created_at"),
                    "hashtags": p.get("hashtags", []),
                    "cryptocurrencies": p.get("extracted_entities", {}).get("cryptocurrencies", []),
                    "engagement": p.get("engagement_score", 0),
                })

            return Response({
                "query": query,
                "results": results,
                "total": len(results)
            })

        except Exception as e:
            logger.error(f"Error searching social posts: {e}")
            return Response({
                "error": "Search failed",
                "detail": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def safe_pubdate(val):
    # Handles int (timestamp), datetime, string, None
    if isinstance(val, int):
        # Assume it's a Unix timestamp
        try:
            return datetime.utcfromtimestamp(val).isoformat()
        except Exception:
            return ""
    elif isinstance(val, datetime):
        return val.isoformat()
    elif isinstance(val, str):
        try:
            # Try to parse string to datetime
            return dt_parse(val).isoformat()
        except Exception:
            return val
    return ""

class GetCombinedFeedAPI(APIView):
    """Get combined news articles and social posts feed"""
    permission_classes = [AllowAny]

    # Helper: always return a dict
    def safe_dict(self, value):
        return value if isinstance(value, dict) else {}

    # Helper: always return a string
    def safe_str(self, value):
        return str(value) if value is not None else ""

    @swagger_auto_schema(
        operation_description="Get combined feed of news articles and social posts.",
        manual_parameters=[
            openapi.Parameter('threshold', openapi.IN_QUERY,
                description="Minimum trust score",
                type=openapi.TYPE_NUMBER, default=6.0),
            openapi.Parameter('limit', openapi.IN_QUERY,
                description="Number of items to return",
                type=openapi.TYPE_INTEGER, default=50),
            openapi.Parameter('hours', openapi.IN_QUERY,
                description="Hours of historical data",
                type=openapi.TYPE_INTEGER, default=24),
            openapi.Parameter('content_type', openapi.IN_QUERY,
                description="Filter by type",
                type=openapi.TYPE_STRING,
                enum=['all', 'news', 'social'], default='all'),
        ],
        responses={200: "Combined feed"},
        tags=['News Feed']
    )
    def get(self, request):

        try:
            threshold = float(request.GET.get('threshold', 6.0))
            limit = int(request.GET.get('limit', 50))
            hours = int(request.GET.get('hours', 24))
            content_type = request.GET.get('content_type', 'all')

            mongo_manager = get_mongo_manager()
            combined_items = []

            # ===================================
            # FETCH NEWS ARTICLES
            # ===================================
            if content_type in ['all', 'news']:
                articles = mongo_manager.get_high_credibility_articles(
                    trust_score_threshold=threshold,
                    hours_back=hours
                )

                for article in articles:
                    if not isinstance(article, dict):
                        logger.warning(f"Ignoring invalid article (not dict): {article}")
                        continue

                    source = self.safe_dict(article.get('source'))
                    sentiment = self.safe_dict(article.get('sentiment_analysis'))

                    combined_items.append({
                        "id": self.safe_str(article.get('_id') or article.get('id')),
                        "type": "news",
                        "title": article.get("title"),
                        "content": article.get("description", "")[:200],
                        "url": article.get("url"),
                        "source": source.get("title", "Unknown"),
                        "platform": "news",
                        "published_at": article.get("published_at"),
                        "trust_score": round(article.get("trust_score", 0), 2),
                        "sentiment": sentiment.get("label", "neutral"),
                    })

            # ===================================
            # FETCH SOCIAL POSTS
            # ===================================
            if content_type in ['all', 'social']:
                posts = mongo_manager.get_high_credibility_social_posts(
                    trust_score_threshold=threshold,
                    hours_back=hours
                )

                for post in posts:
                    if not isinstance(post, dict):
                        logger.warning(f"Ignoring invalid post (not dict): {post}")
                        continue

                    sentiment = self.safe_dict(post.get('sentiment_analysis'))

                    combined_items.append({
                        "id": self.safe_str(post.get('_id') or post.get('source_id')),
                        "type": "social",
                        "title": post.get("title", post.get("content", "")[:100]),
                        "content": post.get("content", "")[:200],
                        "url": post.get("url"),
                        "source": post.get("author_username", "Unknown"),
                        "platform": post.get("platform", "unknown"),
                        "published_at": post.get("published_at"),
                        "trust_score": round(post.get("trust_score", 0), 2),
                        "sentiment": sentiment.get("label", "neutral"),
                    })

            # ===================================
            # SORT COMBINED RESULT
            # ===================================
            combined_items.sort(
                key=lambda x: safe_pubdate(x.get("published_at")),
                reverse=True
            )

            # Limit the output
            final_items = combined_items[:limit]

            # ===================================
            # RESPONSE
            # ===================================
            return Response({
                "items": final_items,
                "total": len(final_items),
                "breakdown": {
                    "news": sum(1 for i in final_items if i["type"] == "news"),
                    "social": sum(1 for i in final_items if i["type"] == "social"),
                },
                "filters_applied": {
                    "threshold": threshold,
                    "hours": hours,
                    "content_type": content_type
                },
                "timestamp": timezone.now().isoformat()
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error fetching combined feed: {e}", exc_info=True)
            return Response({
                "error": "Failed to fetch combined feed",
                "detail": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GetSocialStatsAPI(APIView):
    """Get social media statistics"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get statistics for social media posts.",
        manual_parameters=[
            openapi.Parameter('hours', openapi.IN_QUERY, 
                description="Hours to analyze", 
                type=openapi.TYPE_INTEGER, default=24),
        ],
        responses={200: "Social media statistics"},
        tags=['Social Feed']
    )
    def get(self, request):
        try:
            hours = int(request.GET.get('hours', 24))
            
            mongo_manager = get_mongo_manager()
            stats = mongo_manager.get_social_statistics(hours_back=hours)
            
            return Response(stats, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching social stats: {e}")
            return Response({
                'error': 'Failed to fetch statistics',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ============================================================================
# CREDIBILITY APIs
# ============================================================================

class AnalyzeContentAPI(APIView):
    """Analyze content credibility on-demand"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Analyze content credibility in real-time.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'title': openapi.Schema(type=openapi.TYPE_STRING, description="Content title"),
                'description': openapi.Schema(type=openapi.TYPE_STRING, description="Content body"),
                'source': openapi.Schema(type=openapi.TYPE_STRING, description="Source name"),
                'url': openapi.Schema(type=openapi.TYPE_STRING, description="Content URL"),
            },
            required=['title', 'description']
        ),
        responses={
            200: openapi.Response(
                description="Credibility analysis result",
                examples={
                    "application/json": {
                        "trust_score": 7.5,
                        "breakdown": {
                            "source": 8.0,
                            "content": 6.5,
                            "cross_check": 7.0,
                            "source_history": 8.0,
                            "recency": 9.0
                        },
                        "action": "normal_flow",
                        "flags": [],
                        "reasoning": "..."
                    }
                }
            )
        },
        tags=['Credibility']
    )
    def post(self, request):
        try:
            data = request.data
            
            if not data.get('title') or not data.get('description'):
                return Response({
                    'error': 'Title and description are required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            engine = get_credibility_engine()
            
            # Prepare content for analysis
            content = {
                'source_id': f'manual_analysis_{timezone.now().timestamp()}',
                'title': data.get('title'),
                'description': data.get('description'),
                'url': data.get('url', ''),
                'source': {'title': data.get('source', 'Unknown')},
                'platform': 'manual',
                'published_at': timezone.now().isoformat(),
            }
            
            # Calculate trust score
            trust_score = engine.calculate_trust_score(
                content_data=content,
                sentiment_data=None,
                entities_data=None,
                verification_result=None
            )
            action = engine.determine_content_action(trust_score)
            
            return Response({
                'trust_score': round(trust_score.final_score, 2),
                'breakdown': {
                    # ✅ FIX: Use correct field names from TrustScore
                    'source_score': round(trust_score.source_score, 2),
                    'content_score': round(trust_score.content_score, 2),
                    'engagement_score': round(trust_score.engagement_score, 2),
                    'cross_check_score': round(trust_score.cross_check_score, 2),
                    'source_history_score': round(trust_score.source_history_score, 2),
                    'recency_score': round(trust_score.recency_score, 2)
                },
                'weights': {
                    'source': 0.40,
                    'content': 0.25,
                    'cross_check': 0.15,
                    'source_history': 0.15,
                    'recency': 0.05
                },
                'action': action['action'],
                'action_details': action,
                'flags': trust_score.flags,
                'reasoning': trust_score.reasoning,
                'confidence': round(trust_score.confidence, 2),
                'cross_reference_matches': trust_score.cross_reference_matches,
                'corroboration_sources': trust_score.corroboration_sources
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            import traceback
            traceback.print_exc()
            return Response({
                'error': 'Analysis failed',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GetSourceHistoryAPI(APIView):
    """Get source reliability history"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get historical reliability data for news sources.",
        manual_parameters=[
            openapi.Parameter('source', openapi.IN_QUERY, 
                description="Filter by source name", type=openapi.TYPE_STRING),
            openapi.Parameter('min_articles', openapi.IN_QUERY, 
                description="Minimum articles for source to be included", 
                type=openapi.TYPE_INTEGER, default=5),
        ],
        responses={
            200: openapi.Response(
                description="Source reliability data",
                examples={
                    "application/json": {
                        "sources": [
                            {
                                "name": "CoinDesk",
                                "total_articles": 150,
                                "accuracy_rate": 0.92,
                                "reliability_score": 8.5,
                                "flagged_articles": 5,
                                "retracted_articles": 0
                            }
                        ]
                    }
                }
            )
        },
        tags=['Credibility']
    )
    def get(self, request):
        try:
            source_filter = request.GET.get('source')
            min_articles = int(request.GET.get('min_articles', 5))
            
            engine = get_credibility_engine()
            
            sources_data = []
            for source_name, record in engine.source_history.items():
                # Apply filters
                if source_filter and source_filter.lower() not in source_name.lower():
                    continue
                if record.total_articles < min_articles:
                    continue
                
                sources_data.append({
                    'name': source_name,
                    'total_articles': record.total_articles,
                    'accurate_articles': record.accurate_articles,
                    'accuracy_rate': round(record.accuracy_rate, 3),
                    'reliability_score': round(record.reliability_score, 2),
                    'flagged_articles': record.flagged_articles,
                    'retracted_articles': record.retracted_articles,
                    # ✅ FIX: Use 'avg_trust_score' instead of 'average_trust_score'
                    'average_trust_score': round(record.avg_trust_score, 2),
                    'last_updated': record.last_updated.isoformat() if record.last_updated else None
                })
            
            # Sort by reliability score
            sources_data.sort(key=lambda x: x['reliability_score'], reverse=True)
            
            return Response({
                'sources': sources_data,
                'total_tracked': len(sources_data)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching source history: {e}")
            return Response({
                'error': 'Failed to fetch source history',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



class GetCredibilityStatsAPI(APIView):
    """Get credibility system statistics"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get overall credibility system statistics and distributions.",
        responses={200: "System statistics"},
        tags=['Credibility']
    )
    def get(self, request):
        try:
            mongo_manager = get_mongo_manager()
            engine = get_credibility_engine()
            threshold_manager = get_threshold_manager()
            
            # Get article statistics
            collection = mongo_manager.collections['news_articles']
            
            total_articles = collection.count_documents({'isdeleted': {'$ne': True}})
            
            # Score distribution
            score_ranges = {
                'high_credibility': collection.count_documents({
                    'trust_score': {'$gte': 8.0},
                    'isdeleted': {'$ne': True}
                }),
                'medium_credibility': collection.count_documents({
                    'trust_score': {'$gte': 6.0, '$lt': 8.0},
                    'isdeleted': {'$ne': True}
                }),
                'low_credibility': collection.count_documents({
                    'trust_score': {'$gte': 4.0, '$lt': 6.0},
                    'isdeleted': {'$ne': True}
                }),
                'very_low_credibility': collection.count_documents({
                    'trust_score': {'$lt': 4.0},
                    'isdeleted': {'$ne': True}
                })
            }
            
            # Status distribution
            status_distribution = {
                'approved': collection.count_documents({'status': 'approved', 'isdeleted': {'$ne': True}}),
                'pending': collection.count_documents({'status': 'pending', 'isdeleted': {'$ne': True}}),
                'flagged': collection.count_documents({'status': 'flagged', 'isdeleted': {'$ne': True}}),
                'rejected': collection.count_documents({'status': 'rejected', 'isdeleted': {'$ne': True}})
            }
            
            # Average scores (using aggregation)
            avg_pipeline = [
                {'$match': {'isdeleted': {'$ne': True}, 'trust_score': {'$exists': True}}},
                {'$group': {
                    '_id': None,
                    'avg_trust_score': {'$avg': '$trust_score'},
                    'min_trust_score': {'$min': '$trust_score'},
                    'max_trust_score': {'$max': '$trust_score'}
                }}
            ]
            avg_result = list(collection.aggregate(avg_pipeline))
            avg_stats = avg_result[0] if avg_result else {
                'avg_trust_score': 0, 'min_trust_score': 0, 'max_trust_score': 0
            }
            
            return Response({
                'total_articles': total_articles,
                'score_distribution': score_ranges,
                'status_distribution': status_distribution,
                'score_statistics': {
                    'average': round(avg_stats.get('avg_trust_score', 0), 2),
                    'minimum': round(avg_stats.get('min_trust_score', 0), 2),
                    'maximum': round(avg_stats.get('max_trust_score', 0), 2)
                },
                'thresholds': threshold_manager.get_thresholds(),
                'news_weights': engine.news_weights,
                'social_weights': engine.social_weights,
                'sources_tracked': len(engine.source_history)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching credibility stats: {e}")
            return Response({
                'error': 'Failed to fetch statistics',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




class GetSentimentOverviewAPI(APIView):
    """Get overall market sentiment from content"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get overall market sentiment aggregated from news articles.",
        manual_parameters=[
            openapi.Parameter('hours', openapi.IN_QUERY, 
                description="Hours to analyze", type=openapi.TYPE_INTEGER, default=24),
            openapi.Parameter('crypto', openapi.IN_QUERY, 
                description="Filter by cryptocurrency", type=openapi.TYPE_STRING),
        ],
        responses={
            200: openapi.Response(
                description="Sentiment overview",
                examples={
                    "application/json": {
                        "overall_sentiment": "bullish",
                        "sentiment_score": 0.35,
                        "distribution": {
                            "bullish": 45,
                            "neutral": 30,
                            "bearish": 25
                        },
                        "trend": "improving"
                    }
                }
            )
        },
        tags=['Trending & Topics']
    )
    def get(self, request):
        try:
            hours = int(request.GET.get('hours', 24))
            crypto = request.GET.get('crypto')
            
            mongo_manager = get_mongo_manager()
            collection = mongo_manager.collections['news_articles']
            
            # Calculate time range
            time_threshold = timezone.now() - timedelta(hours=hours)
            
            # Build query
            query = {
                'isdeleted': {'$ne': True},
                'published_at': {'$gte': time_threshold.isoformat()}
            }
            
            if crypto:
                query['$or'] = [
                    {'extracted_entities.cryptocurrencies': {'$regex': crypto, '$options': 'i'}},
                    {'crypto_relevance.mentioned_cryptocurrencies': {'$regex': crypto, '$options': 'i'}},
                    {'crypto_analysis.mentioned_cryptocurrencies': {'$regex': crypto, '$options': 'i'}}
                ]
            
            # Get articles with sentiment
            articles = list(collection.find(query, {
                'sentiment_analysis': 1,
                'published_at': 1
            }))
            
            if not articles:
                return Response({
                    'overall_sentiment': 'neutral',
                    'sentiment_score': 0,
                    'distribution': {'bullish': 0, 'neutral': 0, 'bearish': 0},
                    'total_articles': 0,
                    'message': 'No articles found in the specified time range'
                }, status=status.HTTP_200_OK)
            
            # Calculate distribution
            distribution = {'bullish': 0, 'neutral': 0, 'bearish': 0}
            total_score = 0
            scored_articles = 0
            
            for article in articles:
                sentiment = article.get('sentiment_analysis', {})
                
                # ✅ FIX: Use 'label' instead of 'sentiment_label'
                label = sentiment.get('label', 'neutral').lower()
                
                if label in ['bullish', 'positive']:
                    distribution['bullish'] += 1
                elif label in ['bearish', 'negative']:
                    distribution['bearish'] += 1
                else:
                    distribution['neutral'] += 1
                
                # ✅ FIX: Use 'score' instead of 'crypto_sentiment_score'
                score = sentiment.get('score', 0)
                if score:
                    total_score += score
                    scored_articles += 1
            
            avg_score = total_score / scored_articles if scored_articles > 0 else 0
            
            # Determine overall sentiment
            if avg_score > 0.2:
                overall = 'bullish'
            elif avg_score < -0.2:
                overall = 'bearish'
            else:
                overall = 'neutral'
            
            return Response({
                'overall_sentiment': overall,
                'sentiment_score': round(avg_score, 3),
                'distribution': distribution,
                'distribution_percentages': {
                    'bullish': round(distribution['bullish'] / len(articles) * 100, 1),
                    'neutral': round(distribution['neutral'] / len(articles) * 100, 1),
                    'bearish': round(distribution['bearish'] / len(articles) * 100, 1)
                },
                'total_articles': len(articles),
                'time_range_hours': hours,
                'crypto_filter': crypto
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching sentiment overview: {e}")
            return Response({
                'error': 'Failed to fetch sentiment overview',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




# ============================================================================
# TRENDING & TOPICS APIs
# ============================================================================
class GetTrendingAPI(APIView):
    """Get trending hashtags and keywords"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get trending hashtags and keywords in crypto news.",
        manual_parameters=[
            openapi.Parameter('limit', openapi.IN_QUERY, 
                description="Number of trending items", type=openapi.TYPE_INTEGER, default=20),
            openapi.Parameter('hours', openapi.IN_QUERY, 
                description="Hours to analyze", type=openapi.TYPE_INTEGER, default=24),
            openapi.Parameter('type', openapi.IN_QUERY, 
                description="Type of trending items", type=openapi.TYPE_STRING,
                enum=['all', 'hashtags', 'keywords'], default='all'),
        ],
        responses={
            200: openapi.Response(
                description="Trending data",
                examples={
                    "application/json": {
                        "hashtags": [
                            {"tag": "#Bitcoin", "count": 150, "sentiment": 0.6, "trend_score": 8.5}
                        ],
                        "keywords": [
                            {"keyword": "ETF", "count": 80, "sentiment": 0.4, "trend_score": 7.2}
                        ]
                    }
                }
            )
        },
        tags=['Trending & Topics']
    )
    def get(self, request):
        try:
            limit = int(request.GET.get('limit', 20))
            hours = int(request.GET.get('hours', 24))
            trend_type = request.GET.get('type', 'all')
            
            # Calculate time threshold
            time_threshold = timezone.now() - timedelta(hours=hours)
            
            response_data = {}
            
            if trend_type in ['all', 'hashtags']:
                from myapp.models import Trendinghashtag
                
                # Get hashtags within time window, ordered by trend_score
                trending_hashtags = Trendinghashtag.objects.filter(
                    timestamp__gte=time_threshold
                ).order_by('-trend_score', '-timestamp')[:limit]
                
                response_data['hashtags'] = [
                    {
                        'tag': th.hashtag,
                        'count': th.count_24h,  # Use 24h count for display
                        'sentiment': round(th.avg_sentiment, 3),
                        'trend_score': round(th.trend_score, 2),
                        'velocity': round(th.velocity, 2),
                        'counts': {
                            '1h': th.count_1h,
                            '6h': th.count_6h,
                            '24h': th.count_24h
                        },
                        'timestamp': th.timestamp.isoformat()
                    }
                    for th in trending_hashtags
                ]
            
            if trend_type in ['all', 'keywords']:
                from myapp.models import Trendingkeyword
                
                # Get keywords within time window, ordered by velocity
                trending_keywords = Trendingkeyword.objects.filter(
                    timestamp__gte=time_threshold
                ).order_by('-velocity', '-timestamp')[:limit]
                
                response_data['keywords'] = [
                    {
                        'keyword': tk.keyword,
                        'count': tk.count_24h,  # Use 24h count for display
                        'sentiment': round(tk.avg_sentiment, 3),
                        'velocity': round(tk.velocity, 2),
                        'counts': {
                            '1h': tk.count_1h,
                            '6h': tk.count_6h,
                            '24h': tk.count_24h
                        },
                        'sources': tk.sources,  # JSONField with platform counts
                        'timestamp': tk.timestamp.isoformat()
                    }
                    for tk in trending_keywords
                ]
            
            # ================================================================
            # Summary statistics
            # ================================================================
            summary = {}
            if trend_type in ['all', 'hashtags']:
                summary['total_hashtags'] = Trendinghashtag.objects.filter(
                    timestamp__gte=time_threshold
                ).count()
            
            if trend_type in ['all', 'keywords']:
                summary['total_keywords'] = Trendingkeyword.objects.filter(
                    timestamp__gte=time_threshold
                ).count()
            
            summary['analysis_window_hours'] = hours
            summary['last_updated'] = timezone.now().isoformat()
            
            response_data['summary'] = summary
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching trending data: {e}")
            import traceback
            traceback.print_exc()
            return Response({
                'error': 'Failed to fetch trending data',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class GetTopicsAPI(APIView):
    """Get discovered topics from news content"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get automatically discovered topics from news content.",
        manual_parameters=[
            openapi.Parameter('limit', openapi.IN_QUERY, 
                description="Number of topics", type=openapi.TYPE_INTEGER, default=10),
            openapi.Parameter('hours', openapi.IN_QUERY,
                description="Hours to analyze", type=openapi.TYPE_INTEGER, default=24),
            openapi.Parameter('spikes_only', openapi.IN_QUERY,
                description="Show only spiking topics", type=openapi.TYPE_BOOLEAN, default=False),
        ],
        responses={
            200: openapi.Response(
                description="Discovered topics",
                examples={
                    "application/json": {
                        "topics": [
                            {
                                "id": 0,
                                "name": "Bitcoin_ETF_regulation",
                                "keywords": ["bitcoin", "etf", "sec", "approval"],
                                "document_count": 45,
                                "is_spike": True
                            }
                        ]
                    }
                }
            )
        },
        tags=['Trending & Topics']
    )
    def get(self, request):
        try:
            limit = int(request.GET.get('limit', 10))
            hours = int(request.GET.get('hours', 24))
            spikes_only = request.GET.get('spikes_only', 'false').lower() == 'true'
            
            # Calculate time threshold
            time_threshold = timezone.now() - timedelta(hours=hours)
            
            from myapp.models import Trendingtopic
            
            # Build query
            query = Trendingtopic.objects.filter(timestamp__gte=time_threshold)
            
            if spikes_only:
                query = query.filter(is_spike=True)
            
            # Order by velocity (most trending first)
            trending_topics = query.order_by('-velocity', '-timestamp')[:limit]
            
            topics = []
            for topic in trending_topics:
                topics.append({
                    'id': topic.topic_id,
                    'name': topic.topic_name,
                    'keywords': topic.keywords[:10],  # Top 10 keywords
                    'document_count': topic.document_count,
                    'velocity': round(topic.velocity, 2),
                    'avg_sentiment': round(topic.avg_sentiment, 3),
                    'is_spike': topic.is_spike,
                    'timestamp': topic.timestamp.isoformat()
                })
            
            # Get statistics
            total_topics = Trendingtopic.objects.filter(
                timestamp__gte=time_threshold
            ).values('topic_id').distinct().count()
            
            spiking_topics = Trendingtopic.objects.filter(
                timestamp__gte=time_threshold,
                is_spike=True
            ).count()
            
            return Response({
                'topics': topics,
                'statistics': {
                    'total_topics': total_topics,
                    'spiking_topics': spiking_topics,
                    'time_window_hours': hours,
                    'topics_returned': len(topics)
                },
                'timestamp': timezone.now().isoformat()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching topics: {e}")
            import traceback
            traceback.print_exc()
            return Response({
                'error': 'Failed to fetch topics',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GetTrendingHistoryAPI(APIView):
    """Get historical trending data for charts"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get historical trending data for visualization.",
        manual_parameters=[
            openapi.Parameter('item', openapi.IN_QUERY,
                description="Hashtag or keyword to get history for",
                type=openapi.TYPE_STRING, required=True),
            openapi.Parameter('type', openapi.IN_QUERY,
                description="Type of item",
                type=openapi.TYPE_STRING,
                enum=['hashtag', 'keyword'], required=True),
            openapi.Parameter('hours', openapi.IN_QUERY,
                description="Hours of history",
                type=openapi.TYPE_INTEGER, default=24),
        ],
        responses={200: "Historical trending data"},
        tags=['Trending & Topics']
    )
    def get(self, request):
        try:
            item = request.GET.get('item', '').strip()
            item_type = request.GET.get('type', 'hashtag')
            hours = int(request.GET.get('hours', 24))
            
            if not item:
                return Response({
                    'error': 'Item parameter is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            time_threshold = timezone.now() - timedelta(hours=hours)
            
            history = []
            
            if item_type == 'hashtag':
                from myapp.models import Trendinghashtag
                
                records = Trendinghashtag.objects.filter(
                    hashtag=item,
                    timestamp__gte=time_threshold
                ).order_by('timestamp')
                
                history = [
                    {
                        'timestamp': record.timestamp.isoformat(),
                        'count_1h': record.count_1h,
                        'count_6h': record.count_6h,
                        'count_24h': record.count_24h,
                        'velocity': round(record.velocity, 2),
                        'sentiment': round(record.avg_sentiment, 3),
                        'trend_score': round(record.trend_score, 2)
                    }
                    for record in records
                ]
                
            elif item_type == 'keyword':
                from myapp.models import Trendingkeyword
                
                records = Trendingkeyword.objects.filter(
                    keyword=item,
                    timestamp__gte=time_threshold
                ).order_by('timestamp')
                
                history = [
                    {
                        'timestamp': record.timestamp.isoformat(),
                        'count_1h': record.count_1h,
                        'count_6h': record.count_6h,
                        'count_24h': record.count_24h,
                        'velocity': round(record.velocity, 2),
                        'sentiment': round(record.avg_sentiment, 3),
                        'sources': record.sources
                    }
                    for record in records
                ]
            
            return Response({
                'item': item,
                'type': item_type,
                'history': history,
                'data_points': len(history),
                'time_range_hours': hours
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching trending history: {e}")
            return Response({
                'error': 'Failed to fetch history',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            