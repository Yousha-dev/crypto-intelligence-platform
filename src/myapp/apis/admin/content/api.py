import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.utils import timezone
 
from myapp.permissions import IsUserAccess
from myapp.services.content.integrator_service import get_integrator_service
from myapp.services.content.credibility_engine import get_credibility_engine, get_threshold_manager
from myapp.services.mongo_manager import get_mongo_manager

logger = logging.getLogger(__name__)
 
# ============================================================================
# 👮 MODERATION APIs 
# ============================================================================

class GetModerationQueueAPI(APIView):
    """Get content pending moderation"""
    permission_classes = [IsUserAccess]
 
    @swagger_auto_schema(
        operation_description="Get articles pending moderation review.",
        manual_parameters=[
            openapi.Parameter('status', openapi.IN_QUERY, 
                description="Filter by status", type=openapi.TYPE_STRING,
                enum=['pending', 'flagged', 'all'], default='flagged'),
            openapi.Parameter('priority', openapi.IN_QUERY, 
                description="Filter by priority", type=openapi.TYPE_STRING,
                enum=['high', 'medium', 'low', 'all'], default='all'),
            openapi.Parameter('limit', openapi.IN_QUERY, 
                description="Max results", type=openapi.TYPE_INTEGER, default=50),
        ],
        responses={200: "Moderation queue"},
        tags=['Moderation']
    )
    def get(self, request):
        try:
            status_filter = request.GET.get('status', 'flagged')
            priority = request.GET.get('priority', 'all')
            limit = int(request.GET.get('limit', 50))
            
            mongo_manager = get_mongo_manager()
            collection = mongo_manager.collections['news_articles']
            
            # Build query
            query = {'isdeleted': {'$ne': True}}
            
            if status_filter == 'flagged':
                query['status'] = 'flagged'
            elif status_filter == 'pending':
                query['status'] = 'pending'
            elif status_filter == 'all':
                query['status'] = {'$in': ['pending', 'flagged']}
            
            if priority != 'all':
                query['credibility_analysis.recommended_action.priority'] = priority
            
            # Get articles sorted by priority and score
            articles = list(
                collection.find(query)
                .sort([
                    ('credibility_analysis.recommended_action.priority', 1),
                    ('trust_score', 1)
                ])
                .limit(limit)
            )
            
            # Format queue items
            queue_items = []
            for article in articles:
                credibility = article.get('credibility_analysis', {})
                action = credibility.get('recommended_action', {})
                
                queue_items.append({
                    'id': str(article.get('_id', article.get('id'))),
                    'title': article.get('title'),
                    'source': article.get('source', {}).get('title', 'Unknown'),
                    'published_at': article.get('published_at'),
                    'trust_score': round(article.get('trust_score', 0), 2),
                    'status': article.get('status'),
                    'priority': action.get('priority', 'normal'),
                    'action': action.get('action'),
                    'flags': credibility.get('flags', []),
                    'reasoning': credibility.get('reasoning', ''),
                    'url': article.get('url')
                })
            
            # Count by priority
            priority_counts = {
                'high': sum(1 for item in queue_items if item['priority'] == 'high'),
                'medium': sum(1 for item in queue_items if item['priority'] == 'medium'),
                'normal': sum(1 for item in queue_items if item['priority'] == 'normal'),
                'low': sum(1 for item in queue_items if item['priority'] == 'low')
            }
            
            return Response({
                'queue': queue_items,
                'total': len(queue_items),
                'priority_counts': priority_counts
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching moderation queue: {e}")
            return Response({
                'error': 'Failed to fetch moderation queue',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ApproveContentAPI(APIView):
    """Approve content for publication"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Approve content for publication.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'notes': openapi.Schema(type=openapi.TYPE_STRING, description="Moderator notes"),
            }
        ),
        responses={200: "Content approved", 404: "Content not found"},
        tags=['Moderation']
    )
    def post(self, request, article_id):
        try:
            user_id = getattr(request, "user_id", None)
            notes = request.data.get('notes', '')
            
            mongo_manager = get_mongo_manager()
            engine = get_credibility_engine()
            
            # Update article status
            result = mongo_manager.collections['news_articles'].update_one(
                {'$or': [{'id': article_id}, {'_id': article_id}]},
                {
                    '$set': {
                        'status': 'approved',
                        'moderation': {
                            'action': 'approved',
                            'moderator_id': user_id,
                            'timestamp': timezone.now().isoformat(),
                            'notes': notes
                        }
                    }
                }
            )
            
            if result.modified_count == 0:
                return Response({
                    'error': 'Article not found or already processed'
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Update source history (accurate prediction)
            article = mongo_manager.collections['news_articles'].find_one({
                '$or': [{'id': article_id}, {'_id': article_id}]
            })
            
            if article:
                source_name = article.get('source', {}).get('title', 'unknown')
                engine.update_source_history(
                    source_name=source_name,
                    was_accurate=True,
                    was_flagged=False,
                    was_retracted=False,
                    trust_score=article.get('trust_score', 5.0)
                )
            
            return Response({
                'message': 'Content approved successfully',
                'article_id': article_id
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error approving content: {e}")
            return Response({
                'error': 'Failed to approve content',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RejectContentAPI(APIView):
    """Reject content"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Reject content and prevent publication.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'reason': openapi.Schema(type=openapi.TYPE_STRING, description="Rejection reason"),
                'notes': openapi.Schema(type=openapi.TYPE_STRING, description="Additional notes"),
            },
            required=['reason']
        ),
        responses={200: "Content rejected", 404: "Content not found"},
        tags=['Moderation']
    )
    def post(self, request, article_id):
        try:
            user_id = getattr(request, "user_id", None)
            reason = request.data.get('reason', 'Not specified')
            notes = request.data.get('notes', '')
            
            mongo_manager = get_mongo_manager()
            engine = get_credibility_engine()
            
            # Get article first for source history update
            article = mongo_manager.collections['news_articles'].find_one({
                '$or': [{'id': article_id}, {'_id': article_id}]
            })
            
            if not article:
                return Response({
                    'error': 'Article not found'
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Update article status
            result = mongo_manager.collections['news_articles'].update_one(
                {'$or': [{'id': article_id}, {'_id': article_id}]},
                {
                    '$set': {
                        'status': 'rejected',
                        'moderation': {
                            'action': 'rejected',
                            'reason': reason,
                            'moderator_id': user_id,
                            'timestamp': timezone.now().isoformat(),
                            'notes': notes
                        }
                    }
                }
            )
            
            # Update source history (flagged/inaccurate)
            source_name = article.get('source', {}).get('title', 'unknown')
            engine.update_source_history(
                source_name=source_name,
                was_accurate=False,
                was_flagged=True,
                was_retracted=False,
                trust_score=article.get('trust_score', 5.0)
            )
            
            return Response({
                'message': 'Content rejected successfully',
                'article_id': article_id,
                'reason': reason
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error rejecting content: {e}")
            return Response({
                'error': 'Failed to reject content',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FlagContentAPI(APIView):
    """Flag content for additional review"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Flag content for additional review (user reporting).",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'reason': openapi.Schema(type=openapi.TYPE_STRING, 
                    description="Flag reason",
                    enum=['misinformation', 'spam', 'inappropriate', 'outdated', 'other']),
                'details': openapi.Schema(type=openapi.TYPE_STRING, description="Additional details"),
            },
            required=['reason']
        ),
        responses={200: "Content flagged", 404: "Content not found"},
        tags=['Moderation']
    )
    def post(self, request, article_id):
        try:
            user_id = getattr(request, "user_id", None)
            reason = request.data.get('reason', 'other')
            details = request.data.get('details', '')
            
            mongo_manager = get_mongo_manager()
            
            # Update article with flag
            result = mongo_manager.collections['news_articles'].update_one(
                {'$or': [{'id': article_id}, {'_id': article_id}]},
                {
                    '$set': {'status': 'flagged'},
                    '$push': {
                        'user_flags': {
                            'user_id': user_id,
                            'reason': reason,
                            'details': details,
                            'timestamp': timezone.now().isoformat()
                        }
                    }
                }
            )
            
            if result.modified_count == 0:
                return Response({
                    'error': 'Article not found'
                }, status=status.HTTP_404_NOT_FOUND)
            
            return Response({
                'message': 'Content flagged for review',
                'article_id': article_id
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error flagging content: {e}")
            return Response({
                'error': 'Failed to flag content',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ============================================================================
# ⚙️ SYSTEM APIs
# ============================================================================

class GetSystemHealthAPI(APIView):
    """Get system health status"""
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get system health status including all components.",
        responses={
            200: openapi.Response(
                description="System health",
                examples={
                    "application/json": {
                        "status": "healthy",
                        "components": {
                            "database": {"status": "connected", "articles": 1500},
                            "credibility_engine": {"status": "active"},
                            "sentiment_analyzer": {"status": "active"}
                        }
                    }
                }
            )
        },
        tags=['System']
    )
    def get(self, request):
        try:
            service = get_integrator_service()
            health = service.get_system_health()
            
            return Response(health, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching system health: {e}")
            return Response({
                'status': 'error',
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GetDatabaseStatsAPI(APIView):
    """Get database statistics"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Get detailed database statistics.",
        responses={200: "Database statistics"},
        tags=['System']
    )
    def get(self, request):
        try:
            mongo_manager = get_mongo_manager()
            stats = mongo_manager.get_statistics()
            
            return Response({
                'statistics': stats,
                'timestamp': timezone.now().isoformat()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching database stats: {e}")
            return Response({
                'error': 'Failed to fetch statistics',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class UpdateThresholdsAPI(APIView):
    """Update credibility thresholds"""
    permission_classes = [IsUserAccess]

    @swagger_auto_schema(
        operation_description="Update credibility scoring thresholds.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'high_trust': openapi.Schema(type=openapi.TYPE_NUMBER, description="High trust threshold (default: 8.0)"),
                'medium_trust': openapi.Schema(type=openapi.TYPE_NUMBER, description="Medium trust threshold (default: 6.0)"),
                'low_trust': openapi.Schema(type=openapi.TYPE_NUMBER, description="Low trust threshold (default: 4.0)"),
                'very_low_trust': openapi.Schema(type=openapi.TYPE_NUMBER, description="Very low trust threshold (default: 2.0)"),
            }
        ),
        responses={200: "Thresholds updated", 400: "Invalid values"},
        tags=['System']
    )
    def post(self, request):
        try:
            manager = get_threshold_manager()
            
            updates = {}
            for key in ['high_trust', 'medium_trust', 'low_trust', 'very_low_trust']:
                if key in request.data:
                    value = float(request.data[key])
                    if 0 <= value <= 10:
                        manager.set_threshold(key, value)
                        updates[key] = value
                    else:
                        return Response({
                            'error': f'Invalid value for {key}: must be between 0 and 10'
                        }, status=status.HTTP_400_BAD_REQUEST)
            
            return Response({
                'message': 'Thresholds updated successfully',
                'updated': updates,
                'current_thresholds': manager.get_thresholds()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error updating thresholds: {e}")
            return Response({
                'error': 'Failed to update thresholds',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)