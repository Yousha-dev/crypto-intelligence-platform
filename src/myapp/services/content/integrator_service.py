"""
Content Integration Service - Main orchestrator for content ingestion pipeline
Coordinates text processing, sentiment analysis, credibility scoring, and storage
"""

import logging
from datetime import datetime, timedelta
from django.utils import timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from celery import shared_task
import hashlib

from datetime import timezone as timezoneDt
from .text_processor import get_text_processor
from .sentiment_analyzer import get_sentiment_analyzer
from .topic_modeler import get_topic_modeler
from .cross_verification_engine import (
    get_verification_engine, 
    VerificationResult 
)
from ..mongo_manager import get_mongo_manager
from .credibility_engine import get_credibility_engine, get_threshold_manager, TrustScore

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of content processing"""
    total_processed: int
    approved: int
    pending: int
    flagged: int
    errors: int
    average_trust_score: float
    processing_time_seconds: float
 
class ContentIntegrationService:
    """
    Main orchestrator service - coordinates all content processing steps
    Follows single responsibility: orchestration only, delegates to specialized services
    """
    
    def __init__(self):
        self.mongo_manager = get_mongo_manager()
        self.credibility_engine = get_credibility_engine()
        self.threshold_manager = get_threshold_manager()
        self.text_processor = get_text_processor()
        self.sentiment_analyzer = get_sentiment_analyzer()
        self.verification_engine = get_verification_engine()
        
        self.config = {
            'batch_size': 50,
            'cross_check_window_hours': 24,
            'auto_approve_threshold': 8.0,
            'manual_review_threshold': 4.0,
            'max_processing_time_minutes': 30 
        }
        
    def _safe_int(self, value, default: int = 0) -> int:
        """Safely convert to int"""
        if value is None:
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except (ValueError, TypeError):
                return default
        return default
    
    def _safe_float(self, value, default: float = 0.0) -> float:
        """Safely convert to float"""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        return default
    
    def _safe_dict(self, value, default=None) -> Dict:
        """Safely ensure dict type"""
        if default is None:
            default = {}
        if isinstance(value, dict):
            return value
        return default
    
    def _safe_list(self, value, default=None) -> List:
        """Safely ensure list type"""
        if default is None:
            default = []
        if isinstance(value, list):
            return value
        return default
    
    def _safe_str(self, value, default: str = '') -> str:
        """Safely ensure string type"""
        if value is None:
            return default
        if isinstance(value, str):
            return value
        return str(value) if value else default
        
    def _prepare_news_document(self, article_data: Dict) -> Dict:
        """
        Prepare news article document for MongoDB storage
        Ensures all fields are safe and consistent with MongoDB schema
        """
        now = timezone.now()
        
        # Identity fields
        source_id = (
            article_data.get('source_id') or 
            article_data.get('id') or
            article_data.get('guid') or
            str(hash(f"{article_data.get('title', '')}{article_data.get('url', '')}"))
        )
        
        content_for_hash = f"{article_data.get('title', '')}{article_data.get('url', '')}"
        content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()
        
        raw_platform = (
            article_data.get('platform') or
            article_data.get('source_name') or
            'unknown'
        )
        platform = raw_platform.lower()
        
        title = article_data.get('title', '').strip()
        if not title:
            logger.error(f"Article missing title: {article_data.get('source_id')}")
            title = f"[Untitled - {article_data.get('platform', 'unknown')}]"
        
        # Build document with type safety
        document = {
            # Required fields
            'source_id': str(source_id),
            'platform': platform,
            'content_hash': content_hash,
            'title': title,
            'trust_score': self._safe_float(article_data.get('trust_score'), 0.0),
            'status': self._safe_str(article_data.get('status', 'pending')),
            'created_at': now,
            'updated_at': now,
            
            # Content fields
            'description': self._safe_str(article_data.get('description')),
            'content': self._safe_str(article_data.get('content')),
            'url': self._safe_str(article_data.get('url')),
            'author': self._safe_str(article_data.get('author')),
            'published_at': (
                article_data.get('published_at') or 
                article_data.get('publishedAt') or
                article_data.get('published_on') or
                article_data.get('published') or
                now.isoformat()
            ),
            
            # Source info
            'source': self._safe_dict(article_data.get('source')),
            
            # Platform-specific fields
            'votes': self._safe_dict(article_data.get('votes')),
            'instruments': self._safe_list(article_data.get('instruments')),
            'kind': self._safe_str(article_data.get('kind')),
            'upvotes': self._safe_float(article_data.get('upvotes'), 0),
            'downvotes': self._safe_float(article_data.get('downvotes'), 0),
            'tags': self._safe_str(article_data.get('tags')),
            'categories': self._safe_str(article_data.get('categories')),
            'references': self._safe_list(article_data.get('references')),
            
            # Analysis results (type-safe)
            'credibility_analysis': self._safe_dict(article_data.get('credibility_analysis')),
            'sentiment_analysis': self._safe_dict(article_data.get('sentiment_analysis')) or self._create_empty_sentiment(),
            'extracted_entities': self._safe_dict(article_data.get('extracted_entities')),
            'text_processing': self._safe_dict(article_data.get('text_processing')),
            'action_info': self._safe_dict(article_data.get('action_info')),
            
            # Verification details
            'verification_details': self._safe_dict(article_data.get('verification_details')),
            
            # Timestamps
            'processing_timestamp': article_data.get('processing_timestamp'),
            'analysis_timestamp': article_data.get('analysis_timestamp'),
        }
        
        # Validation
        if not document['sentiment_analysis'] or document['sentiment_analysis'] == {}:
            logger.warning(f"Empty sentiment_analysis for article {source_id}")
            document['sentiment_analysis'] = self._create_empty_sentiment()
        
        if not document['credibility_analysis'] or document['credibility_analysis'] == {}:
            logger.warning(f"Empty credibility_analysis for article {source_id}")
        
        # Final validation
        assert isinstance(document['source_id'], str), "source_id must be string"
        assert isinstance(document['platform'], str), "platform must be string"
        assert isinstance(document['title'], str) and len(document['title']) > 0, "title must be non-empty string"
        assert isinstance(document['trust_score'], (int, float)), "trust_score must be number"
        assert isinstance(document['status'], str), "status must be string"
        assert isinstance(document['created_at'], datetime), "created_at must be datetime"
        
        return document
    
    # ========================================================================
    # SOCIAL DOCUMENT PREPARATION (NOW TYPE-SAFE!)
    # ========================================================================
    
    def _prepare_social_document(self, post_data: Dict) -> Dict:
        """
        Prepare social media post document for MongoDB storage
        NOW WITH COMPLETE TYPE SAFETY!
        """
        now = timezone.now()
        
        # Get platform
        raw_platform = post_data.get('platform', 'unknown')
        platform = raw_platform.lower()
        
        # ========================================================================
        # Extract and validate content
        # ========================================================================
        content = ""
        title = ""
        
        if platform == 'reddit':
            title = post_data.get('title', '')
            selftext = post_data.get('selftext', '')
            content = f"{title} {selftext}".strip()
            
        elif platform == 'twitter':
            title = post_data.get('text', '')[:100]
            content = post_data.get('text', '').strip()
            
        elif platform == 'youtube':
            title = post_data.get('title', '')
            description = post_data.get('description', '')
            content = f"{title} {description}".strip()
        
        # Ensure content is never empty
        if not content or content == '':
            content = title.strip()
            if not content:
                content = (
                    post_data.get('content', '') or
                    post_data.get('text', '') or
                    post_data.get('selftext', '') or
                    f"[{platform.title()} post]"
                )
        
        if not title or title == '':
            title = content[:100] if len(content) > 100 else content
        
        # Extract source_id
        source_id = (
            post_data.get('source_id') or
            post_data.get('id') or
            post_data.get('video_id') or
            str(hash(f"{title}{content[:100]}"))
        )
        
        content = content.strip() if content else ""
        
        # Validate minimum length
        if len(content) < 5:
            content = (
                post_data.get('summary', '') or
                post_data.get('description', '') or
                f"[{platform.title()} post - source_id: {source_id}]"
            )
        
        if not content or len(content.strip()) < 3:
            raise ValueError(
                f"Cannot create {platform} post without content: {source_id}"
            )
        
        # Extract published_at
        published_at = None
        if platform == 'reddit':
            created_utc = post_data.get('created_utc', 0)
            if created_utc:
                try:
                    published_at = datetime.fromtimestamp(created_utc, tz=timezoneDt.utc)
                except:
                    pass
        elif platform in ['twitter', 'youtube']:
            pub_at = post_data.get('published_at') or post_data.get('created_at')
            if pub_at:
                try:
                    if isinstance(pub_at, str):
                        published_at = datetime.fromisoformat(pub_at.replace('Z', '+00:00'))
                    elif isinstance(pub_at, datetime):
                        published_at = pub_at if pub_at.tzinfo else pub_at.replace(tzinfo=timezoneDt.utc)
                except:
                    pass
        
        if not published_at:
            published_at = now
        
        # Content hash
        content_for_hash = f"{title}{content[:500]}"
        content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()
        
        # ========================================================================
        # Build document with TYPE SAFETY (using self._safe_* methods)
        # ========================================================================
        document = {
            # REQUIRED fields (MongoDB schema)
            'source_id': str(source_id),
            'platform': platform,
            'content': content,
            'trust_score': self._safe_float(post_data.get('trust_score'), 0.0),  # FIX
            'status': self._safe_str(post_data.get('status', 'pending')),  # FIX
            'created_at': now,
            'updated_at': now,
            
            # Important fields
            'title': title,
            'content_hash': content_hash,
            'type': 'social',
            'url': self._safe_str(post_data.get('url')),  # FIX
            'author_username': self._safe_str(post_data.get('author_username', 'Unknown')),  # FIX
            'published_at': published_at,
            
            # Analysis results (with type safety)
            'credibility_analysis': self._safe_dict(post_data.get('credibility_analysis')),  # FIX
            'sentiment_analysis': self._safe_dict(post_data.get('sentiment_analysis')) or self._create_empty_sentiment(),  # Already safe
            'extracted_entities': self._safe_dict(post_data.get('extracted_entities')),  # FIX
            'text_processing': self._safe_dict(post_data.get('text_processing')),  # FIX
            'action_info': self._safe_dict(post_data.get('action_info')),  # FIX
            
            # Normalized metrics
            'engagement_metrics': self._safe_dict(post_data.get('engagement_metrics')),  # FIX
            'user_credibility': self._safe_dict(post_data.get('user_credibility')),  # FIX
            
            # Timestamps
            'processing_timestamp': post_data.get('processing_timestamp'),
            'analysis_timestamp': post_data.get('analysis_timestamp'),
            'fetched_at': self._parse_fetched_at(post_data.get('fetched_at')),
        }
        
        # ========================================================================
        # Platform-specific fields (NOW WITH TYPE SAFETY!)
        # ========================================================================
        if platform == 'reddit':
            document.update({
                'subreddit': self._safe_str(post_data.get('subreddit')),  # FIX
                'selftext': self._safe_str(post_data.get('selftext')),  # FIX
                'score': self._safe_int(post_data.get('score', 0)),  # Already safe
                'upvote_ratio': self._safe_float(post_data.get('upvote_ratio', 0.0)),  # FIX
                'num_comments': self._safe_int(post_data.get('num_comments', 0)),  # Already safe
                'total_awards_received': self._safe_int(post_data.get('total_awards_received', 0)),  # Already safe
                'author_info': self._safe_dict(post_data.get('author_info')),  # FIX
                'subreddit_info': self._safe_dict(post_data.get('subreddit_info')),  # FIX
            })
            
        elif platform == 'twitter':
            document.update({
                'text': self._safe_str(post_data.get('text')),  # FIX
                'public_metrics': self._safe_dict(post_data.get('public_metrics')),  # FIX
                'user_info': self._safe_dict(post_data.get('user_info')),  # FIX
                'entities': self._safe_dict(post_data.get('entities')),  # FIX
            })
            
        elif platform == 'youtube':
            document.update({
                'video_id': self._safe_str(post_data.get('video_id')),  # FIX
                'channel_id': self._safe_str(post_data.get('channel_id')),  # FIX
                'channel_title': self._safe_str(post_data.get('channel_title')),  # FIX
                'view_count': self._safe_int(post_data.get('view_count', 0)),  # Already safe
                'like_count': self._safe_int(post_data.get('like_count', 0)),  # Already safe
                'comment_count': self._safe_int(post_data.get('comment_count', 0)),  # Already safe
                'duration_seconds': self._safe_int(post_data.get('duration_seconds', 0)),  # Already safe
                'channel_info': self._safe_dict(post_data.get('channel_info')),  # FIX
            })
        
        # ========================================================================
        # VALIDATION (Same as news documents)
        # ========================================================================
        if not document.get('sentiment_analysis') or document['sentiment_analysis'] == {}:
            logger.warning(f"Empty sentiment_analysis for post {source_id}")
            document['sentiment_analysis'] = self._create_empty_sentiment()
        
        if not document.get('credibility_analysis') or document['credibility_analysis'] == {}:
            logger.warning(f"Empty credibility_analysis for post {source_id}")
        
        # Final validation
        assert isinstance(document['source_id'], str), "source_id must be string"
        assert isinstance(document['platform'], str), "platform must be string"
        assert isinstance(document['content'], str) and len(document['content']) >= 3, "content must be non-empty string"
        assert isinstance(document['trust_score'], (int, float)), "trust_score must be number"
        assert isinstance(document['status'], str), "status must be string"
        assert isinstance(document['created_at'], datetime), "created_at must be datetime"
        
        return document

    
    def _parse_fetched_at(self, fetched_at_value) -> Optional[datetime]:
        """Parse fetched_at timestamp safely"""
        if not fetched_at_value:
            return None
        
        try:
            if isinstance(fetched_at_value, str):
                return datetime.fromisoformat(fetched_at_value.replace('Z', '+00:00'))
            elif isinstance(fetched_at_value, datetime):
                return fetched_at_value if fetched_at_value.tzinfo else fetched_at_value.replace(tzinfo=timezoneDt.utc)
        except:
            pass
        
        return None


    
    def _extract_author_safe(self, post_data: Dict, platform: str) -> str:
        """
        Safely extract author from various field formats
        Handles all possible locations and formats for author data
        """
        author = ''
        
        # Priority order for direct fields
        direct_fields = ['author', 'username', 'channel_title', 'author_username']
        for field in direct_fields:
            value = post_data.get(field)
            if value and isinstance(value, str) and value.strip():
                author = value.strip()
                break
        
        # If not found, check nested structures
        if not author:
            # Reddit: author_info.name
            author_info = post_data.get('author_info')
            if isinstance(author_info, dict):
                author = (
                    author_info.get('name') or 
                    author_info.get('username') or
                    ''
                )
            
            # Twitter: user_info.username
            if not author:
                user_info = post_data.get('user_info')
                if isinstance(user_info, dict):
                    author = (
                        user_info.get('username') or 
                        user_info.get('name') or
                        ''
                    )
            
            # YouTube: channel_info.title or channel_info.channel_name
            if not author:
                channel_info = post_data.get('channel_info')
                if isinstance(channel_info, dict):
                    author = (
                        channel_info.get('title') or 
                        channel_info.get('channel_name') or
                        channel_info.get('channel_title') or
                        ''
                    )
        
        # Final cleanup and validation
        if author:
            author = str(author).strip()
            # Remove [deleted] and other invalid authors
            if author in ['[deleted]', '[removed]', 'None', '']:
                author = 'Unknown'
        else:
            author = 'Unknown'
        
        return author
    
    def _calculate_engagement_metrics(self, platform: str, post_data: Dict) -> Dict:

        normalized = {'total_engagement': 0, 'engagement_rate': 0}
        
        if platform == 'reddit':
            score = self._safe_int(post_data.get('score', 0))
            upvote_ratio = self._safe_float(post_data.get('upvote_ratio', 0.5))
            num_comments = self._safe_int(post_data.get('num_comments', 0))
            awards = self._safe_int(post_data.get('total_awards_received', 0))
            
            normalized.update({
                'score': score,
                'upvote_ratio': upvote_ratio,
                'num_comments': num_comments,
                'total_awards_received': awards,
            })
            normalized['total_engagement'] = score + (num_comments * 2) + (awards * 10)
            
        elif platform == 'twitter':
            public_metrics = post_data.get('public_metrics', {})
            likes = self._safe_int(public_metrics.get('like_count', 0))
            retweets = self._safe_int(public_metrics.get('retweet_count', 0))
            replies = self._safe_int(public_metrics.get('reply_count', 0))
            quotes = self._safe_int(public_metrics.get('quote_count', 0))
            
            normalized.update({
                'like_count': likes,
                'retweet_count': retweets,
                'reply_count': replies,
                'quote_count': quotes,
            })
            normalized['total_engagement'] = likes + (retweets * 2) + (replies * 3) + (quotes * 2)
            
            user_info = post_data.get('user_info', {})
            user_metrics = user_info.get('public_metrics', {})
            followers = self._safe_int(user_metrics.get('followers_count', 0))
            if followers > 0:
                normalized['engagement_rate'] = (normalized['total_engagement'] / followers) * 100
            
        elif platform == 'youtube':
            views = self._safe_int(post_data.get('view_count', 0))
            likes = self._safe_int(post_data.get('like_count', 0))
            comments = self._safe_int(post_data.get('comment_count', 0))
            duration = self._safe_int(post_data.get('duration_seconds', 0))
            
            normalized.update({
                'view_count': views,
                'like_count': likes,
                'comment_count': comments,
                'duration_seconds': duration,
            })
            normalized['total_engagement'] = likes + (comments * 3)
            
            if views > 0:
                normalized['engagement_rate'] = ((likes + comments) / views) * 100
        
        return normalized
    
    from typing import Dict, Optional, Any

    def _calculate_user_credibility(
            self, 
            platform: str, 
            post_data: Dict[str, Any]
        ) -> Dict[str, Any]:
        """
        Calculate user credibility metrics.
        
        Returns:
            Dict with fields:
            - exists: bool
            - followers: int
            - account_age_days: int
            - verified: bool
            - influence_level: str
        """
        
        # ============================================================================
        # DEBUG: Show what data we actually received
        # ============================================================================
        # import logging
        # logger = logging.getLogger(__name__)
        
        # logger.warning("=" * 80)
        # logger.warning(f"USER CRED DEBUG - Platform: {platform}")
        # logger.warning(f"   post_data type: {type(post_data)}")
        # logger.warning(f"   post_data keys: {list(post_data.keys())[:20]}")  # First 20 keys
        
        # # Check each possible location
        # has_user_info = 'user_info' in post_data
        # has_author_info = 'author_info' in post_data  
        # has_channel_info = 'channel_info' in post_data
        
        # logger.warning(f"   Has 'user_info': {has_user_info}")
        # logger.warning(f"   Has 'author_info': {has_author_info}")
        # logger.warning(f"   Has 'channel_info': {has_channel_info}")
        
        # # If found, show what's in it
        # if has_user_info:
        #     user_data = post_data.get('user_info')
        #     logger.warning(f"   user_info type: {type(user_data)}")
        #     if isinstance(user_data, dict):
        #         logger.warning(f"   user_info keys: {list(user_data.keys())}")
        #         # Check nested public_metrics for Twitter
        #         public_metrics = user_data.get('public_metrics', {})
        #         if isinstance(public_metrics, dict):
        #             logger.warning(f"   followers_count: {public_metrics.get('followers_count', 'NOT FOUND')}")
        #         else:
        #             logger.warning(f"   followers_count: NOT FOUND (no public_metrics)")
        #     else:
        #         logger.warning(f"   user_info is not a dict: {user_data}")
        
        # if has_author_info:
        #     author_data = post_data.get('author_info')
        #     logger.warning(f"   author_info type: {type(author_data)}")
        #     if isinstance(author_data, dict):
        #         logger.warning(f"   author_info keys: {list(author_data.keys())}")
        #         # Show both karma fields and computed total
        #         link_karma = author_data.get('link_karma', 0)
        #         comment_karma = author_data.get('comment_karma', 0)
        #         total_karma = author_data.get('total_karma', 0)
        #         computed_total = link_karma + comment_karma if total_karma == 0 else total_karma
        #         logger.warning(f"   link_karma: {link_karma}")
        #         logger.warning(f"   comment_karma: {comment_karma}")
        #         logger.warning(f"   total_karma (computed): {computed_total}")
        #     else:
        #         logger.warning(f"   author_info is not a dict: {author_data}")
        
        # if has_channel_info:
        #     channel_data = post_data.get('channel_info')
        #     logger.warning(f"   channel_info type: {type(channel_data)}")
        #     if isinstance(channel_data, dict):
        #         logger.warning(f"   channel_info keys: {list(channel_data.keys())}")
        #         logger.warning(f"   subscriber_count: {channel_data.get('subscriber_count', 'NOT FOUND')}")
        #     else:
        #         logger.warning(f"   channel_info is not a dict: {channel_data}")

        
        logger.warning("=" * 80)
        
        credibility = {
            'exists': True,
            'account_age_days': 0,
            'credibility_indicators': {}
        }
        
        now = timezone.now()
        
        try:
            if platform == 'reddit':
                author_info = post_data.get('author_info', {})
                
                if not author_info or author_info.get('unavailable'):
                    return {
                        'exists': False,
                        'account_age_days': 0,
                        'followers': 0,
                        'following': 0,
                        'post_count': 0,
                        'verified': False,
                        'influence_level': 'None',
                        'credibility_indicators': {}
                    }
                
                # Account age
                created_utc = author_info.get('created_utc')
                if created_utc:
                    try:
                        if isinstance(created_utc, str):
                            created_utc = float(created_utc)
                        if isinstance(created_utc, (int, float)) and created_utc > 0:
                            created_date = datetime.fromtimestamp(created_utc, tz=timezoneDt.utc)
                            credibility['account_age_days'] = (now - created_date).days
                    except (ValueError, TypeError, OSError):
                        pass
                
                # Karma as proxy for followers
                link_karma = self._safe_int(author_info.get('link_karma', 0))
                comment_karma = self._safe_int(author_info.get('comment_karma', 0))
                total_karma = self._safe_int(author_info.get('total_karma', 0))
                
                # Use total_karma if available, otherwise sum
                followers = total_karma if total_karma > 0 else link_karma + comment_karma
                
                credibility.update({
                    'followers': followers,  # ADDED for consistency
                    'following': 0,  # Reddit doesn't have following
                    'post_count': 0,  # Not directly available
                    'total_karma': followers,
                    'link_karma': link_karma,
                    'comment_karma': comment_karma,
                    'is_verified': bool(author_info.get('is_verified', False)),
                    'has_verified_email': bool(author_info.get('has_verified_email', False)),
                    'is_gold': bool(author_info.get('is_gold', False)),
                    'is_mod': bool(author_info.get('is_mod', False)),
                    'verified': bool(author_info.get('is_verified', False)) or bool(author_info.get('is_gold', False)),
                })
                
                # Influence level based on karma
                if followers > 100000:
                    credibility['influence_level'] = 'High'
                elif followers > 10000:
                    credibility['influence_level'] = 'Medium'
                elif followers > 1000:
                    credibility['influence_level'] = 'Low'
                else:
                    credibility['influence_level'] = 'None'
                
                subreddit_info = post_data.get('subreddit_info', {})
                if isinstance(subreddit_info, dict):
                    credibility['subreddit_subscribers'] = self._safe_int(
                        subreddit_info.get('subscribers', 0)
                    )
                
            elif platform == 'twitter':
                user_info = post_data.get('user_info', {})
                
                if not user_info:
                    credibility['exists'] = False
                    return credibility
                
                # Account age
                created_at = user_info.get('created_at', '')
                if created_at:
                    try:
                        if isinstance(created_at, str):
                            created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            credibility['account_age_days'] = (now - created_date).days
                    except (ValueError, TypeError):
                        pass
                
                # Twitter stores metrics in nested public_metrics
                user_metrics = user_info.get('public_metrics', {})
                if not isinstance(user_metrics, dict):
                    user_metrics = {}
                
                followers = self._safe_int(user_metrics.get('followers_count', 0))
                following = self._safe_int(user_metrics.get('following_count', 0))
                tweet_count = self._safe_int(user_metrics.get('tweet_count', 0))
                
                credibility.update({
                    'followers': followers,  # Now gets correct value
                    'following': following,
                    'post_count': tweet_count,
                    'followers_count': followers,
                    'following_count': following,
                    'tweet_count': tweet_count,
                    'listed_count': self._safe_int(user_metrics.get('listed_count', 0)),
                    'verified': bool(user_info.get('verified', False)),
                    'verified_type': str(user_info.get('verified_type', '')),
                })
                
                # Influence level
                if followers >= 500000:
                    credibility['influence_level'] = 'High'
                elif followers >= 50000:
                    credibility['influence_level'] = 'Medium'
                elif followers >= 5000:
                    credibility['influence_level'] = 'Low'
                else:
                    credibility['influence_level'] = 'None'
                
                if following > 0:
                    credibility['follower_ratio'] = followers / following
                
            elif platform == 'youtube':
                channel_info = post_data.get('channel_info', {})
                
                if not channel_info:
                    credibility['exists'] = False
                    return credibility
                
                channel_created = channel_info.get('channel_created', '')
                if channel_created:
                    try:
                        if isinstance(channel_created, str):
                            created_date = datetime.fromisoformat(channel_created.replace('Z', '+00:00'))
                            credibility['account_age_days'] = (now - created_date).days
                    except (ValueError, TypeError):
                        pass
                
                subscribers = self._safe_int(channel_info.get('subscriber_count', 0))
                video_count = self._safe_int(channel_info.get('video_count', 0))
                
                credibility.update({
                    'followers': subscribers,  # ADDED for consistency
                    'following': 0,  # Not applicable
                    'post_count': video_count,
                    'subscriber_count': subscribers,
                    'subscriber_count_hidden': bool(channel_info.get('subscriber_count_hidden', False)),
                    'total_view_count': self._safe_int(channel_info.get('total_view_count', 0)),
                    'video_count': video_count,
                    'verified': False,  # YouTube doesn't expose this in basic API
                })
                
                # Influence level
                if subscribers >= 1000000:
                    credibility['influence_level'] = 'High'
                elif subscribers >= 100000:
                    credibility['influence_level'] = 'Medium'
                elif subscribers >= 10000:
                    credibility['influence_level'] = 'Low'
                else:
                    credibility['influence_level'] = 'None'
            
            else:
                # Unknown platform - try generic fields
                credibility['exists'] = False
        
        except Exception as e:
            logger.warning(f"Error calculating user credibility for {platform}: {e}")
            import traceback
            traceback.print_exc()
            credibility['exists'] = False
        
        return credibility
    
    def _get_verification_candidate_pool(self) -> List[Dict]:
        """
        Get candidate pool for cross-verification
        Simply fetch recent high-quality content - no keyword filtering needed
        Verification engine will handle similarity matching
        """
        try:
            hours_back = self.config.get('cross_check_window_hours', 24)
            
            # Get recent articles (trust_score >= 6.0 for quality)
            recent_articles = self.mongo_manager.get_recent_articles(
                hours_back=hours_back,
                trust_score_threshold=6.0,
                limit=500  # Large pool for semantic matching
            )
            
            # Get recent social posts (for diverse perspectives)
            recent_posts = self.mongo_manager.get_recent_social_posts(
                hours_back=hours_back,
                trust_score_threshold=6.0,
                limit=200
            )
            
            candidate_pool = recent_articles + recent_posts
            
            logger.info(f"Fetched {len(candidate_pool)} candidates for verification pool")
            return candidate_pool
            
        except Exception as e:
            logger.error(f"Error fetching candidate pool: {e}")
            return []
        
    def process_news_batch(self, news_articles: List[Dict], 
                  source_name: str = "unknown") -> ProcessingResult:
        """
        Process a batch of news articles through the complete pipeline
        With comprehensive logging to debug storage issues
        """
        start_time = datetime.now()
        logger.info(f"START: Processing {len(news_articles)} articles from {source_name}")
        
        stats = {
            'approved': 0,
            'pending': 0,
            'flagged': 0,
            'errors': 0,
            'trust_scores': []
        }
        
        # Get cross-verification data using topic modeler
        candidate_pool = self._get_verification_candidate_pool()
        
        processed_articles = []
        high_credibility_article_ids = []
        
        for i in range(0, len(news_articles), self.config['batch_size']):
            batch = news_articles[i:i + self.config['batch_size']]
            batch_num = i//self.config['batch_size'] + 1
            
            try:
                logger.info(f"Processing batch {batch_num} ({len(batch)} articles)...")
                batch_results = self._process_article_batch(batch, candidate_pool)
                logger.info(f"Batch {batch_num} analysis complete: {len(batch_results)} results")
                
                for article, trust_score in batch_results:
                    # Determine final status
                    action_info = self.credibility_engine.determine_content_action(trust_score)
                    
                    if action_info['action'] == 'auto_approve':
                        article['status'] = 'approved'
                        stats['approved'] += 1
                    elif action_info['action'] == 'manual_review':
                        article['status'] = 'flagged'
                        stats['flagged'] += 1
                    else:
                        article['status'] = 'pending'
                        stats['pending'] += 1
                    
                    article.update({
                        'trust_score': trust_score.final_score,
                        'processing_timestamp': timezone.now().isoformat(),
                        'source_batch': source_name,
                        'action_info': action_info
                    })
                    
                    processed_articles.append(article)
                    stats['trust_scores'].append(trust_score.final_score)
                    
                    if trust_score.final_score >= 7.0:
                        high_credibility_article_ids.append(article.get('source_id'))
                
                logger.info(f"Batch {batch_num} complete: {len(batch_results)} articles processed")
                
            except Exception as e:
                logger.error(f"ERROR in batch {batch_num}: {e}")
                import traceback
                traceback.print_exc()
                stats['errors'] += len(batch)
        
        logger.info(f"ANALYSIS COMPLETE: {len(processed_articles)} articles processed, {stats['errors']} errors")
        
        # ===================================================================
        # CORRECTED STORAGE SECTION WITH COMPREHENSIVE LOGGING
        # ===================================================================
        if processed_articles:
            try:
                logger.info(f"STORAGE: Starting - {len(processed_articles)} articles to store")
                logger.debug(f"STORAGE: First 3 articles: {[a.get('source_id') for a in processed_articles[:3]]}")

                prepared_documents = []
                preparation_errors = 0

                for i, article in enumerate(processed_articles):
                    try:
                        source_id = article.get('source_id', f'unknown_{i}')
                        logger.debug(f"  Preparing {i+1}/{len(processed_articles)}: {source_id}")
                        logger.debug(f"  Article keys: {list(article.keys())}")

                        prepared_doc = self._prepare_news_document(article)

                        missing_fields = []
                        if not prepared_doc.get('source_id'):
                            missing_fields.append('source_id')
                        if not prepared_doc.get('title'):
                            missing_fields.append('title')
                        if not prepared_doc.get('platform'):
                            missing_fields.append('platform')
                        if not prepared_doc.get('trust_score'):
                            missing_fields.append('trust_score')

                        if missing_fields:
                            logger.error(f"  Prepared doc missing fields: {missing_fields}")
                            logger.error(f"     Original article keys: {list(article.keys())[:10]}")
                            logger.error(f"     Prepared doc keys: {list(prepared_doc.keys())[:10]}")
                            logger.error(f"     Prepared doc: {prepared_doc}")
                            preparation_errors += 1
                            continue

                        sentiment = prepared_doc.get('sentiment_analysis', {})
                        if not sentiment or sentiment == {}:
                            logger.warning(f"  ️  Empty sentiment_analysis for {source_id}")

                        prepared_documents.append(prepared_doc)
                        logger.debug(f"  Prepared: {source_id}")

                    except Exception as e:
                        logger.error(f"  Error preparing article {i+1}: {e}")
                        import traceback
                        traceback.print_exc()
                        logger.error(f"  Article data: {article}")
                        preparation_errors += 1

                logger.info(f"PREPARATION COMPLETE: {len(prepared_documents)} docs ready, {preparation_errors} errors")
                logger.debug(f"PREPARED DOCS: First 3: {[d.get('source_id') for d in prepared_documents[:3]]}")

                if not prepared_documents:
                    logger.error("STORAGE ABORTED: No documents successfully prepared!")
                    stats['errors'] += len(processed_articles)
                else:
                    logger.info(f"CALLING MongoDB bulk_insert_articles with {len(prepared_documents)} documents...")
                    logger.debug(f"Bulk insert doc source_ids: {[d.get('source_id') for d in prepared_documents]}")

                    storage_stats = self.mongo_manager.bulk_insert_articles(prepared_documents)

                    logger.info(f"STORAGE RESULT: {storage_stats}")
                    logger.debug(f"STORAGE RESULT DETAILS: {storage_stats}")

                    # Enhanced duplicate debugging
                    if storage_stats['inserted'] == 0:
                        logger.error(f"STORAGE FAILED: 0 documents inserted!")
                        logger.error(f"   Errors: {storage_stats['errors']}, Duplicates: {storage_stats['duplicates']}")
                        logger.error(f"   Duplicate source_ids: {storage_stats.get('duplicate_ids', [])}")
                        if storage_stats.get('duplicate_ids'):
                            logger.error(f"   Duplicate IDs details: {storage_stats['duplicate_ids']}")
                        # Log all source_ids for comparison
                        logger.debug(f"   All attempted source_ids: {[d.get('source_id') for d in prepared_documents]}")
                        if storage_stats['errors'] > 0:
                            logger.error("   MongoDB validation is rejecting documents")
                            logger.error("   Check mongo_manager.bulk_insert_articles validation logic")
                    else:
                        logger.info(f"STORAGE SUCCESS: {storage_stats['inserted']} documents inserted")

                    if high_credibility_article_ids:
                        logger.info(f"Triggering RAG indexing for {len(high_credibility_article_ids)} high-credibility articles")
                        logger.debug(f"RAG article IDs: {high_credibility_article_ids}")
                        self._trigger_rag_indexing(high_credibility_article_ids, 'news')

            except Exception as e:
                logger.error(f"STORAGE EXCEPTION: {e}")
                import traceback
                traceback.print_exc()
                stats['errors'] += len(processed_articles)
        else:
            logger.warning(f"STORAGE SKIPPED: processed_articles list is empty")
            logger.warning(f"   This means all batches failed or returned no results")
            logger.warning(f"   Check error logs above for batch processing failures")
        
        # Calculate final stats
        processing_time = (datetime.now() - start_time).total_seconds()
        avg_trust_score = sum(stats['trust_scores']) / len(stats['trust_scores']) if stats['trust_scores'] else 0
        
        result = ProcessingResult(
            total_processed=len(news_articles),
            approved=stats['approved'],
            pending=stats['pending'],
            flagged=stats['flagged'],
            errors=stats['errors'],
            average_trust_score=avg_trust_score,
            processing_time_seconds=processing_time
        )
        
        logger.info(f"COMPLETE: {source_name} - {result}")
        return result


    
    def process_social_posts_batch(self, social_posts: List[Dict], 
                       platform: str = "unknown") -> ProcessingResult:
        """
        Process social media posts batch
        Prepares documents before calling bulk_insert
        """
        start_time = datetime.now()
        logger.info(f"Processing {len(social_posts)} posts from {platform}")
        
        stats = {
            'approved': 0,
            'pending': 0,
            'flagged': 0,
            'errors': 0,
            'trust_scores': []
        }
        
        processed_posts = []
        high_credibility_post_ids = []
        
        for i, post in enumerate(social_posts):
            try:
                post['type'] = 'social'
                post['platform'] = platform
                
                # Step 0: Calculate metrics BEFORE document preparation
                engagement_metrics = self._calculate_engagement_metrics(platform, post)
                user_credibility = self._calculate_user_credibility(platform, post)
                
                # Add calculated metrics to post data
                post['engagement_metrics'] = engagement_metrics
                post['user_credibility'] = user_credibility
                
                if not user_credibility.get('exists', True):
                    logger.warning(f"No user credibility data for {platform} post {post.get('source_id')}")
                
                # Extract author safely
                post['author_username'] = self._extract_author_safe(post, platform)
                
                # Step 1: Text Processing
                text_content = f"{post.get('title', '')} {post.get('content', post.get('text', ''))}"
                
                if text_content.strip():
                    processed_text = self.text_processor.preprocess(text_content)
                    entities = self.text_processor.extract_entities(text_content)
                    
                    post['extracted_entities'] = {
                        'cryptocurrencies': entities.cryptocurrencies,
                        'exchanges': entities.exchanges,
                        'persons': entities.persons,
                        'organizations': entities.organizations,
                        'money_amounts': entities.money_amounts,
                        'dates': entities.dates,
                        'locations': entities.locations
                    }
                    
                    post['text_processing'] = {
                        'language': processed_text.language,
                        'language_confidence': processed_text.language_confidence,
                        'is_english': processed_text.is_english,
                        'word_count': processed_text.word_count,
                        'hashtags': processed_text.hashtags,
                        'mentions': processed_text.mentions
                    }
                    
                    # Step 2: Sentiment Analysis
                    sentiment_result = self.sentiment_analyzer.analyze(text_content)
                    
                    post['sentiment_analysis'] = {
                        'label': sentiment_result.label.value,
                        'score': sentiment_result.score,
                        'confidence': sentiment_result.confidence,
                        'bullish_probability': sentiment_result.bullish_probability,
                        'bearish_probability': sentiment_result.bearish_probability,
                        'neutral_probability': sentiment_result.neutral_probability,
                        'emotions': sentiment_result.emotions,
                        'flags': sentiment_result.flags,
                        'predicted_market_impact': sentiment_result.predicted_market_impact,
                        'impact_confidence': sentiment_result.impact_confidence,
                        'aspect_sentiments': sentiment_result.aspect_sentiments,
                        'model_scores': sentiment_result.model_scores
                    }
                else:
                    post['sentiment_analysis'] = self._create_empty_sentiment()
                
                # Step 3: Credibility Scoring (uses pre-calculated metrics)
                trust_score = self.credibility_engine.calculate_trust_score(
                    content_data=post,
                    sentiment_data=post.get('sentiment_analysis'),
                    entities_data=post.get('extracted_entities'),
                    verification_result=None  
                )
                
                action_info = self.credibility_engine.determine_content_action(trust_score)
                
                # Build credibility analysis
                post['credibility_analysis'] = {
                    'source_score': trust_score.source_score,
                    'content_score': trust_score.content_score,
                    'engagement_score': trust_score.engagement_score,
                    'cross_check_score': trust_score.cross_check_score,
                    'source_history_score': trust_score.source_history_score,
                    'recency_score': trust_score.recency_score,
                    'confidence': trust_score.confidence,
                    'flags': trust_score.flags,
                    'reasoning': trust_score.reasoning,
                    'recommended_action': action_info
                }
                
                stats['trust_scores'].append(trust_score.final_score)
                
                # Determine status
                if action_info['action'] == 'auto_approve':
                    post['status'] = 'approved'
                    stats['approved'] += 1
                elif action_info['action'] == 'manual_review':
                    post['status'] = 'flagged'
                    stats['flagged'] += 1
                else:
                    post['status'] = 'pending'
                    stats['pending'] += 1
                
                # Add final metadata
                post.update({
                    'trust_score': trust_score.final_score,
                    'processing_timestamp': timezone.now().isoformat(),
                    'platform_batch': platform,
                    'action_info': action_info
                })
                
                # Add to processed list
                processed_posts.append(post)
                
                if trust_score.final_score >= 7.0:
                    high_credibility_post_ids.append(post.get('source_id') or post.get('id'))
                
            except Exception as e:
                logger.error(f"Error processing post {i}: {e}")
                import traceback
                traceback.print_exc()
                stats['errors'] += 1
        
        # ===================================================================
        # CORRECTED STORAGE SECTION
        # ===================================================================
        if processed_posts:
            try:
                logger.info(f"SOCIAL STORAGE: Preparing {len(processed_posts)} posts for storage")
                logger.debug(f"SOCIAL STORAGE: First 3 post IDs: {[p.get('source_id') for p in processed_posts[:3]]}")

                prepared_documents = []
                for post in processed_posts:
                    prepared_doc = self._prepare_social_document(post)
                    logger.debug(f"  Prepared social doc: {prepared_doc.get('source_id')}, platform: {prepared_doc.get('platform')}")
                    prepared_documents.append(prepared_doc)

                logger.info(f"SOCIAL STORAGE: Prepared {len(prepared_documents)} documents")
                logger.debug(f"SOCIAL STORAGE: Bulk insert doc IDs: {[d.get('source_id') for d in prepared_documents]}")

                storage_stats = self.mongo_manager.bulk_insert_social_posts(prepared_documents)
                logger.info(f"Social posts storage results for {platform}: {storage_stats}")
                logger.debug(f"Social posts storage details: {storage_stats}")

                # Enhanced duplicate debugging
                if storage_stats['inserted'] == 0:
                    logger.error(f"SOCIAL STORAGE FAILED: 0 documents inserted!")
                    logger.error(f"   Errors: {storage_stats['errors']}, Duplicates: {storage_stats['duplicates']}")
                    logger.error(f"   Duplicate source_ids: {storage_stats.get('duplicate_ids', [])}")
                    if storage_stats.get('duplicate_ids'):
                        logger.error(f"   Duplicate IDs details: {storage_stats['duplicate_ids']}")
                    logger.debug(f"   All attempted source_ids: {[d.get('source_id') for d in prepared_documents]}")

                if high_credibility_post_ids:
                    logger.info(f"Triggering RAG indexing for {len(high_credibility_post_ids)} high-credibility social posts")
                    logger.debug(f"RAG social post IDs: {high_credibility_post_ids}")
                    self._trigger_rag_indexing(high_credibility_post_ids, 'social')

            except Exception as e:
                logger.error(f"Error storing posts from {platform}: {e}")
                import traceback
                traceback.print_exc()
                stats['errors'] += len(processed_posts)
        
        # Calculate final stats
        processing_time = (datetime.now() - start_time).total_seconds()
        avg_trust_score = sum(stats['trust_scores']) / len(stats['trust_scores']) if stats['trust_scores'] else 0
        
        result = ProcessingResult(
            total_processed=len(social_posts),
            approved=stats['approved'],
            pending=stats['pending'],
            flagged=stats['flagged'],
            errors=stats['errors'],
            average_trust_score=avg_trust_score,
            processing_time_seconds=processing_time
        )
        
        logger.info(f"Completed processing {platform}: {result}")
        return result

    
    
    def _process_article_batch(self, articles: List[Dict], 
                      candidate_pool: List[Dict]) -> List[Tuple[Dict, TrustScore]]:
        """Process a batch of articles - extract features and calculate trust scores"""
        results = []
        
        for article in articles:
            try:
                # Step 1: Text Processing
                text_content = f"{article.get('title', '')} {article.get('description', article.get('content', ''))}"
                
                if text_content.strip():
                    try:
                        # CRITICAL: Ensure text processor is working
                        processed_text = self.text_processor.preprocess(text_content)
                        entities = self.text_processor.extract_entities(text_content)
                        
                        article['extracted_entities'] = {
                            'cryptocurrencies': entities.cryptocurrencies,
                            'exchanges': entities.exchanges,
                            'persons': entities.persons,
                            'organizations': entities.organizations,
                            'money_amounts': entities.money_amounts,
                            'dates': entities.dates,
                            'locations': entities.locations
                        }
                        
                        article['text_processing'] = {
                            'language': processed_text.language,
                            'language_confidence': processed_text.language_confidence,
                            'is_english': processed_text.is_english,
                            'word_count': processed_text.word_count,
                            'hashtags': processed_text.hashtags,
                            'mentions': processed_text.mentions
                        }
                        
                        logger.debug(f"Text processing complete for article: {article.get('source_id')}")
                        
                    except Exception as e:
                        logger.error(f"Text processing failed: {e}")
                        import traceback
                        traceback.print_exc()
                        article['extracted_entities'] = {}
                        article['text_processing'] = {}
                    
                    # Step 2: Sentiment Analysis - CRITICAL SECTION
                    try:
                        logger.debug(f"Starting sentiment analysis for: {article.get('source_id')}")
                        sentiment_result = self.sentiment_analyzer.analyze(text_content)
                        
                        # CRITICAL: Verify sentiment_result is valid
                        if not sentiment_result:
                            logger.error(f"Sentiment analyzer returned None for: {article.get('source_id')}")
                            article['sentiment_analysis'] = self._create_empty_sentiment()
                        else:
                            article['sentiment_analysis'] = {
                                'label': sentiment_result.label.value if hasattr(sentiment_result.label, 'value') else str(sentiment_result.label),
                                'score': sentiment_result.score,
                                'confidence': sentiment_result.confidence,
                                'bullish_probability': sentiment_result.bullish_probability,
                                'bearish_probability': sentiment_result.bearish_probability,
                                'neutral_probability': sentiment_result.neutral_probability,
                                'emotions': sentiment_result.emotions,
                                'flags': sentiment_result.flags,
                                'predicted_market_impact': sentiment_result.predicted_market_impact,
                                'impact_confidence': sentiment_result.impact_confidence,
                                'aspect_sentiments': sentiment_result.aspect_sentiments,
                                'model_scores': sentiment_result.model_scores
                            }
                            logger.debug(f"Sentiment analysis SUCCESS: {article['sentiment_analysis']['label']}")
                            
                    except Exception as e:
                        logger.error(f"Sentiment analysis FAILED for {article.get('source_id')}: {e}")
                        import traceback
                        traceback.print_exc()
                        article['sentiment_analysis'] = self._create_empty_sentiment()
                else:
                    logger.warning(f"No text content for article: {article.get('source_id')}")
                    article['sentiment_analysis'] = self._create_empty_sentiment()
                    article['extracted_entities'] = {}
                    article['text_processing'] = {}
                
                # VERIFICATION: Check sentiment_analysis is not empty
                if not article.get('sentiment_analysis') or article.get('sentiment_analysis') == {}:
                    logger.error(f"EMPTY sentiment_analysis for {article.get('source_id')}")
                    article['sentiment_analysis'] = self._create_empty_sentiment()
                else:
                    logger.debug(f"sentiment_analysis populated: {len(article['sentiment_analysis'])} keys")
                
                verification_result: VerificationResult = None
                if candidate_pool:
                    try:
                        verification_result = self.verification_engine.verify_content(
                            content_data=article,
                            candidate_pool=candidate_pool
                        )
                        
                        # NEW: Store verification details in article
                        article['verification_details'] = {
                            'corroboration_score': verification_result.corroboration_score,
                            'total_references': verification_result.total_references,
                            'unique_sources': verification_result.unique_sources,
                            'avg_similarity': verification_result.avg_similarity,
                            'source_diversity_score': verification_result.source_diversity_score,
                            'temporal_clustering_score': verification_result.temporal_clustering_score,
                            'confidence': verification_result.confidence,
                            'verified_claims': verification_result.verified_claims[:5],  # Top 5
                            'contradicted_claims': verification_result.contradicted_claims[:5],
                            'flags': verification_result.flags,
                            'reasoning': verification_result.reasoning,
                            'verified_at': timezone.now().isoformat()
                        }
                        
                    except Exception as e:
                        logger.error(f"Verification failed: {e}")
                        verification_result = None
                        article['verification_details'] = None

                # Step 3: Credibility Scoring
                try:
                    trust_score = self.credibility_engine.calculate_trust_score(
                        content_data=article,
                        sentiment_data=article.get('sentiment_analysis'),
                        entities_data=article.get('extracted_entities'),
                        verification_result=verification_result
                    )
                    
                    action_info = self.credibility_engine.determine_content_action(trust_score)
                    
                    # Build credibility analysis
                    article['credibility_analysis'] = {
                        'source_score': trust_score.source_score,
                        'content_score': trust_score.content_score,
                        'engagement_score': trust_score.engagement_score,
                        'cross_check_score': trust_score.cross_check_score,
                        'source_history_score': trust_score.source_history_score,
                        'recency_score': trust_score.recency_score,
                        'confidence': trust_score.confidence,
                        'flags': trust_score.flags,
                        'reasoning': trust_score.reasoning,
                        'cross_reference_matches': trust_score.cross_reference_matches,
                        'corroboration_sources': trust_score.corroboration_sources,
                        'recommended_action': action_info
                    }
                    
                    # VERIFICATION: Check credibility_analysis is not empty
                    if not article.get('credibility_analysis') or article.get('credibility_analysis') == {}:
                        logger.error(f"EMPTY credibility_analysis for {article.get('source_id')}")
                    else:
                        logger.debug(f"credibility_analysis populated: {len(article['credibility_analysis'])} keys")
                    
                    results.append((article, trust_score))
                    
                except Exception as e:
                    logger.error(f"Credibility scoring failed for {article.get('source_id')}: {e}")
                    import traceback
                    traceback.print_exc()
                    fallback_score = self.credibility_engine._create_fallback_score()
                    article['credibility_analysis'] = {}
                    results.append((article, fallback_score))
                
            except Exception as e:
                logger.error(f"Error processing individual article {article.get('source_id', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
                fallback_score = self.credibility_engine._create_fallback_score()
                # Ensure minimum fields exist
                if 'sentiment_analysis' not in article:
                    article['sentiment_analysis'] = self._create_empty_sentiment()
                if 'credibility_analysis' not in article:
                    article['credibility_analysis'] = {}
                if 'extracted_entities' not in article:
                    article['extracted_entities'] = {}
                if 'text_processing' not in article:
                    article['text_processing'] = {}
                results.append((article, fallback_score))
        
        return results
    
    def _create_empty_sentiment(self) -> Dict:
        """Create empty sentiment structure for content without text"""
        return {
            'label': 'neutral',
            'score': 0.0,
            'confidence': 0.0,
            'bullish_probability': 0.33,
            'bearish_probability': 0.33,
            'neutral_probability': 0.34,
            'emotions': {},
            'flags': ['insufficient_content'],
            'predicted_market_impact': 'none',
            'impact_confidence': 0.0,
            'aspect_sentiments': {},
            'model_scores': {}
        }
    
    def _trigger_rag_indexing(self, content_ids: List[str], content_type: str = 'news') -> None:
        """Trigger event-driven RAG indexing for high-credibility content"""
        if not content_ids:
            return
        
        try:
            from myapp.tasks.rag_tasks import event_driven_index
            event_driven_index.delay(content_ids, content_type)
            logger.info(f"Triggered RAG indexing for {len(content_ids)} high-credibility {content_type} items")
        except ImportError:
            logger.warning("RAG tasks not available, skipping event-driven indexing")
        except Exception as e:
            logger.error(f"Error triggering RAG indexing: {e}")
    
    # Management methods (unchanged but included for completeness)
    def get_curated_feed(self, trust_score_threshold: float = 7.0, limit: int = 50,
                        hours_back: int = 24, platforms: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get curated news feed with high-credibility content"""
        try:
            articles = self.mongo_manager.get_high_credibility_articles(
                trust_score_threshold=trust_score_threshold,
                limit=limit // 2,
                hours_back=hours_back,
                platform=None
            )
            
            all_content = []
            for article in articles:
                sentiment_data = article.get('sentiment_analysis', {})
                sentiment = (
                    sentiment_data.get('label') or
                    sentiment_data.get('sentiment_label') or
                    'neutral'
                )
                all_content.append({
                    'type': 'article',
                    'content': article,
                    'trust_score': article.get('trust_score', 0),
                    'created_at': article.get('created_at'),
                    'platform': article.get('platform'),
                    'sentiment': sentiment
                })
            
            all_content.sort(
                key=lambda x: (x['trust_score'], x['created_at'] or datetime.min.replace(tzinfo=timezoneDt.utc)),
                reverse=True
            )
            
            if platforms:
                all_content = [item for item in all_content if item['platform'] in platforms]
            
            curated_content = all_content[:limit]
            
            feed_metadata = {
                'generated_at': timezone.now().isoformat(),
                'trust_score_threshold': trust_score_threshold,
                'total_items': len(curated_content),
                'time_window_hours': hours_back,
                'platforms_included': platforms or 'all',
                'average_trust_score': sum(item['trust_score'] for item in curated_content) / len(curated_content) if curated_content else 0,
                'content_breakdown': {
                    'articles': sum(1 for item in curated_content if item['type'] == 'article'),
                    'social_posts': sum(1 for item in curated_content if item['type'] == 'social_post')
                }
            }
            
            return {'feed': curated_content, 'metadata': feed_metadata}
            
        except Exception as e:
            logger.error(f"Error generating curated feed: {e}")
            return {
                'feed': [],
                'metadata': {'error': str(e)},
                'generated_at': timezone.now().isoformat()
            }
    
    def get_pending_review_queue(self, priority: str = 'all', limit: int = 50) -> Dict[str, Any]:
        """Get content pending manual review"""
        try:
            pending_content = self.mongo_manager.get_pending_content_for_review(limit=limit)
            
            review_metadata = {
                'generated_at': timezone.now().isoformat(),
                'total_pending': pending_content['total'],
                'priority_filter': priority,
                'articles_pending': len(pending_content['articles']),
                'posts_pending': len(pending_content['posts'])
            }
            
            return {'review_queue': pending_content, 'metadata': review_metadata}
            
        except Exception as e:
            logger.error(f"Error getting review queue: {e}")
            return {
                'review_queue': {'articles': [], 'posts': [], 'total': 0},
                'metadata': {'error': str(e)}
            }
    
    def update_content_status(self, content_id: str, new_status: str, 
                            content_type: str = 'news_articles',
                            reviewer_notes: Optional[str] = None) -> bool:
        """Update content status with feedback tracking"""
        try:
            success = self.mongo_manager.update_content_status(content_id, new_status, content_type)
            
            if success:
                self.threshold_manager.record_performance(
                    content_id=content_id,
                    predicted_action='pending',
                    actual_outcome=new_status,
                    user_feedback=reviewer_notes
                )
                logger.info(f"Updated {content_type} {content_id} status to {new_status}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating content status: {e}")
            return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health and performance metrics"""
        try:
            mongo_stats = self.mongo_manager.get_statistics()
            performance_analysis = self.threshold_manager.analyze_performance()
            
            now = timezone.now()
            
            health_info = {
                'timestamp': now.isoformat(),
                'database': {
                    'status': 'healthy' if mongo_stats else 'error',
                    'statistics': mongo_stats
                },
                'credibility_engine': {
                    'status': 'healthy',
                    'news_weights': self.credibility_engine.news_weights,
                    'social_weights': self.credibility_engine.social_weights,
                    'thresholds': self.credibility_engine.trust_thresholds
                },
                'performance': performance_analysis,
                'system_load': {
                    'cache_hit_rate': 0.85,  # Placeholder
                    'avg_processing_time': 2.5  # Placeholder
                }
            }
            
            return health_info
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                'timestamp': timezone.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    def optimize_thresholds(self) -> Dict[str, Any]:
        """Run threshold optimization based on performance data"""
        try:
            performance = self.threshold_manager.analyze_performance()
            
            if performance['status'] != 'analysis_complete':
                return {
                    'status': 'insufficient_data',
                    'message': 'Need more performance data for optimization'
                }
            
            adjusted = self.threshold_manager.auto_adjust_thresholds(max_adjustment=0.3)
            
            result = {
                'status': 'completed',
                'thresholds_adjusted': adjusted,
                'performance_analysis': performance,
                'new_thresholds': self.credibility_engine.trust_thresholds if adjusted else None
            }
            
            if adjusted:
                logger.info("Thresholds automatically adjusted based on performance")
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing thresholds: {e}")
            return {'status': 'error', 'error': str(e)}


# Conversion functions (unchanged)
def convert_fetcher_to_service_format(articles: List[Dict], source_name: str) -> List[Dict]:
    """Convert RAW fetcher output format to integrator service expected format"""
    formatted_articles = []
    
    for article in articles:
        try:
            source_id = (
                article.get('id') or 
                article.get('guid') or 
                article.get('source_id') or
                str(hash(f"{article.get('title', '')}{article.get('url', '')}"))
            )
            
            raw_source = article.get('source', {})
            if isinstance(raw_source, dict):
                source_info = raw_source
            elif isinstance(raw_source, str):
                source_info = {'title': raw_source, 'name': raw_source, 'domain': ''}
            else:
                source_info = {'title': 'Unknown', 'name': 'Unknown', 'domain': ''}
            
            description = (
                article.get('description') or 
                article.get('body') or 
                article.get('summary') or
                article.get('content') or ''
            )
            
            content = (
                article.get('content') or 
                article.get('body') or 
                article.get('description') or
                article.get('summary') or ''
            )
            
            formatted_article = {
                'source_id': str(source_id),
                'platform': map_source_to_platform(source_name),
                'type': 'news',
                'title': article.get('title', ''),
                'description': description,
                'content': content,
                'url': article.get('url') or article.get('link', ''),
                'author': article.get('author') or source_info.get('name', 'Unknown'),
                'published_at': (
                    article.get('published_at') or 
                    article.get('publishedAt') or 
                    article.get('published') or
                    article.get('published_on') or
                    timezone.now().isoformat()
                ),
                'created_at': timezone.now(),
                'updated_at': timezone.now(),
                'status': 'pending',
                'source': source_info,
                'votes': article.get('votes', {}),
                'instruments': article.get('instruments', []),
                'kind': article.get('kind', ''),
                'upvotes': article.get('upvotes', 0),
                'downvotes': article.get('downvotes', 0),
                'tags': article.get('tags', ''),
                'categories': article.get('categories', ''),
                'references': article.get('references', []),
                'fetcher_metadata': {
                    'source_name': source_name,
                    'fetched_at': timezone.now().isoformat(),
                    'raw_id': article.get('id'),
                }
            }
            
            preserve_fields = ['image', 'thumbnail', 'media', 'language', 'region', 'imageurl', 'urlToImage']
            for field in preserve_fields:
                if field in article:
                    formatted_article[field] = article[field]
            
            formatted_articles.append(formatted_article)
             
        except Exception as e:
            logger.error(f"Error formatting article from {source_name}: {e}")
            continue
     
    return formatted_articles


def convert_social_fetcher_to_service_format(posts: List[Dict], platform_name: str) -> List[Dict]:
    """Convert RAW social media fetcher output to service format"""
    formatted_posts = []
    
    for post in posts:
        try:
            post_id = (
                post.get('id') or 
                post.get('video_id') or 
                post.get('source_id') or
                str(hash(f"{post.get('title', '')}{post.get('url', '')}"))
            )
            
            formatted_post = {
                'source_id': str(post_id),
                'platform': platform_name.lower(),
                'type': 'social',
                'title': post.get('title') or (post.get('text', '')[:100] if post.get('text') else ''),
                'content': post.get('content') or post.get('text') or post.get('selftext', ''),
                'text': post.get('text') or post.get('content') or post.get('selftext', ''),
                'url': post.get('url') or post.get('permalink', ''),
                'author': post.get('author') or post.get('username') or post.get('channel_title', 'Unknown'),
                'published_at': post.get('created_at') or post.get('published_at'),
                'created_utc': post.get('created_utc'),
                'created_at': timezone.now(),
                'updated_at': timezone.now(),
                'status': 'pending',
                'fetcher_metadata': {
                    'platform': platform_name,
                    'fetched_at': timezone.now().isoformat(),
                }
            }
            
            if platform_name.lower() == 'reddit':
                formatted_post.update(_extract_reddit_raw_fields(post))
            elif platform_name.lower() == 'twitter':
                formatted_post.update(_extract_twitter_raw_fields(post))
            elif platform_name.lower() == 'youtube':
                formatted_post.update(_extract_youtube_raw_fields(post))
            
            formatted_posts.append(formatted_post)
            
        except Exception as e:
            logger.error(f"Error formatting post from {platform_name}: {e}")
            continue
    
    return formatted_posts


def _extract_reddit_raw_fields(post: Dict) -> Dict:
    """Extract and preserve RAW Reddit fields"""
    return {
        'score': post.get('score', 0),
        'upvote_ratio': post.get('upvote_ratio', 0.5),
        'num_comments': post.get('num_comments', 0),
        'total_awards_received': post.get('total_awards_received', 0),
        'author_info': post.get('author_info', {}),
        'subreddit': post.get('subreddit', ''),
        'subreddit_info': post.get('subreddit_info', {}),
        'selftext': post.get('selftext', ''),
        'is_self': post.get('is_self', False),
        'over_18': post.get('over_18', False),
        'spoiler': post.get('spoiler', False),
        'stickied': post.get('stickied', False),
        'locked': post.get('locked', False),
        'permalink': post.get('permalink', ''),
    }


def _extract_twitter_raw_fields(post: Dict) -> Dict:
    """Extract and preserve RAW Twitter fields"""
    return {
        'public_metrics': post.get('public_metrics', {}),
        'like_count': post.get('like_count', post.get('public_metrics', {}).get('like_count', 0)),
        'retweet_count': post.get('retweet_count', post.get('public_metrics', {}).get('retweet_count', 0)),
        'reply_count': post.get('reply_count', post.get('public_metrics', {}).get('reply_count', 0)),
        'quote_count': post.get('quote_count', post.get('public_metrics', {}).get('quote_count', 0)),
        'user_info': post.get('user_info', post.get('author_info', {})),
        'conversation_id': post.get('conversation_id'),
        'in_reply_to_user_id': post.get('in_reply_to_user_id'),
        'referenced_tweets': post.get('referenced_tweets', []),
        'entities': post.get('entities', {}),
        'context_annotations': post.get('context_annotations', []),
    }


def _extract_youtube_raw_fields(post: Dict) -> Dict:
    """Extract and preserve RAW YouTube fields"""
    return {
        'view_count': post.get('view_count', 0),
        'like_count': post.get('like_count', 0),
        'comment_count': post.get('comment_count', 0),
        'duration_seconds': post.get('duration_seconds', 0),
        'channel_info': post.get('channel_info', {}),
        'video_id': post.get('video_id', post.get('id', '')),
        'channel_id': post.get('channel_id', ''),
        'channel_title': post.get('channel_title', ''),
        'description': post.get('description', ''),
        'caption': post.get('caption', ''),
        'transcript': post.get('transcript', ''),
        'tags': post.get('tags', []),
        'category_id': post.get('category_id'),
        'default_language': post.get('default_language'),
        'thumbnail': post.get('thumbnail', {}),
    }


def map_source_to_platform(source_name: str) -> str:
    """Map fetcher source names to platform names - ALWAYS lowercase"""
    mapping = {
        'cryptopanic': 'cryptopanic',
        'cryptocompare': 'cryptocompare',
        'newsapi': 'newsapi',
        'messari': 'messari',
        'coindesk': 'coindesk',
        'reddit': 'reddit',
        'twitter': 'twitter',
        'youtube': 'youtube',
    }
    result = mapping.get(source_name.lower(), source_name.lower())
    return result.lower()  # Force lowercase


# Celery tasks
@shared_task(bind=True, max_retries=3)
def process_news_batch_async(self, news_articles: List[Dict], source_name: str = "unknown"):
    """Async task for processing news batches"""
    try:
        service = ContentIntegrationService()
        result = service.process_news_batch(news_articles, source_name)
        logger.info(f"Async processing completed for {source_name}: {result}")
        return result.__dict__
    except Exception as e:
        logger.error(f"Error in async news processing: {e}")
        self.retry(countdown=60, exc=e)


@shared_task(bind=True, max_retries=3)
def process_social_posts_async(self, social_posts: List[Dict], platform: str = "unknown"):
    """Async task for processing social media posts"""
    try:
        service = ContentIntegrationService()
        result = service.process_social_posts_batch(social_posts, platform)
        logger.info(f"Async processing completed for {platform}: {result}")
        return result.__dict__
    except Exception as e:
        logger.error(f"Error in async social processing: {e}")
        self.retry(countdown=60, exc=e)


@shared_task
def daily_threshold_optimization():
    """Daily task to optimize thresholds based on performance"""
    try:
        service = ContentIntegrationService()
        result = service.optimize_thresholds()
        logger.info(f"Daily threshold optimization: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in daily optimization: {e}")
        return {'status': 'error', 'error': str(e)}


@shared_task
def cleanup_old_cache():
    """Daily task to clean up old cache entries"""
    try:
        mongo_manager = get_mongo_manager()
        cleaned_count = mongo_manager.cleanup_old_cache(days_old=7)
        logger.info(f"Cleaned up {cleaned_count} old cache entries")
        return {'cleaned_entries': cleaned_count}
    except Exception as e:
        logger.error(f"Error in cache cleanup: {e}")
        return {'status': 'error', 'error': str(e)}


# Singleton instance
_integrator_service_instance = None


def get_integrator_service() -> ContentIntegrationService:
    """Get singleton integrator service instance"""
    global _integrator_service_instance
    if _integrator_service_instance is None:
        _integrator_service_instance = ContentIntegrationService()
    return _integrator_service_instance