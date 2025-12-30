"""
Real-Time Credibility Scoring Engine for cryptocurrency news and social content
Focuses ONLY on credibility scoring - accepts pre-processed data from integrator
"""
import os
import hashlib
import logging
from datetime import datetime, timedelta
from django.utils import timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import re 
from datetime import timezone as timezoneDt
import math
from django.core.cache import cache
from django.conf import settings

from .cross_verification_engine import VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class TrustScore:
    """Trust score result with breakdown"""
    final_score: float
    source_score: float
    content_score: float
    engagement_score: float  # For social media
    cross_check_score: float
    source_history_score: float
    recency_score: float
    confidence: float
    flags: List[str]
    reasoning: str
    cross_reference_matches: int = 0
    corroboration_sources: List[str] = field(default_factory=list)
    verified_claims: List[str] = field(default_factory=list)
    contradicted_claims: List[str] = field(default_factory=list)


@dataclass
class SourceHistoryRecord:
    """Track source historical accuracy"""
    source_name: str
    total_articles: int = 0
    accurate_articles: int = 0
    flagged_articles: int = 0
    retracted_articles: int = 0
    avg_trust_score: float = 5.0
    last_updated: datetime = field(default_factory=lambda: timezone.now())
    
    @property
    def accuracy_rate(self) -> float:
        if self.total_articles == 0:
            return 0.5
        return self.accurate_articles / self.total_articles
    
    @property
    def reliability_score(self) -> float:
        if self.total_articles < 5:
            return 5.0
        
        base_score = self.accuracy_rate * 10
        retraction_penalty = (self.retracted_articles / self.total_articles) * 3.0
        flag_penalty = (self.flagged_articles / self.total_articles) * 1.5
        
        return max(0.0, min(base_score - retraction_penalty - flag_penalty, 10.0))


class CryptoCredibilityEngine:
    """
    Real-Time Credibility Scoring Engine - SCORING ONLY
    
    Accepts pre-processed data from integrator:
    - Sentiment analysis results
    - Extracted entities
    - Cross-check data
    
    Formula (News):
    Final = 0.40(source) + 0.25(content) + 0.15(cross_ref) + 0.15(source_history) + 0.05(recency)
    
    Formula (Social):
    Final = 0.35(source) + 0.20(content) + 0.25(engagement) + 0.15(source_history) + 0.05(recency)
    """ 
    
    def __init__(self):
        # Weights for news content
        self.news_weights = {
            'source': 0.40,
            'content': 0.25,
            'cross_check': 0.15,
            'source_history': 0.15,
            'recency': 0.05
        }
        
        # Weights for social media
        self.social_weights = {
            'source': 0.35,
            'content': 0.20,
            'engagement': 0.25,
            'source_history': 0.15,
            'recency': 0.05
        }
        
        # Source reputation database
        self.source_reputation = self._initialize_source_reputation()
        
        # Source historical accuracy tracking
        self.source_history: Dict[str, SourceHistoryRecord] = {}
        self._load_source_history()
         
        # Entity/event tracking for cross-reference
        self.entity_event_cache: Dict[str, List[Dict]] = defaultdict(list)
        self.entity_cache_ttl = timedelta(hours=24)
        
        # Content quality indicators
        self.content_quality_config = {
            'min_word_count': 50,
            'ideal_word_count': 300,
            'spam_patterns': [
                r'(?i)buy now', r'(?i)limited time', r'(?i)act fast',
                r'(?i)guaranteed profit', r'(?i)100% return', r'(?i)risk free',
                r'(?i)send .* to .* address', r'(?i)double your',
            ],
            'quality_indicators': [
                'analysis', 'research', 'data', 'report', 'study',
                'according to', 'sources say', 'confirmed', 'announced'
            ]
        }
        
        # Cross-verification settings
        self.cross_check_config = {
            'time_window_hours': 6,
            'min_sources_for_verification': 2,
            'entity_match_threshold': 0.5,
            'corroboration_bonus': 2.0,
            'multi_source_bonus': 1.5,
        }
        
        # Trust thresholds
        self.trust_thresholds = {
            'high_trust': 8.0,
            'medium_trust': 6.0,
            'low_trust': 4.0,
            'very_low_trust': 2.0
        }
        
        # Cache settings
        self.cache_timeout = 3600
        
        logger.info("CryptoCredibilityEngine initialized (scoring only)")
    
    def calculate_trust_score(self, 
                             content_data: Dict,
                             sentiment_data: Optional[Dict] = None,
                             entities_data: Optional[Dict] = None,
                             verification_result: Optional[VerificationResult] = None) -> TrustScore:
        """
        CORRECTED SIGNATURE: Now accepts verification_result instead of cross_check_data
        
        Args:
            content_data: RAW content from fetchers
            sentiment_data: Pre-computed sentiment analysis
            entities_data: Pre-extracted entities
            verification_result: VerificationResult from cross_verification_engine (NEW)
        """
        try:

            # Generate cache key
            cache_key = self._generate_cache_key(content_data)
            
            # Check cache
            cached_score = cache.get(cache_key)
            if cached_score:
                return TrustScore(**cached_score)
            
            # Detect platform/type
            platform = content_data.get('platform', 'unknown').lower()
            content_type = content_data.get('type', 'news')
            
            # Choose weights based on content type
            if content_type == 'social' or platform in ['reddit', 'twitter', 'youtube']:
                weights = self.social_weights
                is_social = True
            else:
                weights = self.news_weights
                is_social = False
            
            # 1. Source Score
            source_score = self._calculate_source_score(content_data, platform)
            
            # 2. Content Quality Score
            content_score = self._calculate_content_quality_score(
                content_data, 
                sentiment_data,
                entities_data
            )
            
            # 3. Engagement Score (social media only)
            engagement_score = 0.0
            if is_social:
                engagement_score = self._calculate_engagement_score(content_data, platform)
            
            # 4. Cross-reference validation
            cross_check_result = self._calculate_cross_check_score_from_verification(
                verification_result 
            )

            cross_check_score = cross_check_result['score']
            cross_ref_matches = cross_check_result['matches']
            corroboration_sources = cross_check_result['sources']
            verified_claims = cross_check_result.get('verified_claims', [])
            contradicted_claims = cross_check_result.get('contradicted_claims', [])
            
            # 5. Source historical accuracy
            source_history_score = self._calculate_source_history_score(content_data, platform)
            
            # 6. Recency score
            recency_score = self._calculate_recency_score(content_data)
            
            # Calculate weighted final score
            if is_social:
                final_score = (
                    source_score * weights['source'] +
                    content_score * weights['content'] +
                    engagement_score * weights['engagement'] +
                    source_history_score * weights['source_history'] +
                    recency_score * weights['recency']
                )
            else:
                final_score = (
                    source_score * weights['source'] +
                    content_score * weights['content'] +
                    cross_check_score * weights['cross_check'] +
                    source_history_score * weights['source_history'] +
                    recency_score * weights['recency']
                )
            
            # Ensure bounds
            final_score = max(0.0, min(final_score, 10.0))
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                content_data, sentiment_data, entities_data, cross_ref_matches
            )
            
            # Identify flags
            flags = self._identify_content_flags(
                content_data, source_score, content_score, 
                sentiment_data, entities_data
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                source_score, content_score, engagement_score,
                cross_check_score, source_history_score, recency_score,
                flags, cross_ref_matches, corroboration_sources, is_social
            )
            
            # Create result
            trust_score = TrustScore(
                final_score=final_score,
                source_score=source_score,
                content_score=content_score,
                engagement_score=engagement_score,
                cross_check_score=cross_check_score,
                source_history_score=source_history_score,
                recency_score=recency_score,
                confidence=confidence,
                flags=flags,
                reasoning=reasoning,
                cross_reference_matches=cross_ref_matches,
                corroboration_sources=corroboration_sources,
                verified_claims=verified_claims, 
                contradicted_claims=contradicted_claims 
            )
            
            # Cache result
            cache.set(cache_key, trust_score.__dict__, self.cache_timeout)
            
            # Update entity cache
            self._update_entity_cache(content_data, entities_data, trust_score)
            
            logger.debug(
                f"Trust score: {final_score:.2f} | "
                f"Source: {source_score:.1f} | Content: {content_score:.1f} | "
                f"CrossRef: {cross_check_score:.1f}"
            )
            
            return trust_score
            
        except Exception as e:
            logger.error(f"Error calculating trust score: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_score()
    
    # =========================================================================
    # SOURCE SCORE CALCULATION
    # =========================================================================
    
    def _calculate_source_score(self, content_data: Dict, platform: str) -> float:
        """Calculate source credibility score from RAW data"""
        try:
            # Social media platforms - use user/channel credibility
            if platform in ['reddit', 'twitter', 'youtube']:
                return self._calculate_social_source_score(content_data, platform)
            
            # News sources - use source info from raw data
            source_info = content_data.get('source', {})
            
            if isinstance(source_info, dict):
                source_name = source_info.get('title', source_info.get('name', '')).lower()
                source_domain = source_info.get('domain', '').lower()
            else:
                source_name = str(source_info).lower() if source_info else ''
                source_domain = ''
            
            # Try to match against reputation database
            base_score = self._match_source_reputation(source_name, source_domain)
            
            # Platform-specific adjustments
            platform_multiplier = {
                'cryptopanic': 0.95,
                'cryptocompare': 0.95,
                'newsapi': 0.85,
                'messari': 1.0,
                'coindesk': 1.0,
            }.get(platform, 0.8)
            
            # Check for CryptoPanic votes (raw API field)
            votes = content_data.get('votes', {})
            if votes:
                positive = self._safe_int(votes.get('positive', 0))
                negative = self._safe_int(votes.get('negative', 0))
                important = self._safe_int(votes.get('important', 0))
                
                if positive + negative > 0:
                    vote_ratio = (positive + important) / (positive + negative + important + 1)
                    vote_adjustment = (vote_ratio - 0.5) * 2
                    base_score += vote_adjustment

            # Check for CryptoCompare upvotes/downvotes
            upvotes = self._safe_int(content_data.get('upvotes', 0))
            downvotes = self._safe_int(content_data.get('downvotes', 0))
            if upvotes + downvotes > 0:
                vote_ratio = upvotes / (upvotes + downvotes)
                vote_adjustment = (vote_ratio - 0.5) * 1.5
                base_score += vote_adjustment
            
            return min(max(base_score * platform_multiplier, 0.0), 10.0)
            
        except Exception as e:
            logger.warning(f"Error calculating source score: {e}")
            return 5.0
    
    def _match_source_reputation(self, source_name: str, source_domain: str = '') -> float:
        """Match source against reputation database"""
        for known_source, info in self.source_reputation.items():
            if known_source in source_name or (source_domain and known_source in source_domain):
                return info['score']
        
        trusted_domains = {
            'coindesk.com': 9.0, 'reuters.com': 9.5, 'bloomberg.com': 9.0,
            'cointelegraph.com': 8.5, 'decrypt.co': 8.0, 'theblock.co': 8.5,
            'messari.io': 9.0, 'forbes.com': 8.0, 'cnbc.com': 8.5,
        }
        
        for domain, score in trusted_domains.items():
            if domain in source_domain or domain in source_name:
                return score
        
        return 5.0

    # credibility_engine.py
#
# FIX 1: Better debugging for user credibility issues (Line 169)
# FIX 2: More detailed logging for social source scoring
#
# To apply: Replace the _calculate_social_source_score method in credibility_engine.py

    def _calculate_social_source_score(self, content_data: Dict, platform: str) -> float:
        """
        Calculate source score for social media using PRE-CALCULATED user_credibility
        Now with comprehensive debugging to identify data issues
        """
        # Get pre-calculated user credibility from integrator
        user_credibility = content_data.get('user_credibility', {})
        
        # ============================================================================
        # ENHANCED DEBUGGING: Log what we received
        # ============================================================================
        if not user_credibility or not user_credibility.get('exists', True):
            logger.warning(f"No user_credibility data for {platform} post")
            logger.debug(f"   user_credibility keys: {list(user_credibility.keys()) if user_credibility else 'EMPTY'}")
            logger.debug(f"   user_credibility.exists: {user_credibility.get('exists', 'N/A')}")
            logger.debug(f"   content_data keys (first 10): {list(content_data.keys())[:10]}")
            
            # Check if raw user data exists but wasn't processed
            if platform == 'twitter' and 'user_info' in content_data:
                logger.warning(f"   ️  Twitter user_info EXISTS but user_credibility is empty!")
                logger.debug(f"   user_info keys: {list(content_data['user_info'].keys())}")
            elif platform == 'reddit' and 'author_info' in content_data:
                logger.warning(f"   ️  Reddit author_info EXISTS but user_credibility is empty!")
                logger.debug(f"   author_info keys: {list(content_data['author_info'].keys())}")
            elif platform == 'youtube' and 'channel_info' in content_data:
                logger.warning(f"   ️  YouTube channel_info EXISTS but user_credibility is empty!")
                logger.debug(f"   channel_info keys: {list(content_data['channel_info'].keys())}")
            
            return 3.0
        
        # Start with base score
        base_score = 4.0
        
        # ========================================================================
        # FOLLOWER/SUBSCRIBER BONUSES - Platform-specific
        # ========================================================================
        followers = user_credibility.get('followers', 0)
        
        # ENHANCED LOGGING
        logger.debug(f"Social source score calculation:")
        logger.debug(f"   Platform: {platform}")
        logger.debug(f"   Followers: {followers:,}")
        logger.debug(f"   Account Age: {user_credibility.get('account_age_days', 0)} days")
        logger.debug(f"   Verified: {user_credibility.get('verified', False)}")
        
        if platform == 'twitter':
            # Twitter follower thresholds
            if followers >= 1000000:
                base_score += 3.0
                logger.debug(f"   Twitter: +3.0 for 1M+ followers")
            elif followers >= 500000:
                base_score += 2.5
                logger.debug(f"   Twitter: +2.5 for 500K+ followers")
            elif followers >= 100000:
                base_score += 2.0
                logger.debug(f"   Twitter: +2.0 for 100K+ followers")
            elif followers >= 50000:
                base_score += 1.5
                logger.debug(f"   Twitter: +1.5 for 50K+ followers")
            elif followers >= 10000:
                base_score += 1.0
                logger.debug(f"   Twitter: +1.0 for 10K+ followers")
            elif followers >= 5000:
                base_score += 0.5
                logger.debug(f"   Twitter: +0.5 for 5K+ followers")
            elif followers < 100:
                base_score -= 1.0
                logger.debug(f"   ️  Twitter: -1.0 for <100 followers")
        
        elif platform == 'reddit':
            # Reddit uses karma as follower proxy
            total_karma = followers  # Already set as karma in user_credibility
            if total_karma >= 100000:
                base_score += 3.0
                logger.debug(f"   Reddit: +3.0 for 100K+ karma")
            elif total_karma >= 50000:
                base_score += 2.5
                logger.debug(f"   Reddit: +2.5 for 50K+ karma")
            elif total_karma >= 20000:
                base_score += 2.0
                logger.debug(f"   Reddit: +2.0 for 20K+ karma")
            elif total_karma >= 10000:
                base_score += 1.5
                logger.debug(f"   Reddit: +1.5 for 10K+ karma")
            elif total_karma >= 5000:
                base_score += 1.0
                logger.debug(f"   Reddit: +1.0 for 5K+ karma")
            elif total_karma >= 1000:
                base_score += 0.5
                logger.debug(f"   Reddit: +0.5 for 1K+ karma")
            elif total_karma < 100:
                base_score -= 1.0
                logger.debug(f"   Reddit: -1.0 for <100 karma")
        
        elif platform == 'youtube':
            # YouTube subscriber thresholds
            if followers >= 1000000:
                base_score += 3.5
                logger.debug(f"   YouTube: +3.5 for 1M+ subscribers")
            elif followers >= 500000:
                base_score += 3.0
                logger.debug(f"   YouTube: +3.0 for 500K+ subscribers")
            elif followers >= 100000:
                base_score += 2.5
                logger.debug(f"   YouTube: +2.5 for 100K+ subscribers")
            elif followers >= 50000:
                base_score += 2.0
                logger.debug(f"   YouTube: +2.0 for 50K+ subscribers")
            elif followers >= 10000:
                base_score += 1.5
                logger.debug(f"   YouTube: +1.5 for 10K+ subscribers")
            elif followers >= 1000:
                base_score += 0.5
                logger.debug(f"   YouTube: +0.5 for 1K+ subscribers")
            elif followers < 100:
                base_score -= 0.5
                logger.debug(f"   ️  YouTube: -0.5 for <100 subscribers")
        
        # ========================================================================
        # ACCOUNT AGE BONUS
        # ========================================================================
        account_age = user_credibility.get('account_age_days', 0)
        age_bonus = 0.0
        if account_age > 2555:  # 7+ years
            age_bonus = 1.5
        elif account_age > 1825:  # 5+ years
            age_bonus = 1.2
        elif account_age > 1095:  # 3+ years
            age_bonus = 1.0
        elif account_age > 730:  # 2+ years
            age_bonus = 0.8
        elif account_age > 365:  # 1+ years
            age_bonus = 0.5
        elif account_age < 30:  # Less than 1 month
            age_bonus = -1.0
        
        if age_bonus != 0:
            base_score += age_bonus
            logger.debug(f"   Account age: {age_bonus:+.1f} ({account_age} days)")
        
        # ========================================================================
        # VERIFICATION BONUS
        # ========================================================================
        if user_credibility.get('verified', False):
            base_score += 1.0
            logger.debug(f"   Verified account: +1.0")
        
        if user_credibility.get('is_mod', False):  # Reddit moderator
            base_score += 0.5
            logger.debug(f"   Reddit moderator: +0.5")
        
        if user_credibility.get('is_gold', False):  # Reddit gold
            base_score += 0.3
            logger.debug(f"   Reddit gold: +0.3")
        
        # ========================================================================
        # ACTIVITY INDICATORS
        # ========================================================================
        post_count = user_credibility.get('post_count', 0)
        if post_count > 10000:
            base_score += 0.5
            logger.debug(f"   Very active: +0.5 ({post_count:,} posts)")
        elif post_count > 5000:
            base_score += 0.3
            logger.debug(f"   Active: +0.3 ({post_count:,} posts)")
        elif post_count > 1000:
            base_score += 0.2
            logger.debug(f"   Moderately active: +0.2 ({post_count:,} posts)")
        
        # Follower ratio (Twitter/YouTube)
        follower_ratio = user_credibility.get('follower_ratio', 0)
        if follower_ratio > 10:  # Many more followers than following
            base_score += 0.5
            logger.debug(f"   High follower ratio: +0.5 ({follower_ratio:.1f}x)")
        elif follower_ratio < 0.1 and followers > 100:  # Suspicious ratio
            base_score -= 0.5
            logger.debug(f"   ️  Low follower ratio: -0.5 ({follower_ratio:.2f}x)")
        
        final_score = min(base_score, 10.0)
        
        logger.debug(f"   Final source score: {final_score:.2f}")
        
        return final_score



    # =========================================================================
    # CONTENT QUALITY SCORE
    # =========================================================================

    def _calculate_content_quality_score(self, content_data: Dict, 
                                         sentiment_data: Optional[Dict],
                                         entities_data: Optional[Dict]) -> float:
        """
        Calculate content quality score using pre-processed data
        """
        score = 5.0
        
        # Get text content
        title = content_data.get('title', '') or ''
        description = content_data.get('description', '') or ''
        content = (
            content_data.get('content') or 
            content_data.get('body') or 
            content_data.get('selftext') or 
            content_data.get('text') or 
            content_data.get('summary') or
            ''
        )
        
        full_text = f"{title} {description} {content}".strip()
        
        if not full_text:
            return 3.0
        
        # Word count analysis
        word_count = len(full_text.split())
        
        if word_count >= self.content_quality_config['ideal_word_count']:
            score += 1.5
        elif word_count >= self.content_quality_config['min_word_count']:
            score += 0.5
        elif word_count < 20:
            score -= 1.5
        
        # Spam pattern detection
        text_lower = full_text.lower()
        spam_count = 0
        for pattern in self.content_quality_config['spam_patterns']:
            if re.search(pattern, text_lower):
                spam_count += 1
        
        if spam_count > 0:
            score -= min(spam_count * 1.5, 4.0)
        
        # Quality indicator detection
        quality_count = 0
        for indicator in self.content_quality_config['quality_indicators']:
            if indicator in text_lower:
                quality_count += 1
        
        if quality_count >= 3:
            score += 1.5
        elif quality_count >= 1:
            score += 0.5
        
        # Title quality
        if title:
            if title.isupper() and len(title) > 10:
                score -= 1.0
            
            if title.count('!') > 2 or title.count('?') > 2:
                score -= 0.5
            
            clickbait_patterns = [
                r'(?i)you won\'?t believe', r'(?i)shocking', r'(?i)secret',
                r'(?i)they don\'?t want you to know', r'(?i)one weird trick'
            ]
            for pattern in clickbait_patterns:
                if re.search(pattern, title):
                    score -= 0.5
                    break
        
        # URL/link density
        url_count = len(re.findall(r'https?://\S+', full_text))
        if url_count > 5:
            score -= 1.0
        elif url_count > 3:
            score -= 0.5
        
        # Crypto relevance from entities
        if entities_data:
            crypto_count = len(entities_data.get('cryptocurrencies', []))
            if crypto_count >= 2:
                score += 0.5
            elif crypto_count == 0:
                score -= 0.5
        
        # Check raw instruments field
        instruments = content_data.get('instruments', [])
        if instruments and len(instruments) > 0:
            score += 0.3
        
        # Sentiment extremity penalty
        if sentiment_data:
            sentiment_flags = sentiment_data.get('flags', [])
            if 'extreme_sentiment' in sentiment_flags:
                score -= 1.5
            if 'potential_pump_spam' in sentiment_flags:
                score -= 2.0
            if 'potential_fud' in sentiment_flags:
                score -= 1.5
        
        return max(0.0, min(score, 10.0))

    # =========================================================================
    # ENGAGEMENT SCORE (Social Media)
    # =========================================================================

    def _calculate_engagement_score(self, content_data: Dict, platform: str) -> float:
        """Calculate engagement quality score for social media"""
        if platform == 'reddit':
            return self._calculate_reddit_engagement(content_data)
        elif platform == 'twitter':
            return self._calculate_twitter_engagement(content_data)
        elif platform == 'youtube':
            return self._calculate_youtube_engagement(content_data)
        return 5.0

    def _calculate_reddit_engagement(self, content_data: Dict) -> float:
        """
        Calculate Reddit engagement score with better thresholds
        """
        score = 5.0
        
        post_score = self._safe_int(content_data.get('score', 0))
        upvote_ratio = self._safe_float(content_data.get('upvote_ratio', 0.5))
        num_comments = self._safe_int(content_data.get('num_comments', 0))
        awards = self._safe_int(content_data.get('total_awards_received', 0))
        
        logger.debug(f"Reddit engagement: score={post_score}, ratio={upvote_ratio}, comments={num_comments}, awards={awards}")
        
        # ========================================================================
        # POST SCORE THRESHOLDS - More granular
        # ========================================================================
        if post_score > 10000:
            score += 3.0
        elif post_score > 5000:
            score += 2.8
        elif post_score > 2000:
            score += 2.5
        elif post_score > 1000:
            score += 2.2
        elif post_score > 500:
            score += 2.0
        elif post_score > 250:
            score += 1.5
        elif post_score > 100:
            score += 1.2
        elif post_score > 50:
            score += 1.0
        elif post_score > 20:
            score += 0.5
        elif post_score > 10:
            score += 0.3
        elif post_score < 0:  # Downvoted
            score -= 2.0
        
        # ========================================================================
        # UPVOTE RATIO QUALITY INDICATOR
        # ========================================================================
        if upvote_ratio > 0.95:
            score += 2.0
        elif upvote_ratio > 0.9:
            score += 1.5
        elif upvote_ratio > 0.8:
            score += 1.0
        elif upvote_ratio > 0.7:
            score += 0.5
        elif upvote_ratio < 0.5:
            score -= 1.0
        elif upvote_ratio < 0.3:
            score -= 2.0
        
        # ========================================================================
        # DISCUSSION QUALITY (comments relative to score)
        # ========================================================================
        if post_score > 0:
            comment_ratio = num_comments / max(post_score, 1)
            if comment_ratio > 0.8:
                score += 1.5
            elif comment_ratio > 0.5:
                score += 1.0
            elif comment_ratio > 0.2:
                score += 0.5
        
        # Absolute comment count
        if num_comments > 500:
            score += 1.0
        elif num_comments > 200:
            score += 0.5
        elif num_comments > 100:
            score += 0.3
        
        # ========================================================================
        # AWARDS INDICATE VALUE
        # ========================================================================
        if awards >= 10:
            score += 2.0
        elif awards >= 5:
            score += 1.5
        elif awards >= 3:
            score += 1.0
        elif awards >= 1:
            score += 0.5
        
        # ========================================================================
        # SPAM DETECTION
        # ========================================================================
        if post_score < 5 and num_comments > 50:
            score -= 1.5  # Likely spam/bot
        
        return max(0.0, min(score, 10.0))


    def _calculate_twitter_engagement(self, content_data: Dict) -> float:
        """
        Calculate Twitter engagement score with better thresholds
        """
        score = 5.0
        
        public_metrics = content_data.get('public_metrics', {})
        likes = self._safe_int(public_metrics.get('like_count', 0))
        retweets = self._safe_int(public_metrics.get('retweet_count', 0))
        replies = self._safe_int(public_metrics.get('reply_count', 0))
        quotes = self._safe_int(public_metrics.get('quote_count', 0))
        
        total_engagement = likes + retweets + replies + quotes
        
        logger.debug(f"Twitter engagement: total={total_engagement}, likes={likes}, RT={retweets}, replies={replies}")
        
        # ========================================================================
        # TOTAL ENGAGEMENT THRESHOLDS - More granular
        # ========================================================================
        if total_engagement > 50000:
            score += 3.0
        elif total_engagement > 20000:
            score += 2.8
        elif total_engagement > 10000:
            score += 2.5
        elif total_engagement > 5000:
            score += 2.2
        elif total_engagement > 2000:
            score += 2.0
        elif total_engagement > 1000:
            score += 1.5
        elif total_engagement > 500:
            score += 1.2
        elif total_engagement > 200:
            score += 1.0
        elif total_engagement > 100:
            score += 0.8
        elif total_engagement > 50:
            score += 0.5
        elif total_engagement > 20:
            score += 0.3
        elif total_engagement < 5:
            score -= 1.0
        
        # ========================================================================
        # MEANINGFUL ENGAGEMENT BONUS (replies + quotes)
        # ========================================================================
        meaningful = replies + quotes
        if total_engagement > 0:
            meaningful_ratio = meaningful / total_engagement
            if meaningful_ratio > 0.5:
                score += 1.5
            elif meaningful_ratio > 0.3:
                score += 1.0
            elif meaningful_ratio > 0.15:
                score += 0.5
        
        # ========================================================================
        # RETWEET QUALITY INDICATOR
        # ========================================================================
        if likes > 0 and retweets > 0:
            rt_ratio = retweets / likes
            if rt_ratio > 0.5:  # Very high retweet ratio = high value
                score += 1.0
            elif rt_ratio > 0.3:
                score += 0.5
            elif rt_ratio < 0.05:  # Very low RT ratio
                score -= 0.3
        
        # ========================================================================
        # SPAM DETECTION - Too many replies relative to likes
        # ========================================================================
        if likes < 10 and replies > 50:
            score -= 1.5  # Likely spam/bot
        
        return max(0.0, min(score, 10.0))


    def _calculate_youtube_engagement(self, content_data: Dict) -> float:
        """
        Calculate YouTube engagement score with better thresholds
        """
        score = 5.0
        
        view_count = self._safe_int(content_data.get('view_count', 0))
        like_count = self._safe_int(content_data.get('like_count', 0))
        comment_count = self._safe_int(content_data.get('comment_count', 0))
        
        logger.debug(f"YouTube engagement: views={view_count}, likes={like_count}, comments={comment_count}")
        
        # ========================================================================
        # ABSOLUTE VIEW COUNT
        # ========================================================================
        if view_count > 5000000:
            score += 3.0
        elif view_count > 1000000:
            score += 2.5
        elif view_count > 500000:
            score += 2.2
        elif view_count > 100000:
            score += 2.0
        elif view_count > 50000:
            score += 1.5
        elif view_count > 10000:
            score += 1.0
        elif view_count > 5000:
            score += 0.8
        elif view_count > 1000:
            score += 0.5
        elif view_count > 500:
            score += 0.3
        elif view_count < 100:
            score -= 0.5
        
        if view_count > 0:
            # ========================================================================
            # LIKE RATIO - More granular thresholds
            # ========================================================================
            like_ratio = like_count / view_count
            
            if like_ratio > 0.10:  # 10%+ exceptional
                score += 2.5
            elif like_ratio > 0.08:  # 8%+ excellent
                score += 2.0
            elif like_ratio > 0.05:  # 5%+ very good
                score += 1.5
            elif like_ratio > 0.03:  # 3%+ good
                score += 1.0
            elif like_ratio > 0.02:  # 2%+ acceptable
                score += 0.5
            elif like_ratio > 0.01:  # 1%+ minimal
                score += 0.2
            elif like_ratio < 0.005:  # <0.5% poor
                score -= 1.0
            
            # ========================================================================
            # COMMENT RATIO
            # ========================================================================
            comment_ratio = comment_count / view_count
            
            if comment_ratio > 0.02:  # 2%+ very high engagement
                score += 2.0
            elif comment_ratio > 0.01:  # 1%+ high engagement
                score += 1.5
            elif comment_ratio > 0.005:  # 0.5%+ good engagement
                score += 1.0
            elif comment_ratio > 0.002:  # 0.2%+ acceptable
                score += 0.5
            elif comment_ratio < 0.0005:  # <0.05% very low
                score -= 0.5
        
        # ========================================================================
        # ABSOLUTE ENGAGEMENT COUNTS
        # ========================================================================
        if like_count > 50000:
            score += 1.0
        elif like_count > 10000:
            score += 0.5
        
        if comment_count > 5000:
            score += 1.0
        elif comment_count > 1000:
            score += 0.5
        
        # ========================================================================
        # SPAM DETECTION
        # ========================================================================
        if view_count > 10000 and like_count < 10:
            score -= 1.5  # Suspicious - high views, almost no likes
        
        return max(0.0, min(score, 10.0))

    
    # =========================================================================
    # CROSS-VERIFICATION SCORE
    # =========================================================================
    
    def _calculate_cross_check_score_from_verification(
        self, 
        verification_result: Optional[VerificationResult]
    ) -> Dict[str, Any]:
        """
        Calculate cross-check score from VerificationResult
        
        Args:
            verification_result: Result from cross_verification_engine
            
        Returns:
            Dict with score, matches, sources, verified_claims, contradicted_claims
        """
        result = {
            'score': 5.0, 
            'matches': 0, 
            'sources': [],
            'verified_claims': [],
            'contradicted_claims': []
        }
        
        if not verification_result:
            return result
        
        try:
            # Use corroboration_score directly (already 0-10 scale)
            result['score'] = verification_result.corroboration_score
            result['matches'] = verification_result.total_references
            result['sources'] = [
                ref.metadata.get('target_platform', 'unknown')
                for ref in verification_result.references[:5]
            ]
            result['verified_claims'] = verification_result.verified_claims
            result['contradicted_claims'] = verification_result.contradicted_claims
            
            logger.debug(
                f"Cross-check from verification: score={result['score']:.2f}, "
                f"refs={result['matches']}, sources={len(result['sources'])}"
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Error processing verification result: {e}")
            return result

    
    # =========================================================================
    # SOURCE HISTORY SCORE
    # =========================================================================
    
    def _calculate_source_history_score(self, content_data: Dict, platform: str) -> float:
        """Calculate source historical accuracy score"""
        try:
            source_name = self._get_source_name(content_data).lower()
            
            if source_name in self.source_history:
                record = self.source_history[source_name]
                if record.total_articles >= 5:
                    return record.reliability_score
            
            if source_name in self.source_reputation:
                base_score = self.source_reputation[source_name]['score']
                if source_name not in self.source_history:
                    base_score *= 0.9
                return base_score
            
            if platform in ['reddit', 'twitter', 'youtube']:
                return self._calculate_social_history_score(content_data, platform)
            
            return 4.0
            
        except Exception:
            return 5.0
    
    def _calculate_social_history_score(self, content_data: Dict, platform: str) -> float:
        """Calculate history score for social media"""
        base_score = 4.0
        
        if platform == 'reddit':
            author_info = content_data.get('author_info', {})
            if author_info.get('created_utc'):
                created = self._safe_timestamp(author_info['created_utc'])
                if created:
                    account_age = (timezone.now() - created).days
                else:
                    account_age = 0
            else:
                account_age = 0
        elif platform == 'twitter':
            user_info = content_data.get('user_info', {})
            created_at = self._safe_timestamp(user_info.get('created_at'))
            if created_at:
                account_age = (timezone.now() - created_at).days
            else:
                account_age = 0
        elif platform == 'youtube':
            channel_info = content_data.get('channel_info', {})
            created = self._safe_timestamp(channel_info.get('channel_created'))
            if created:
                account_age = (timezone.now() - created).days
            else:
                account_age = 0
        else:
            account_age = 0
        
        if account_age > 1095:
            base_score += 2.5
        elif account_age > 730:
            base_score += 2.0
        elif account_age > 365:
            base_score += 1.5
        elif account_age > 180:
            base_score += 1.0
        elif account_age > 90:
            base_score += 0.5
        
        return min(base_score, 10.0)
    
    # =========================================================================
    # RECENCY SCORE
    # =========================================================================
    
    def _calculate_recency_score(self, content_data: Dict) -> float:
        """Calculate recency score"""
        try:
            content_time = self._parse_content_timestamp(content_data)
            
            if not content_time:
                return 5.0
            
            now = timezone.now()
            age_hours = (now - content_time).total_seconds() / 3600
            
            if age_hours < 1:
                base_score = 9.0
            elif age_hours < 3:
                base_score = 8.5
            elif age_hours < 6:
                base_score = 8.0
            elif age_hours < 12:
                base_score = 7.5
            elif age_hours < 24:
                base_score = 7.0
            elif age_hours < 48:
                base_score = 6.0
            elif age_hours < 72:
                base_score = 5.5
            elif age_hours < 168:
                base_score = 5.0
            else:
                base_score = max(3.0, 7.0 - math.log10(age_hours))
            
            return max(0.0, min(base_score, 10.0))
            
        except Exception:
            return 6.0
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _safe_int(self, value, default: int = 0) -> int:
        """Safely convert value to int"""
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
        """Safely convert value to float"""
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
    
    def _safe_timestamp(self, value) -> Optional[datetime]:
        """Safely parse timestamp"""
        if value is None:
            return None
        
        try:
            if isinstance(value, datetime):
                return value.replace(tzinfo=timezoneDt.utc) if value.tzinfo is None else value
            
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(value, tz=timezoneDt.utc)
            
            if isinstance(value, str):
                try:
                    return datetime.fromtimestamp(float(value), tz=timezoneDt.utc)
                except (ValueError, TypeError):
                    pass
                
                try:
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    pass
            
            return None
        except (ValueError, OSError, OverflowError):
            return None
    
    def _parse_content_timestamp(self, content_data: Dict) -> Optional[datetime]:
        """Parse timestamp from various fields"""
        timestamp_fields = [
            'created_at', 'published_at', 'publishedAt', 'published',
            'created_utc', 'published_on'
        ]
        
        for field in timestamp_fields:
            timestamp = content_data.get(field)
            if timestamp:
                result = self._safe_timestamp(timestamp)
                if result:
                    return result
        return None
    
    def _get_source_name(self, content_data: Dict) -> str:
        """Extract source name"""
        source = content_data.get('source', {})
        if isinstance(source, dict):
            return source.get('title', source.get('name', source.get('domain', 'unknown')))
        return str(source) if source else content_data.get('platform', 'unknown')
    
    def _calculate_confidence(self, content_data: Dict,
                             sentiment_data: Optional[Dict],
                             entities_data: Optional[Dict],
                             cross_ref_matches: int) -> float:
        """Calculate confidence score"""
        confidence = 0.5
        
        if content_data.get('title'):
            confidence += 0.1
        
        if content_data.get('description') or content_data.get('content'):
            confidence += 0.1
        
        if sentiment_data and sentiment_data.get('confidence', 0) > 0.7:
            confidence += 0.1
        
        if entities_data:
            total_entities = (
                len(entities_data.get('cryptocurrencies', [])) +
                len(entities_data.get('persons', [])) +
                len(entities_data.get('organizations', []))
            )
            if total_entities >= 3:
                confidence += 0.1
        
        if cross_ref_matches >= 3:
            confidence += 0.2
        elif cross_ref_matches >= 2:
            confidence += 0.15
        elif cross_ref_matches >= 1:
            confidence += 0.1
        
        source_name = self._get_source_name(content_data).lower()
        if source_name in self.source_history:
            record = self.source_history[source_name]
            if record.total_articles >= 10:
                confidence += 0.1
        
        return max(0.1, min(confidence, 1.0))
    
    def _identify_content_flags(self, content_data: Dict, source_score: float,
                               content_score: float, sentiment_data: Optional[Dict],
                               entities_data: Optional[Dict]) -> List[str]:
        """Identify content flags"""
        flags = []
        
        if source_score < 3.0:
            flags.append('low_source_credibility')
        
        if content_score < 3.0:
            flags.append('low_content_quality')
        
        if sentiment_data:
            sentiment_flags = sentiment_data.get('flags', [])
            if 'extreme_sentiment' in sentiment_flags:
                flags.append('extreme_sentiment')
            if 'potential_pump_spam' in sentiment_flags:
                flags.append('potential_pump_spam')
            if 'potential_fud' in sentiment_flags:
                flags.append('potential_fud')
            if 'high_greed_fomo' in sentiment_flags:
                flags.append('hype_driven')
            if 'high_fear' in sentiment_flags:
                flags.append('panic_inducing')
        
        title = content_data.get('title', '').lower()
        
        if title and title.isupper():
            flags.append('all_caps_title')
        
        if title and (title.count('!') > 2 or title.count('?') > 2):
            flags.append('excessive_punctuation')
        
        return list(set(flags))
    
    def _generate_reasoning(self, source_score: float, content_score: float,
                           engagement_score: float, cross_check_score: float,
                           source_history_score: float, recency_score: float,
                           flags: List[str], cross_ref_matches: int,
                           corroboration_sources: List[str], is_social: bool) -> str:
        """Generate reasoning text"""
        parts = []
        
        if source_score >= 8.0:
            parts.append("High-credibility source")
        elif source_score >= 6.0:
            parts.append("Moderately credible source")
        elif source_score < 4.0:
            parts.append("Low-credibility source")
        
        if content_score >= 7.0:
            parts.append("high-quality content")
        elif content_score < 4.0:
            parts.append("low-quality content")
        
        if is_social:
            if engagement_score >= 7.0:
                parts.append("strong engagement")
            elif engagement_score < 4.0:
                parts.append("weak engagement")
        else:
            if cross_ref_matches >= 3:
                sources_str = ", ".join(corroboration_sources[:3])
                parts.append(f"corroborated by {cross_ref_matches} sources ({sources_str})")
            elif cross_ref_matches >= 2:
                parts.append(f"verified by {cross_ref_matches} sources")
            elif cross_check_score < 5.0:
                parts.append("unverified")
        
        if 'panic_inducing' in flags:
            parts.append("panic language")
        if 'hype_driven' in flags:
            parts.append("hype content")
        if 'potential_pump_spam' in flags:
            parts.append("potential spam")
        
        return "; ".join(parts).capitalize() if parts else "Standard content"
    
    def _generate_cache_key(self, content_data: Dict) -> str:
        """Generate cache key"""
        key_data = f"{content_data.get('source_id', '')}{content_data.get('title', '')}{content_data.get('platform', '')}"
        return f"trust_score_v4:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def _create_fallback_score(self) -> TrustScore:
        """Create fallback trust score"""
        return TrustScore(
            final_score=5.0,
            source_score=5.0,
            content_score=5.0,
            engagement_score=5.0,
            cross_check_score=5.0,
            source_history_score=5.0,
            recency_score=5.0,
            confidence=0.3,
            flags=['analysis_error'],
            reasoning="Unable to complete full analysis",
            cross_reference_matches=0,
            corroboration_sources=[]
        )
    
    def _update_entity_cache(self, content_data: Dict, entities_data: Optional[Dict],
                            trust_score: TrustScore):
        """Update entity cache for cross-referencing"""
        try:
            if not entities_data:
                return
            
            cache_entry = {
                'source_id': content_data.get('source_id'),
                'title': content_data.get('title'),
                'source': self._get_source_name(content_data),
                'trust_score': trust_score.final_score,
                'timestamp': timezone.now(),
                'entities': entities_data
            }
            
            for crypto in entities_data.get('cryptocurrencies', [])[:3]:
                key = crypto.lower()
                self.entity_event_cache[key].append(cache_entry)
                
                cutoff = timezone.now() - self.entity_cache_ttl
                self.entity_event_cache[key] = [
                    e for e in self.entity_event_cache[key]
                    if e['timestamp'] > cutoff
                ][-100:]
            
        except Exception as e:
            logger.warning(f"Error updating entity cache: {e}")
    
    def _initialize_source_reputation(self) -> Dict[str, Dict]:
        """Initialize source reputation database"""
        return {
            # Tier 1: Highest Credibility (9.0+)
            'coindesk': {'score': 9.0, 'category': 'established_news', 'tier': 1},
            'reuters': {'score': 9.5, 'category': 'mainstream_news', 'tier': 1},
            'bloomberg': {'score': 9.0, 'category': 'financial_news', 'tier': 1},
            'messari': {'score': 9.0, 'category': 'research', 'tier': 1},
            'the block': {'score': 8.5, 'category': 'crypto_news', 'tier': 1},
            'wall street journal': {'score': 9.5, 'category': 'financial_news', 'tier': 1},
            
            # Tier 2: High Credibility (7.5-8.9)
            'cointelegraph': {'score': 8.5, 'category': 'crypto_news', 'tier': 2},
            'decrypt': {'score': 8.0, 'category': 'crypto_news', 'tier': 2},
            'forbes': {'score': 8.0, 'category': 'business_news', 'tier': 2},
            'bitcoin magazine': {'score': 7.5, 'category': 'crypto_magazine', 'tier': 2},
            'cryptocompare': {'score': 7.5, 'category': 'data_aggregator', 'tier': 2},
            'coinbase': {'score': 7.5, 'category': 'exchange', 'tier': 2},
            'kraken': {'score': 7.5, 'category': 'exchange', 'tier': 2},
             
            # Tier 3: Medium Credibility (5.0-7.4)
            'cryptoslate': {'score': 7.0, 'category': 'crypto_news', 'tier': 3},
            'binance': {'score': 7.0, 'category': 'exchange', 'tier': 3},
            'newsbtc': {'score': 6.5, 'category': 'crypto_news', 'tier': 3},
            'cryptopotato': {'score': 6.0, 'category': 'crypto_news', 'tier': 3},
            'ambcrypto': {'score': 6.0, 'category': 'crypto_news', 'tier': 3},
            'reddit': {'score': 5.5, 'category': 'social', 'tier': 3},
             
            # Tier 4: Lower Credibility (3.0-4.9)
            'twitter': {'score': 4.5, 'category': 'social', 'tier': 4},
            'youtube': {'score': 4.0, 'category': 'video', 'tier': 4},
            'medium': {'score': 4.0, 'category': 'blog_platform', 'tier': 4},
            
            # Tier 5: Unknown
            'unknown': {'score': 3.0, 'category': 'unknown', 'tier': 5}
        }
    
    def _load_source_history(self):
        """Load source historical accuracy data"""
        try:
            cached_history = cache.get('source_history_data')
            if cached_history:
                self.source_history = cached_history
        except Exception as e:
            logger.warning(f"Could not load source history: {e}")
    
    def _save_source_history(self):
        """Save source historical accuracy data"""
        try:
            cache.set('source_history_data', self.source_history, timeout=86400)
        except Exception as e:
            logger.warning(f"Could not save source history: {e}")
    
    def update_source_history(self, source_name: str, was_accurate: bool, 
                             was_flagged: bool = False, was_retracted: bool = False,
                             trust_score: float = 5.0):
        """Update source historical accuracy"""
        source_key = source_name.lower().strip()
        
        if source_key not in self.source_history:
            self.source_history[source_key] = SourceHistoryRecord(source_name=source_key)
        
        record = self.source_history[source_key]
        record.total_articles += 1
        
        if was_accurate:
            record.accurate_articles += 1
        if was_flagged:
            record.flagged_articles += 1
        if was_retracted:
            record.retracted_articles += 1
        
        n = record.total_articles
        record.avg_trust_score = ((record.avg_trust_score * (n - 1)) + trust_score) / n
        record.last_updated = timezone.now()
        
        self._save_source_history()
    
    def determine_content_action(self, trust_score: TrustScore) -> Dict[str, Any]:
        """Determine action based on trust score"""
        score = trust_score.final_score
        flags = trust_score.flags
        confidence = trust_score.confidence
        
        if score >= self.trust_thresholds['high_trust']:
            base_action = 'auto_approve'
            priority = 'high'
            delay_minutes = 0
        elif score >= self.trust_thresholds['medium_trust']:
            base_action = 'normal_flow'
            priority = 'normal'
            delay_minutes = 0
        elif score >= self.trust_thresholds['low_trust']:
            base_action = 'delayed_review'
            priority = 'low'
            delay_minutes = 30
        else:
            base_action = 'manual_review'
            priority = 'review_required'
            delay_minutes = 60
        
        critical_flags = ['panic_inducing', 'extreme_sentiment', 'potential_pump_spam']
        if any(flag in flags for flag in critical_flags):
            if base_action == 'auto_approve':
                base_action = 'delayed_review'
                delay_minutes = 15
            elif base_action == 'normal_flow':
                base_action = 'delayed_review'
                delay_minutes = 30
        
        if confidence < 0.4:
            if base_action == 'auto_approve':
                base_action = 'normal_flow'
            elif base_action == 'normal_flow':
                base_action = 'delayed_review'
                delay_minutes = 20
        
        return {
            'action': base_action,
            'priority': priority,
            'delay_minutes': delay_minutes,
            'reasoning': trust_score.reasoning,
            'flags': flags,
            'trust_score': score,
            'confidence': confidence,
            'requires_human_review': base_action == 'manual_review',
            'auto_approve': base_action == 'auto_approve',
            'cross_reference_matches': trust_score.cross_reference_matches,
            'corroboration_sources': trust_score.corroboration_sources
        }


class ThresholdManager:
    """Manages adaptive thresholds for content moderation"""
    
    def __init__(self, credibility_engine: CryptoCredibilityEngine):
        self.credibility_engine = credibility_engine
        self.performance_records: List[Dict] = []
        self.max_records = 1000
        
        self.thresholds = {
            'high_trust': 8.0,
            'medium_trust': 6.0,
            'low_trust': 4.0,
            'very_low_trust': 2.0
        }
        
        self._load_thresholds()
    
    def _load_thresholds(self):
        """Load thresholds from cache"""
        try:
            cached_thresholds = cache.get('trust_score_thresholds')
            if cached_thresholds:
                self.thresholds = cached_thresholds
                self.credibility_engine.trust_thresholds = cached_thresholds
        except Exception as e:
            logger.warning(f"Could not load thresholds: {e}")
    
    def _save_thresholds(self):
        """Save thresholds to cache"""
        try:
            cache.set('trust_score_thresholds', self.thresholds, timeout=86400 * 30)
            self.credibility_engine.trust_thresholds = self.thresholds
        except Exception as e:
            logger.warning(f"Could not save thresholds: {e}")
    
    
    def record_performance(self, content_id: str, predicted_action: str, 
                          actual_outcome: str, user_feedback: Optional[str] = None):
        """
        Record performance data for threshold optimization
        
        Args:
            content_id: ID of the content
            predicted_action: What the system predicted (auto_approve, pending, etc.)
            actual_outcome: What actually happened (approved, rejected, flagged)
            user_feedback: Optional reviewer notes
        """
        record = {
            'content_id': content_id,
            'predicted_action': predicted_action,
            'actual_outcome': actual_outcome,
            'user_feedback': user_feedback,
            'timestamp': timezone.now().isoformat(),
            'was_correct': self._determine_correctness(predicted_action, actual_outcome)
        }
        
        self.performance_records.append(record)
        
        # Keep only recent records
        if len(self.performance_records) > self.max_records:
            self.performance_records = self.performance_records[-self.max_records:]
        
        logger.debug(f"Recorded performance: {predicted_action} -> {actual_outcome}")
    
    def _determine_correctness(self, predicted: str, actual: str) -> bool:
        """Determine if prediction was correct"""
        correct_mappings = {
            ('auto_approve', 'approved'): True,
            ('normal_flow', 'approved'): True,
            ('normal_flow', 'pending'): True,
            ('delayed_review', 'approved'): True,
            ('delayed_review', 'flagged'): True,
            ('manual_review', 'flagged'): True,
            ('manual_review', 'rejected'): True,
        }
        return correct_mappings.get((predicted, actual), False)
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze recent performance data
        
        Returns:
            Performance analysis with accuracy metrics
        """
        if len(self.performance_records) < 10:
            return {
                'status': 'insufficient_data',
                'records_count': len(self.performance_records),
                'minimum_required': 10
            }
        
        total = len(self.performance_records)
        correct = sum(1 for r in self.performance_records if r['was_correct'])
        
        # Action breakdown
        action_stats = {}
        for record in self.performance_records:
            action = record['predicted_action']
            if action not in action_stats:
                action_stats[action] = {'total': 0, 'correct': 0}
            action_stats[action]['total'] += 1
            if record['was_correct']:
                action_stats[action]['correct'] += 1
        
        # Calculate accuracy per action
        for action in action_stats:
            stats = action_stats[action]
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        # False positive/negative rates
        false_positives = sum(
            1 for r in self.performance_records 
            if r['predicted_action'] == 'manual_review' and r['actual_outcome'] == 'approved'
        )
        
        false_negatives = sum(
            1 for r in self.performance_records 
            if r['predicted_action'] == 'auto_approve' and r['actual_outcome'] in ['rejected', 'flagged']
        )
        
        return {
            'status': 'analysis_complete',
            'total_records': total,
            'overall_accuracy': correct / total,
            'action_breakdown': action_stats,
            'false_positive_rate': false_positives / total,
            'false_negative_rate': false_negatives / total,
            'current_thresholds': self.thresholds,
            'recommendation': self._generate_recommendation(action_stats, false_positives, false_negatives, total)
        }
    
    def _generate_recommendation(self, action_stats: Dict, false_positives: int, 
                                false_negatives: int, total: int) -> str:
        """Generate threshold adjustment recommendation"""
        fp_rate = false_positives / total if total > 0 else 0
        fn_rate = false_negatives / total if total > 0 else 0
        
        if fn_rate > 0.1:
            return "Consider raising auto_approve threshold - too many false negatives"
        elif fp_rate > 0.2:
            return "Consider lowering manual_review threshold - too many false positives"
        elif fp_rate < 0.05 and fn_rate < 0.05:
            return "Thresholds are well-calibrated"
        else:
            return "Monitor performance - minor adjustments may be needed"
    
    def auto_adjust_thresholds(self, max_adjustment: float = 0.3) -> bool:
        """
        Automatically adjust thresholds based on performance
        
        Args:
            max_adjustment: Maximum adjustment per threshold
            
        Returns:
            True if adjustments were made
        """
        analysis = self.analyze_performance()
        
        if analysis['status'] != 'analysis_complete':
            return False
        
        made_adjustments = False
        
        # Adjust based on false positive/negative rates
        fp_rate = analysis['false_positive_rate']
        fn_rate = analysis['false_negative_rate']
        
        if fn_rate > 0.1:
            # Too many false negatives - raise auto_approve threshold
            adjustment = min(fn_rate * 2, max_adjustment)
            self.thresholds['high_trust'] = min(
                self.thresholds['high_trust'] + adjustment, 
                9.5
            )
            made_adjustments = True
            logger.info(f"Raised high_trust threshold to {self.thresholds['high_trust']}")
        
        if fp_rate > 0.2:
            # Too many false positives - lower manual_review threshold
            adjustment = min(fp_rate, max_adjustment)
            self.thresholds['low_trust'] = max(
                self.thresholds['low_trust'] - adjustment,
                2.0
            )
            made_adjustments = True
            logger.info(f"Lowered low_trust threshold to {self.thresholds['low_trust']}")
        
        if made_adjustments:
            self._save_thresholds()
        
        return made_adjustments
    
    def set_threshold(self, threshold_name: str, value: float) -> bool:
        """
        Manually set a threshold
        
        Args:
            threshold_name: Name of threshold to set
            value: New value (0-10)
            
        Returns:
            True if set successfully
        """
        if threshold_name not in self.thresholds:
            logger.warning(f"Unknown threshold: {threshold_name}")
            return False
        
        if not 0 <= value <= 10:
            logger.warning(f"Invalid threshold value: {value}")
            return False
        
        self.thresholds[threshold_name] = value
        self._save_thresholds()
        
        logger.info(f"Set {threshold_name} to {value}")
        return True
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current thresholds"""
        return self.thresholds.copy()


# Singleton instances
_credibility_engine_instance = None
_threshold_manager_instance = None


def get_credibility_engine() -> CryptoCredibilityEngine:
    """Get singleton credibility engine instance"""
    global _credibility_engine_instance
    
    if _credibility_engine_instance is None:
        _credibility_engine_instance = CryptoCredibilityEngine()
    
    return _credibility_engine_instance


def get_threshold_manager() -> ThresholdManager:
    """Get singleton threshold manager instance"""
    global _threshold_manager_instance
    
    if _threshold_manager_instance is None:
        engine = get_credibility_engine()
        _threshold_manager_instance = ThresholdManager(engine)
    
    return _threshold_manager_instance