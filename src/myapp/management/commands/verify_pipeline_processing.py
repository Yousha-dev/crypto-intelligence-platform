"""
Django Management Command: Verify Pipeline Processing
Shows exactly what analysis was performed on stored content
ENHANCED with detailed metric display and validation
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
import json


class Command(BaseCommand):
    help = 'Verify what analysis was performed on stored content'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--limit',
            type=int,
            default=5,
            help='Number of items to examine (default: 5)'
        )
        
        parser.add_argument(
            '--content-type',
            type=str,
            choices=['news', 'social', 'all'],
            default='all',
            help='Which content type to verify'
        )
    
    def handle(self, *args, **options):
        limit = options['limit']
        content_type = options['content_type']
        
        self.stdout.write(self.style.SUCCESS("\n" + "="*80))
        self.stdout.write(self.style.SUCCESS("   PIPELINE PROCESSING VERIFICATION"))
        self.stdout.write(self.style.SUCCESS("="*80 + "\n"))
        
        from myapp.services.mongo_manager import get_mongo_manager
        
        mongo = get_mongo_manager()
        
        # Verify News Articles
        if content_type in ['all', 'news']:
            self.stdout.write(self.style.WARNING("\nVERIFYING NEWS ARTICLES"))
            self.stdout.write("-" * 80)
            
            articles = mongo.collections['news_articles'].find().sort('created_at', -1).limit(limit)
            
            for i, article in enumerate(articles, 1):
                self._display_article_details(article, i)
        
        # Verify Social Posts
        if content_type in ['all', 'social']:
            self.stdout.write(self.style.WARNING("\n\nVERIFYING SOCIAL POSTS"))
            self.stdout.write("-" * 80)
            
            posts = mongo.collections['social_posts'].find().sort('created_at', -1).limit(limit)
            
            for i, post in enumerate(posts, 1):
                self._display_post_details(post, i)
        
        # Check for Hashtag Analysis
        self.stdout.write(self.style.WARNING("\n\nVERIFYING HASHTAG ANALYSIS"))
        self.stdout.write("-" * 80)
        self._verify_hashtag_analysis()
        
        # Check for Topic Modeling
        self.stdout.write(self.style.WARNING("\n\nVERIFYING TOPIC MODELING"))
        self.stdout.write("-" * 80)
        self._verify_topic_modeling()
        
        # Check RAG Indexing
        self.stdout.write(self.style.WARNING("\n\nVERIFYING RAG INDEXING"))
        self.stdout.write("-" * 80)
        self._verify_rag_indexing()
        
        self.stdout.write("\n" + "="*80)
        self.stdout.write(self.style.SUCCESS("Verification complete!"))
        self.stdout.write("="*80 + "\n")
    
    def _display_article_details(self, article, index):
        """Display article details"""
        self.stdout.write(f"\n📄 Article {index}: {article.get('title', 'N/A')[:60]}...")
        self.stdout.write(f"   Source: {article.get('platform', 'N/A')}")
        self.stdout.write(f"   Trust Score: {article.get('trust_score', 0):.2f}")
        
        self._verify_text_processing(article)
        self._verify_sentiment_analysis(article)
        self._verify_credibility_scoring(article)
        self._verify_entity_extraction(article)
    
    def _display_post_details(self, post, index):
        """
        ENHANCED: Display comprehensive post details including raw metrics
        """
        platform = post.get('platform', 'unknown')
        title = post.get('title') or post.get('text') or post.get('content')
        
        self.stdout.write(f"\nPost {index}: {(title or 'N/A')[:60]}...")
        self.stdout.write(f"   Platform: {platform}")
        self.stdout.write(f"   Trust Score: {post.get('trust_score', 0):.2f}")
        
        # ============================================================================
        # NEW: RAW ENGAGEMENT METRICS
        # ============================================================================
        if 'engagement_metrics' in post:
            metrics = post['engagement_metrics']
            self.stdout.write(f"   RAW ENGAGEMENT METRICS:")
            
            if platform == 'youtube':
                views = metrics.get('view_count', 0)
                likes = metrics.get('like_count', 0)
                comments = metrics.get('comment_count', 0)
                duration = metrics.get('duration_seconds', 0)
                
                self.stdout.write(f"      Views: {views:,}")
                self.stdout.write(f"      Likes: {likes:,}")
                self.stdout.write(f"      Comments: {comments:,}")
                self.stdout.write(f"      Duration: {duration}s")
                
                # Calculate ratios for validation
                if views > 0:
                    like_rate = (likes / views) * 100
                    comment_rate = (comments / views) * 100
                    self.stdout.write(f"      Like Rate: {like_rate:.2f}%")
                    self.stdout.write(f"      Comment Rate: {comment_rate:.2f}%")
            
            elif platform == 'twitter':
                likes = metrics.get('like_count', 0)
                retweets = metrics.get('retweet_count', 0)
                replies = metrics.get('reply_count', 0)
                quotes = metrics.get('quote_count', 0)
                total = likes + retweets + replies + quotes
                
                self.stdout.write(f"      Likes: {likes:,}")
                self.stdout.write(f"      Retweets: {retweets:,}")
                self.stdout.write(f"      Replies: {replies:,}")
                self.stdout.write(f"      Quotes: {quotes:,}")
                self.stdout.write(f"      Total Engagement: {total:,}")
                
                if total > 0:
                    meaningful_pct = ((replies + quotes) / total) * 100
                    self.stdout.write(f"      Meaningful Engagement: {meaningful_pct:.1f}%")
            
            elif platform == 'reddit':
                score = metrics.get('score', 0)
                ratio = metrics.get('upvote_ratio', 0)
                comments = metrics.get('num_comments', 0)
                awards = metrics.get('total_awards_received', 0)
                
                self.stdout.write(f"      Score: {score:,}")
                self.stdout.write(f"      Upvote Ratio: {ratio:.2%}")
                self.stdout.write(f"      Comments: {comments:,}")
                self.stdout.write(f"      Awards: {awards}")
                
                # Calculate comment ratio
                if score > 0:
                    comment_ratio = comments / score
                    self.stdout.write(f"      Comment Ratio: {comment_ratio:.2f}")
        
        # ============================================================================
        # NEW: USER CREDIBILITY DETAILS
        # ============================================================================
        if 'user_credibility' in post:
            cred = post['user_credibility']
            self.stdout.write(f"   👤 USER CREDIBILITY:")
            
            exists = cred.get('exists', False)
            self.stdout.write(f"      Data Exists: {exists}")
            
            if exists:
                followers = cred.get('followers', 0)
                following = cred.get('following', 0)
                age_days = cred.get('account_age_days', 0)
                verified = cred.get('verified', False)
                influence = cred.get('influence_level', 'None')
                
                self.stdout.write(f"      Followers: {followers:,}")
                if following > 0:
                    self.stdout.write(f"      Following: {following:,}")
                    ratio = followers / following if following > 0 else 0
                    self.stdout.write(f"      Follower Ratio: {ratio:.1f}x")
                
                self.stdout.write(f"      Account Age: {age_days} days ({age_days/365:.1f} years)")
                self.stdout.write(f"      Verified: {'✓' if verified else '✗'}")
                self.stdout.write(f"      Influence Level: {influence}")
                
                # Platform-specific details
                if platform == 'reddit':
                    karma = cred.get('total_karma', 0)
                    is_mod = cred.get('is_mod', False)
                    is_gold = cred.get('is_gold', False)
                    self.stdout.write(f"      Total Karma: {karma:,}")
                    if is_mod:
                        self.stdout.write(f"      Moderator: ✓")
                    if is_gold:
                        self.stdout.write(f"      Reddit Gold: ✓")
                
                elif platform == 'twitter':
                    tweet_count = cred.get('tweet_count', 0)
                    listed = cred.get('listed_count', 0)
                    self.stdout.write(f"      Total Tweets: {tweet_count:,}")
                    if listed > 0:
                        self.stdout.write(f"      Listed Count: {listed:,}")
                
                elif platform == 'youtube':
                    videos = cred.get('video_count', 0)
                    total_views = cred.get('total_view_count', 0)
                    self.stdout.write(f"      Total Videos: {videos:,}")
                    if total_views > 0:
                        self.stdout.write(f"      Channel Total Views: {total_views:,}")
                        if videos > 0:
                            avg_views = total_views / videos
                            self.stdout.write(f"      Avg Views/Video: {avg_views:,.0f}")
            else:
                self.stdout.write(self.style.WARNING("      ️  No user credibility data found!"))
                self.stdout.write(self.style.WARNING("      This will cause source score to default to 3.0"))
        
        # Continue with standard verification
        self._verify_text_processing(post)
        self._verify_sentiment_analysis(post)
        self._verify_credibility_scoring(post)
        self._verify_entity_extraction(post)
    
    def _verify_text_processing(self, content):
        """Verify text processing was performed"""
        text_proc = content.get('text_processing', {})
        
        if text_proc:
            self.stdout.write(f"   TEXT PROCESSING:")
            self.stdout.write(f"      Language: {text_proc.get('language', 'N/A')} ({text_proc.get('language_confidence', 0):.0%})")
            self.stdout.write(f"      Word Count: {text_proc.get('word_count', 0)}")
            
            hashtags = text_proc.get('hashtags', [])
            if hashtags:
                self.stdout.write(f"      Hashtags: {', '.join(hashtags[:5])}")
            
            mentions = text_proc.get('mentions', [])
            if mentions:
                self.stdout.write(f"      Mentions: {', '.join(mentions[:5])}")
        else:
            self.stdout.write(f"   TEXT PROCESSING: NOT FOUND")
    
    def _verify_sentiment_analysis(self, content):
        """
        ENHANCED: Verify sentiment analysis with emotion validation
        """
        sentiment = content.get('sentiment_analysis', {})
        
        if sentiment:
            self.stdout.write(f"   SENTIMENT ANALYSIS:")
            self.stdout.write(f"      Label: {sentiment.get('label', 'N/A')}")
            self.stdout.write(f"      Score: {sentiment.get('score', 0):.3f}")
            self.stdout.write(f"      Confidence: {sentiment.get('confidence', 0):.2%}")
            
            # Check probabilities
            bullish_prob = sentiment.get('bullish_probability', 0)
            bearish_prob = sentiment.get('bearish_probability', 0)
            neutral_prob = sentiment.get('neutral_probability', 0)
            
            if bullish_prob or bearish_prob or neutral_prob:
                self.stdout.write(f"      Probabilities: Bullish={bullish_prob:.2%}, " +
                                f"Bearish={bearish_prob:.2%}, " +
                                f"Neutral={neutral_prob:.2%}")
            
            # ENHANCED: Show ALL emotions with visual bars
            emotions = sentiment.get('emotions', {})
            if emotions:
                self.stdout.write(f"      Emotions (all):")
                # Sort by value descending
                sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                
                has_non_zero = False
                for emotion, value in sorted_emotions:
                    if value > 0:  # Only show non-zero
                        has_non_zero = True
                        bar = '█' * int(value * 10)  # Visual bar
                        self.stdout.write(f"         {emotion}: {value:.2f} {bar}")
                
                # Flag if all emotions are zero
                if not has_non_zero or all(v == 0 for v in emotions.values()):
                    self.stdout.write(self.style.WARNING(
                        "         ️  ALL EMOTIONS ARE ZERO"
                    )) 
            
            flags = sentiment.get('flags', [])
            if flags:
                self.stdout.write(f"      ️  Flags: {', '.join(flags)}")
        else:
            self.stdout.write(f"   SENTIMENT ANALYSIS: NOT FOUND")
    
    def _verify_credibility_scoring(self, content):
        """
        ENHANCED: Verify credibility scoring with component validation
        """
        cred_analysis = content.get('credibility_analysis', {})
        trust_score = content.get('trust_score')
        platform = content.get('platform', 'unknown')
        
        if cred_analysis or trust_score is not None:
            self.stdout.write(f"   CREDIBILITY SCORING:")
            
            # Top-level trust_score
            if trust_score is not None:
                self.stdout.write(f"      Trust Score: {trust_score:.2f}")
            
            # Credibility analysis breakdown
            if cred_analysis:
                self.stdout.write(f"      Breakdown:")
                
                # Component scores
                source_score = cred_analysis.get('source_score')
                content_score = cred_analysis.get('content_score')
                engagement_score = cred_analysis.get('engagement_score')
                cross_check_score = cred_analysis.get('cross_check_score')
                source_history_score = cred_analysis.get('source_history_score')
                recency_score = cred_analysis.get('recency_score')
                
                if source_score is not None:
                    self.stdout.write(f"         Source: {source_score:.2f}")
                if content_score is not None:
                    self.stdout.write(f"         Content: {content_score:.2f}")
                if engagement_score is not None or engagement_score == 0:
                    self.stdout.write(f"         Engagement: {engagement_score:.2f}")
                if cross_check_score is not None:
                    self.stdout.write(f"         Cross-Check: {cross_check_score:.2f}")
                if source_history_score is not None:
                    self.stdout.write(f"         History: {source_history_score:.2f}")
                if recency_score is not None:
                    self.stdout.write(f"         Recency: {recency_score:.2f}")
                
                confidence = cred_analysis.get('confidence')
                if confidence is not None:
                    self.stdout.write(f"      Confidence: {confidence:.2%}")
                
                flags = cred_analysis.get('flags', [])
                if flags:
                    self.stdout.write(f"      ️  Flags: {', '.join(flags)}")
                
                reasoning = cred_analysis.get('reasoning', '')
                if reasoning:
                    self.stdout.write(f"      Reasoning: {reasoning[:100]}...")
                
                # ========================================================================
                # NEW: VALIDATION CHECKS
                # ========================================================================
                
                # Check for default source score (likely missing user_credibility)
                if source_score == 3.0 and platform in ['twitter', 'reddit', 'youtube']:
                    self.stdout.write(self.style.WARNING(
                        "      ️  Source score is exactly 3.0 - likely defaulted due to missing user_credibility"
                    ))
                    self.stdout.write(self.style.WARNING(
                        "      Check debug logs for user_credibility issues"
                    ))
                
                # Validate high engagement scores
                if engagement_score and engagement_score > 7.0:
                    self.stdout.write(self.style.SUCCESS(
                        f"      High engagement score ({engagement_score:.2f}) - check metrics above to validate"
                    ))
                
                # Check for no cross-references
                if cross_check_score == 5.0:
                    self.stdout.write(self.style.WARNING(
                        "      ℹ️  Cross-check score is exactly 5.0 - likely no cross-references found"
                    ))
        else:
            self.stdout.write(f"   CREDIBILITY SCORING: NOT FOUND")
    
    def _verify_entity_extraction(self, content):
        """
        ENHANCED: Verify entity extraction with spurious detection warnings
        """
        entities = content.get('extracted_entities', {})
        
        if entities:
            self.stdout.write(f"   ENTITY EXTRACTION:")
            
            cryptos = entities.get('cryptocurrencies', [])
            if cryptos:
                self.stdout.write(f"      Cryptocurrencies: {', '.join(cryptos[:5])}")
            
            exchanges = entities.get('exchanges', [])
            if exchanges:
                self.stdout.write(f"      Exchanges: {', '.join(exchanges[:3])}")
            
            persons = entities.get('persons', [])
            if persons:
                self.stdout.write(f"      Persons: {', '.join(persons[:3])}")
            
            orgs = entities.get('organizations', [])
            if orgs:
                # Filter out suspicious single-letter/digit orgs
                valid_orgs = [o for o in orgs if len(o) > 1 and not o.isdigit()]
                if valid_orgs:
                    self.stdout.write(f"      Organizations: {', '.join(valid_orgs[:3])}")
                if len(orgs) != len(valid_orgs):
                    self.stdout.write(self.style.WARNING(
                        f"      ️  Filtered {len(orgs) - len(valid_orgs)} spurious org detections (e.g., 'K', '95')"
                    ))
            
            amounts = entities.get('money_amounts', [])
            if amounts:
                # Filter out suspicious numbers-only amounts
                valid_money = [m for m in amounts if any(c in m for c in ['$', '€', '£', 'K', 'M', 'B'])]
                if valid_money:
                    self.stdout.write(f"      Money Amounts: {', '.join(valid_money[:3])}")
                if len(amounts) != len(valid_money):
                    self.stdout.write(self.style.WARNING(
                        f"      ️  Filtered {len(amounts) - len(valid_money)} spurious money detections"
                    ))
        else:
            self.stdout.write(f"   ENTITY EXTRACTION: NOT FOUND")
    
    def _verify_hashtag_analysis(self):
        """Check if hashtag analysis is being performed"""
        try:
            from myapp.services.content.hashtag_analyzer import get_hashtag_analyzer
            
            analyzer = get_hashtag_analyzer()
            summary = analyzer.get_summary()
            
            if summary.get('total_hashtags', 0) > 0:
                self.stdout.write(f"   HASHTAG ANALYSIS: ACTIVE")
                self.stdout.write(f"      Total Hashtags Tracked: {summary.get('total_hashtags', 0)}")
                self.stdout.write(f"      Total Keywords Tracked: {summary.get('total_keywords', 0)}")
                self.stdout.write(f"      Total Records: {summary.get('total_records', 0)}")
                
                # Get trending
                trending = analyzer.get_trending_hashtags(limit=5)
                if trending:
                    self.stdout.write(f"      Top Trending Hashtags:")
                    for item in trending:
                        self.stdout.write(f"         #{item.item}: velocity={item.velocity:.2f}, score={item.trend_score:.2f}")
            else:
                self.stdout.write(f"   ️  HASHTAG ANALYSIS: NO DATA")
                self.stdout.write(f"      Note: Hashtag analysis runs on stored content")
                self.stdout.write(f"      Run: python manage.py run_pipeline --content-type all")
        
        except Exception as e:
            self.stdout.write(f"   HASHTAG ANALYSIS: ERROR - {e}")
    
    def _verify_topic_modeling(self):
        """Check if topic modeling is being performed"""
        try:
            from myapp.services.content.topic_modeler import get_topic_modeler
            
            modeler = get_topic_modeler()
            summary = modeler.get_topic_summary()
            
            if summary.get('total_topics', 0) > 0:
                self.stdout.write(f"   TOPIC MODELING: ACTIVE")
                self.stdout.write(f"      Number of Topics: {summary.get('total_topics', 0)}")
                self.stdout.write(f"      Trending Topics: {summary.get('trending_topics', 0)}")
                self.stdout.write(f"      Model Loaded: {summary.get('model_loaded', False)}")
                
                # Show trending topics if available
                trending = modeler.get_trending_topics(hours_back=24)
                if trending:
                    self.stdout.write(f"      Top Trending Topics:")
                    for topic in trending[:3]:
                        keywords_str = ', '.join(topic.keywords[:5])
                        self.stdout.write(f"         Topic {topic.topic_id}: {keywords_str}")
                        self.stdout.write(f"            Velocity: {topic.velocity:.2f}x, Documents: {topic.document_count}")
            else:
                self.stdout.write(f"   ️  TOPIC MODELING: NO DATA")
                self.stdout.write(f"      Note: Topic modeling requires manual training")
                self.stdout.write(f"      Run: python manage.py run_pipeline --content-type all")
        
        except Exception as e:
            self.stdout.write(f"   TOPIC MODELING: ERROR - {e}")
    
    def _verify_rag_indexing(self):
        """Check if RAG indexing is working"""
        try:
            from myapp.services.rag.rag_service import get_rag_engine
            
            rag = get_rag_engine()
            stats = rag.get_statistics()
            
            if stats.get('total_documents', 0) > 0:
                self.stdout.write(f"   RAG INDEXING: ACTIVE")
                self.stdout.write(f"      Total Documents Indexed: {stats.get('total_documents', 0)}")
                self.stdout.write(f"      Index Size (MB): {stats.get('index_size_mb', 0):.2f}")
                self.stdout.write(f"      Last Updated: {stats.get('last_updated', 'N/A')}")
                
                # Test retrieval
                self.stdout.write(f"\n      Testing retrieval with query: 'Bitcoin price'...")
                results = rag.retrieve("Bitcoin price", top_k=3)
                
                if results:
                    self.stdout.write(f"      Retrieved {len(results)} documents")
                    for i, doc in enumerate(results[:2], 1):
                        self.stdout.write(f"         {i}. {doc.metadata.get('title', 'N/A')[:50]}...")
                else:
                    self.stdout.write(f"      ️  No results returned")
            else:
                self.stdout.write(f"   ️  RAG INDEXING: NO DOCUMENTS")
                self.stdout.write(f"      Run: python manage.py run_pipeline --content-type all")
        
        except Exception as e:
            self.stdout.write(f"   RAG INDEXING: ERROR - {e}")