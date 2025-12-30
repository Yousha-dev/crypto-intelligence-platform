from django.core.management.base import BaseCommand
from django.conf import settings
import os
import sys

class Command(BaseCommand):
    help = 'Setup and validate the content credibility system'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--setup-mongodb',
            action='store_true',
            help='Setup MongoDB collections and indexes'
        )
        
        parser.add_argument(
            '--validate-config',
            action='store_true',
            help='Validate configuration settings'
        )
        
        parser.add_argument(
            '--skip-tests',
            action='store_true',
            help='Skip basic operation tests'
        )
        
        parser.add_argument(
            '--test-social',
            action='store_true',
            help='Test social media post handling with full metrics'
        )
        
        parser.add_argument(
            '--test-pipeline',
            action='store_true',
            help='Test full pipeline: raw data → analysis → storage'
        )
        
    def handle(self, *args, **options):
        
        if options['validate_config']:
            self.validate_configuration()
        
        if options['setup_mongodb']:
            self.setup_mongodb(skip_tests=options.get('skip_tests', False))
        
        if options['test_social']:
            self.test_social_posts()
        
        if options['test_pipeline']:
            self.test_full_pipeline()

        if not any([options['validate_config'], options['setup_mongodb'], 
                   options['test_social'], options['test_pipeline']]):
            # Run all checks by default
            self.validate_configuration()
            self.setup_mongodb(skip_tests=options.get('skip_tests', False))
    
    def validate_configuration(self):
        """Validate configuration settings"""
        self.stdout.write("Validating configuration...")
         
        # Check MongoDB configuration from Django settings
        mongodb_config = getattr(settings, 'MONGODB_CONFIG', {})
            
        self.stdout.write("MongoDB Configuration:")
        config_items = [
            ('host', 'MONGODB_HOST'),
            ('port', 'MONGODB_PORT'),
            ('database', 'MONGO_DB_NAME'),
            ('username', 'MONGODB_USER'),
            ('password', 'MONGODB_PASSWORD')
        ]
         
        for config_key, env_key in config_items:
            value = mongodb_config.get(config_key) or os.getenv(env_key)
            if value:
                if 'password' in config_key.lower():
                    self.stdout.write(f"  {config_key}: [SET]")
                else:
                    self.stdout.write(f"  {config_key}: {value}")
            else:
                self.stdout.write(self.style.WARNING(f"  ? {config_key}: Not set (using default)"))
        
        # Check Django cache configuration
        if hasattr(settings, 'CACHES') and settings.CACHES:
            self.stdout.write("Django cache configured")
        else:
            self.stdout.write(self.style.WARNING("? Django cache not configured"))
        
        # Check Django settings for news system
        news_settings = [
            'MONGODB_CONFIG',
            'REDIS_URL',
        ]
        
        self.stdout.write("\nNews System Settings:")
        for setting in news_settings:
            if hasattr(settings, setting):
                self.stdout.write(f"  {setting}: configured")
            else:
                self.stdout.write(self.style.WARNING(f"  ? {setting}: not configured"))
        
        # Check fetcher mode
        from myapp.fetchers import USE_MOCK_FETCHERS
        mode = "MOCK" if USE_MOCK_FETCHERS else "REAL"
        self.stdout.write(f"\nFetcher Mode: {mode}")
        
        self.stdout.write(self.style.SUCCESS("Configuration validation complete"))
    
    def setup_mongodb(self, skip_tests=False):
        """Setup MongoDB collections and indexes"""
        self.stdout.write("Setting up MongoDB...")
        
        try:
            from myapp.services.mongo_manager import get_mongo_manager
            
            mongo_manager = get_mongo_manager()
            
            # Test connection
            self.stdout.write("Testing MongoDB connection...")
            stats = mongo_manager.get_statistics()
            self.stdout.write("MongoDB connection successful")
            
            # Show current statistics
            self.stdout.write("\nCurrent database state:")
            if stats:
                for collection, info in stats.items():
                    if isinstance(info, dict) and 'total_documents' in info:
                        self.stdout.write(
                            f"  {collection}: {info['total_documents']} documents, {info['indexes']} indexes"
                        )
                
                if 'recent_activity' in stats:
                    activity = stats['recent_activity']
                    self.stdout.write(f"\nRecent Activity (24h):")
                    self.stdout.write(f"  Articles: {activity.get('articles_24h', 0)}")
                    self.stdout.write(f"  Posts: {activity.get('posts_24h', 0)}")
            else:
                self.stdout.write("  No statistics available (empty database)")
            
            if not skip_tests:
                self._run_basic_tests(mongo_manager)
            
            self.stdout.write(self.style.SUCCESS("\nMongoDB setup complete"))
            
        except ImportError as e:
            self.stdout.write(self.style.ERROR(f"Import error - missing dependency: {e}"))
            self.stdout.write("Run: pip install pymongo")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"MongoDB setup failed: {e}"))
            self._print_troubleshooting()
    
    def _run_basic_tests(self, mongo_manager):
        """Run basic operation tests with RAW data structure"""
        self.stdout.write("\nTesting basic operations...")
        
        # Test news article insert - RAW DATA (as it comes from fetcher)
        test_article = {
            # RAW API fields only (as from CryptoPanic API)
            'id': 'test_setup_article_123',
            'title': 'Setup Test Article - Bitcoin Price Analysis',
            'description': 'This is a test article for setup validation with crypto content',
            'url': 'https://example.com/test-bitcoin-analysis',
            'published_at': '2024-01-01T00:00:00Z',
            'source': {
                'title': 'CryptoPanic Test',
                'domain': 'example.com'
            },
            'votes': {
                'positive': 10,
                'negative': 2,
                'important': 5
            },
            'instruments': [{'code': 'BTC', 'title': 'Bitcoin'}],
            
            # Metadata
            'platform': 'cryptopanic',
            'fetched_at': '2024-01-01T00:00:00Z'
        }
        
        # Process through service to add analysis
        try:
            from myapp.services.content.integrator_service import ContentIntegrationService
            service = ContentIntegrationService()
            
            # Process the raw article (adds credibility, sentiment, etc.)
            result = service.process_news_batch([test_article], 'test_source')
            
            if result.total_processed > 0:
                self.stdout.write("  News article processing successful")
                self.stdout.write(f"    Avg Trust Score: {result.average_trust_score:.2f}")
            
            # Clean up
            mongo_manager.collections['news_articles'].delete_one({'source_id': 'test_setup_article_123'})
            self.stdout.write("  News article cleanup completed")
            
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"  ? News processing test skipped: {e}"))
        
        # Test social post insert with RAW data structure
        self._test_social_post_insert(mongo_manager)
        
    
    def _test_social_post_insert(self, mongo_manager):
        """Test social post insertion with RAW data structure (as from fetchers)"""
        self.stdout.write("\n  Testing social post operations with raw data...")
        
        try:
            from myapp.services.content.integrator_service import ContentIntegrationService
            service = ContentIntegrationService()
            
            # Reddit test post - RAW DATA (as from Reddit API via fetcher)
            reddit_raw = {
                # RAW Reddit API fields
                'id': 'test_reddit_raw_123',
                'title': 'Bitcoin Discussion Test - What do you think about BTC?',
                'selftext': 'This is a test Reddit post about Bitcoin trends and market analysis',
                'url': 'https://reddit.com/r/cryptocurrency/test-post',
                'permalink': 'https://reddit.com/r/cryptocurrency/comments/test123',
                'subreddit': 'cryptocurrency',
                'author': 'crypto_tester',
                'created_utc': 1704067200,
                
                # Raw engagement (from API)
                'score': 500,
                'upvote_ratio': 0.92,
                'num_comments': 120,
                'gilded': 1,
                'total_awards_received': 3,
                
                # Raw post attributes
                'is_self': True,
                'is_video': False,
                'over_18': False,
                'stickied': False,
                'locked': False,
                'link_flair_text': 'Discussion',
                
                # Raw author info (from API)
                'author_info': {
                    'name': 'crypto_tester',
                    'link_karma': 5000,
                    'comment_karma': 15000,
                    'created_utc': 1609459200,
                    'is_gold': False,
                    'is_mod': True,
                    'has_verified_email': True
                },
                
                # Raw subreddit info (from API)
                'subreddit_info': {
                    'name': 'cryptocurrency',
                    'subscribers': 5000000,
                    'created_utc': 1284738600,
                    'public_description': 'The leading community for cryptocurrency discussion',
                    'active_user_count': 10000
                },
                
                # Metadata
                'platform': 'reddit',
                'fetched_at': '2024-01-01T00:00:00Z'
            }
            
            # Process through integrator (adds analysis)
            result = service.process_social_posts_batch([reddit_raw], 'reddit')
            
            if result.total_processed > 0:
                self.stdout.write("    Reddit post processing successful")
                self.stdout.write(f"      Trust Score: {result.average_trust_score:.2f}")
            
            # Clean up
            mongo_manager.collections['social_posts'].delete_one({'source_id': 'test_reddit_raw_123'})
            
            # Twitter test post - RAW DATA (as from Twitter API via fetcher)
            twitter_raw = {
                # RAW Twitter API fields
                'id': 'test_twitter_raw_456',
                'text': 'Bitcoin is showing strong bullish signals. Could we see $100K by EOY? #BTC #Crypto',
                'created_at': '2024-01-01T00:00:00Z',
                'author_id': '123456789',
                'conversation_id': '987654321',
                'lang': 'en',
                'source': 'Twitter Web App',
                
                # Raw engagement metrics (from API)
                'public_metrics': {
                    'like_count': 500,
                    'retweet_count': 150,
                    'reply_count': 45,
                    'quote_count': 20
                },
                
                # Raw entities (from API)
                'entities': {
                    'hashtags': [{'tag': 'BTC'}, {'tag': 'Crypto'}],
                    'cashtags': [{'tag': 'BTC'}]
                },
                
                # Raw user info (from API)
                'user_info': {
                    'id': '123456789',
                    'username': 'crypto_analyst',
                    'name': 'Crypto Analyst',
                    'verified': True,
                    'verified_type': 'blue',
                    'created_at': '2020-01-01T00:00:00Z',
                    'description': 'Crypto analyst and trader',
                    'public_metrics': {
                        'followers_count': 50000,
                        'following_count': 1200,
                        'tweet_count': 5000,
                        'listed_count': 150
                    }
                },
                
                # Convenience fields
                'username': 'crypto_analyst',
                'url': 'https://twitter.com/crypto_analyst/status/test_twitter_raw_456',
                
                # Metadata
                'platform': 'twitter',
                'fetched_at': '2024-01-01T00:00:00Z'
            }
            
            result = service.process_social_posts_batch([twitter_raw], 'twitter')
            
            if result.total_processed > 0:
                self.stdout.write("    Twitter post processing successful")
                self.stdout.write(f"      Trust Score: {result.average_trust_score:.2f}")
            
            mongo_manager.collections['social_posts'].delete_one({'source_id': 'test_twitter_raw_456'})
            
            # YouTube test post - RAW DATA (as from YouTube API via fetcher)
            youtube_raw = {
                # RAW YouTube API fields
                'video_id': 'test_youtube_raw_789',
                'title': 'Bitcoin Technical Analysis - Is $100K Possible?',
                'description': 'Full technical analysis of Bitcoin price action...',
                'published_at': '2024-01-01T00:00:00Z',
                'channel_id': 'UC123456789',
                'channel_title': 'CryptoAnalyst',
                
                # Raw statistics (from API)
                'view_count': 50000,
                'like_count': 2500,
                'comment_count': 300,
                
                # Raw content details
                'duration': 'PT15M30S',
                'duration_seconds': 930,
                'definition': 'hd',
                'caption': True,
                
                # Raw channel info (from API)
                'channel_info': {
                    'channel_id': 'UC123456789',
                    'channel_name': 'CryptoAnalyst',
                    'subscriber_count': 250000,
                    'total_view_count': 50000000,
                    'video_count': 500,
                    'channel_created': '2019-01-01T00:00:00Z'
                },
                
                # Convenience fields
                'url': 'https://youtube.com/watch?v=test_youtube_raw_789',
                'thumbnail': 'https://i.ytimg.com/vi/test/hqdefault.jpg',
                
                # Metadata
                'platform': 'youtube',
                'fetched_at': '2024-01-01T00:00:00Z'
            }
            
            result = service.process_social_posts_batch([youtube_raw], 'youtube')
            
            if result.total_processed > 0:
                self.stdout.write("    YouTube post processing successful")
                self.stdout.write(f"      Trust Score: {result.average_trust_score:.2f}")
            
            mongo_manager.collections['social_posts'].delete_one({'source_id': 'test_youtube_raw_789'})
            
            self.stdout.write("  All social post tests completed")
            
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"  ? Social post test error: {e}"))
            import traceback
            traceback.print_exc()
            
    
    def test_social_posts(self):
        """Test social post handling through the full pipeline"""
        self.stdout.write(self.style.SUCCESS("\n🧪 Testing Social Post Processing Pipeline"))
        self.stdout.write("=" * 60)
        
        try:
            from myapp.services.mongo_manager import get_mongo_manager
            from myapp.services.content.credibility_engine import get_credibility_engine
            from myapp.services.content.integrator_service import ContentIntegrationService
            
            mongo_manager = get_mongo_manager()
            credibility_engine = get_credibility_engine()
            integrator_service = ContentIntegrationService()
            
            # Test 1: Reddit post - RAW → Analysis
            self.stdout.write("\n📌 Test 1: Reddit Post (Raw → Analyzed)")
            reddit_raw = self._create_raw_reddit_post()
            
            self.stdout.write(f"  Raw fields: {list(reddit_raw.keys())}")
            
            # Process through credibility engine
            trust_score = credibility_engine.calculate_trust_score(reddit_raw)
            self.stdout.write(f"  Trust Score: {trust_score.final_score:.2f}")
            self.stdout.write(f"  Source Score: {trust_score.source_score:.2f}")
            self.stdout.write(f"  Confidence: {trust_score.confidence:.2f}")
            
            # Test 2: Twitter post - RAW → Analysis
            self.stdout.write("\n📌 Test 2: Twitter Post (Raw → Analyzed)")
            twitter_raw = self._create_raw_twitter_post()
            
            trust_score = credibility_engine.calculate_trust_score(twitter_raw)
            self.stdout.write(f"  Trust Score: {trust_score.final_score:.2f}")
            self.stdout.write(f"  Source Score: {trust_score.source_score:.2f}")
            
            # Test 3: YouTube post - RAW → Analysis
            self.stdout.write("\n📌 Test 3: YouTube Post (Raw → Analyzed)")
            youtube_raw = self._create_raw_youtube_post()
            
            trust_score = credibility_engine.calculate_trust_score(youtube_raw)
            self.stdout.write(f"  Trust Score: {trust_score.final_score:.2f}")
            self.stdout.write(f"  Source Score: {trust_score.source_score:.2f}")
             
            # Test 4: Batch processing through integrator
            self.stdout.write("\n📌 Test 4: Batch Processing (Integrator Service)")
            batch = [reddit_raw, twitter_raw, youtube_raw]
            
            # Process each platform
            for post in batch:
                platform = post.get('platform', 'unknown')
                result = integrator_service.process_social_posts_batch([post], platform)
                self.stdout.write(f"  {platform}: Processed={result.total_processed}, Avg Score={result.average_trust_score:.2f}")
            
            # Clean up test data
            mongo_manager.collections['social_posts'].delete_many({
                'source_id': {'$in': ['test_reddit_pipeline', 'test_twitter_pipeline', 'test_youtube_pipeline']}
            })
            
            self.stdout.write(self.style.SUCCESS("\nAll social post tests passed!"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"\nTest failed: {e}"))
            import traceback
            traceback.print_exc()
    
    def test_full_pipeline(self):
        """Test the complete pipeline: Fetcher → Integrator → MongoDB"""
        self.stdout.write(self.style.SUCCESS("\n🔄 Testing Full Pipeline: Fetcher → Analysis → Storage"))
        self.stdout.write("=" * 60)
        
        try:
            from myapp.fetchers import (
                fetch_cryptopanic_news,
                fetch_reddit_posts,
                fetch_twitter_posts,
                fetch_youtube_videos,
                USE_MOCK_FETCHERS
            )
            from myapp.services.content.integrator_service import ContentIntegrationService
            from myapp.services.mongo_manager import get_mongo_manager
            
            mode = "MOCK" if USE_MOCK_FETCHERS else "REAL"
            self.stdout.write(f"Fetcher Mode: {mode}")
            
            service = ContentIntegrationService()
            mongo_manager = get_mongo_manager()
            
            # Test News Pipeline
            self.stdout.write("\nNews Pipeline Test")
            self.stdout.write("-" * 40)
            
            news = fetch_cryptopanic_news(max_items=3)
            self.stdout.write(f"  Fetched: {len(news)} raw articles")
            
            if news:
                self.stdout.write(f"  Sample raw fields: {list(news[0].keys())[:8]}...")
                result = service.process_news_batch(news, 'cryptopanic')
                self.stdout.write(f"  Processed: {result.total_processed}")
                self.stdout.write(f"  Avg Trust Score: {result.average_trust_score:.2f}")
            
            # Test Reddit Pipeline
            self.stdout.write("\n📌 Reddit Pipeline Test")
            self.stdout.write("-" * 40)
            
            posts = fetch_reddit_posts(subreddits=['CryptoCurrency'], limit=3)
            self.stdout.write(f"  Fetched: {len(posts)} raw posts")
            
            if posts:
                self.stdout.write(f"  Sample raw fields: {list(posts[0].keys())[:8]}...")
                result = service.process_social_posts_batch(posts, 'reddit')
                self.stdout.write(f"  Processed: {result.total_processed}")
                self.stdout.write(f"  Avg Trust Score: {result.average_trust_score:.2f}")
            
            # Test Twitter Pipeline
            self.stdout.write("\n🐦 Twitter Pipeline Test")
            self.stdout.write("-" * 40)
            
            tweets = fetch_twitter_posts(max_results=3)
            self.stdout.write(f"  Fetched: {len(tweets)} raw tweets")
            
            if tweets:
                self.stdout.write(f"  Sample raw fields: {list(tweets[0].keys())[:8]}...")
                result = service.process_social_posts_batch(tweets, 'twitter')
                self.stdout.write(f"  Processed: {result.total_processed}")
                self.stdout.write(f"  Avg Trust Score: {result.average_trust_score:.2f}")
            
            # Test YouTube Pipeline
            self.stdout.write("\n📺 YouTube Pipeline Test")
            self.stdout.write("-" * 40)
            
            videos = fetch_youtube_videos(max_results=3, min_views=100)
            self.stdout.write(f"  Fetched: {len(videos)} raw videos")
            
            if videos:
                self.stdout.write(f"  Sample raw fields: {list(videos[0].keys())[:8]}...")
                result = service.process_social_posts_batch(videos, 'youtube')
                self.stdout.write(f"  Processed: {result.total_processed}")
                self.stdout.write(f"  Avg Trust Score: {result.average_trust_score:.2f}")
            
            # Show final DB stats
            self.stdout.write("\nDatabase Statistics After Pipeline")
            self.stdout.write("-" * 40)
            stats = mongo_manager.get_statistics()
            if stats:
                for collection, info in stats.items():
                    if isinstance(info, dict) and 'total_documents' in info:
                        self.stdout.write(f"  {collection}: {info['total_documents']} documents")
            
            self.stdout.write(self.style.SUCCESS("\nFull pipeline test completed!"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"\nPipeline test failed: {e}"))
            import traceback
            traceback.print_exc()
    
    def _create_raw_reddit_post(self):
        """Create a RAW Reddit post (as it comes from fetcher)"""
        return {
            # RAW API fields only
            'id': 'test_reddit_pipeline',
            'title': 'Bitcoin showing strong support at $95K - Technical Analysis',
            'selftext': 'Looking at the charts, BTC has strong support levels.',
            'url': 'https://reddit.com/r/cryptocurrency/test',
            'permalink': 'https://reddit.com/r/cryptocurrency/comments/test',
            'subreddit': 'cryptocurrency',
            'author': 'crypto_analyst_pro',
            'created_utc': 1701388800,
            
            # Raw engagement
            'score': 850,
            'upvote_ratio': 0.94,
            'num_comments': 200,
            'gilded': 2,
            'total_awards_received': 5,
            
            # Raw attributes
            'is_self': True,
            'is_video': False,
            'link_flair_text': 'Analysis',
            
            # Raw author info
            'author_info': {
                'name': 'crypto_analyst_pro',
                'link_karma': 15000,
                'comment_karma': 45000,
                'created_utc': 1546300800,
                'is_gold': True,
                'is_mod': True,
                'has_verified_email': True
            },
            
            # Raw subreddit info
            'subreddit_info': {
                'name': 'cryptocurrency',
                'subscribers': 6000000,
                'created_utc': 1284738600,
                'active_user_count': 15000
            },
            
            'platform': 'reddit',
            'fetched_at': '2024-12-01T00:00:00Z'
        } 
     
    def _create_raw_twitter_post(self):
        """Create a RAW Twitter post (as it comes from fetcher)"""
        return {
            # RAW API fields only
            'id': 'test_twitter_pipeline',
            'text': 'Bitcoin just broke through key resistance. Next target: $100K! #BTC',
            'created_at': '2024-12-01T00:00:00Z',
            'author_id': '123456789',
            'lang': 'en',
            'source': 'Twitter Web App',
            
            # Raw metrics
            'public_metrics': {
                'like_count': 2500,
                'retweet_count': 800,
                'reply_count': 150,
                'quote_count': 75
            },
            
            # Raw entities
            'entities': {
                'hashtags': [{'tag': 'BTC'}],
                'cashtags': [{'tag': 'BTC'}]
            },
            
            # Raw user info
            'user_info': {
                'id': '123456789',
                'username': 'verified_analyst',
                'name': 'Verified Analyst',
                'verified': True,
                'verified_type': 'blue',
                'created_at': '2019-01-01T00:00:00Z',
                'public_metrics': {
                    'followers_count': 250000,
                    'following_count': 500,
                    'tweet_count': 15000,
                    'listed_count': 500
                }
            },
            
            'username': 'verified_analyst',
            'url': 'https://twitter.com/verified_analyst/123',
            'platform': 'twitter',
            'fetched_at': '2024-12-01T00:00:00Z'
        }
    
    def _create_raw_youtube_post(self):
        """Create a RAW YouTube post (as it comes from fetcher)"""
        return {
            # RAW API fields only
            'video_id': 'test_youtube_pipeline',
            'title': 'Bitcoin Price Prediction 2025 - Full Technical Analysis',
            'description': 'In this video, we analyze Bitcoin price charts and make predictions...',
            'published_at': '2024-12-01T00:00:00Z',
            'channel_id': 'UC123456789',
            'channel_title': 'CryptoExpert',
            
            # Raw statistics
            'view_count': 150000,
            'like_count': 8000,
            'comment_count': 500,
            
            # Raw content details
            'duration': 'PT30M',
            'duration_seconds': 1800,
            'definition': 'hd',
            'caption': True,
            
            # Raw channel info
            'channel_info': {
                'channel_id': 'UC123456789',
                'channel_name': 'CryptoExpert',
                'subscriber_count': 500000,
                'total_view_count': 100000000,
                'video_count': 800,
                'channel_created': '2018-01-01T00:00:00Z'
            },
            
            'url': 'https://youtube.com/watch?v=test',
            'thumbnail': 'https://i.ytimg.com/vi/test/hqdefault.jpg',
            'platform': 'youtube',
            'fetched_at': '2024-12-01T00:00:00Z'
        }
    
    def _print_troubleshooting(self):
        """Print troubleshooting tips"""
        self.stdout.write("\nTroubleshooting tips:")
        self.stdout.write("1. Make sure MongoDB is running")
        self.stdout.write("2. Check your MongoDB connection settings in .env")
        self.stdout.write("3. Verify MongoDB authentication if enabled")
        self.stdout.write("4. For local development, you can run MongoDB without authentication")
        self.stdout.write("5. Try running with --skip-tests to avoid validation issues")
        self.stdout.write("6. Set USE_MOCK_FETCHERS=true for testing without API calls")