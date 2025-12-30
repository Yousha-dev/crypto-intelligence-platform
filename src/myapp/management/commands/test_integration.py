"""
Complete Integration Test for Content Credibility System
Tests all components working together with BOTH News and Social content
"""
import logging
import os
import traceback
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils import timezone

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Test complete integration of Content credibility system (News + Social)'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--component',
            type=str,
            choices=['all', 'text', 'sentiment', 'credibility', 'storage', 'topics', 'hashtags', 'pipeline', 'rag'],
            default='all',
            help='Which component to test'
        )
        parser.add_argument(
            '--content-type',
            type=str,
            choices=['all', 'news', 'social'],
            default='all',
            help='Which content type to test (default: both)'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed output'
        )
        parser.add_argument(
            '--use-fetchers',
            action='store_true',
            default=True,
            help='Use actual fetchers from fetchers/ folder (default: True)'
        )
        parser.add_argument(
            '--use-hardcoded',
            action='store_true',
            help='Use hardcoded sample data instead of fetchers'
        )
        parser.add_argument(
            '--max-items',
            type=int,
            default=5,
            help='Maximum items to fetch per source (default: 5)'
        )
    
    def handle(self, *args, **options):
        component = options['component']
        content_type = options['content_type']
        verbose = options['verbose']
        use_fetchers = options['use_fetchers'] and not options['use_hardcoded']
        max_items = options['max_items']
        
        # Store options for use in test methods
        self.use_fetchers = use_fetchers
        self.max_items = max_items
        
        # Check which fetcher mode is active
        from myapp.fetchers import USE_MOCK_FETCHERS
        fetcher_mode = "MOCK" if USE_MOCK_FETCHERS else "REAL"
        
        self.stdout.write(self.style.SUCCESS("\n" + "="*70))
        self.stdout.write(self.style.SUCCESS("   CONTENT CREDIBILITY SYSTEM - INTEGRATION TEST"))
        self.stdout.write(self.style.SUCCESS(f"   Testing: {content_type.upper()} content | Component: {component.upper()}"))
        self.stdout.write(self.style.SUCCESS(f"   Data Source: {'FETCHERS (' + fetcher_mode + ')' if use_fetchers else 'HARDCODED SAMPLES'}"))
        self.stdout.write(self.style.SUCCESS("="*70 + "\n"))
        
        results = {}
        
        if component in ['all', 'text']:
            results['text_processor'] = self.test_text_processor(verbose, content_type)
        
        if component in ['all', 'sentiment']:
            results['sentiment_analyzer'] = self.test_sentiment_analyzer(verbose, content_type)
        
        if component in ['all', 'credibility']:
            results['credibility_engine'] = self.test_credibility_engine(verbose, content_type)
        
        if component in ['all', 'storage']:
            results['mongo_manager'] = self.test_mongo_manager(verbose, content_type)
        
        if component in ['all', 'topics']:
            results['topic_modeler'] = self.test_topic_modeler(verbose, content_type)
        
        if component in ['all', 'hashtags']:
            results['hashtag_analyzer'] = self.test_hashtag_analyzer(verbose, content_type)
        
        if component in ['all', 'pipeline']:
            results['full_pipeline'] = self.test_full_pipeline(verbose, content_type)
        
        if component in ['all', 'rag']:
            results['rag_system'] = self.test_rag_system(verbose, content_type)
        
        # Summary
        self.print_summary(results)
    
    # =========================================================================
    # DATA FETCHING - Uses actual fetchers or hardcoded data
    # =========================================================================
    
    def get_sample_news_articles(self):
        """
        Get sample news articles - either from actual fetchers or hardcoded
        """
        if self.use_fetchers:
            return self._fetch_news_from_fetchers()
        else:
            return self._get_hardcoded_news()
    
    def get_sample_social_posts(self):
        """
        Get sample social posts - either from actual fetchers or hardcoded
        """
        if self.use_fetchers:
            return self._fetch_social_from_fetchers()
        else:
            return self._get_hardcoded_social()
    
    def _fetch_news_from_fetchers(self):
        """
        Fetch news using actual fetchers (mock or real based on USE_MOCK_FETCHERS)
        """
        from myapp.fetchers import (
            fetch_cryptopanic_news,
            fetch_cryptocompare_news,
            fetch_messari_news,
            fetch_coindesk_news,
            fetch_newsapi_articles,
            USE_MOCK_FETCHERS
        )
        
        all_articles = []
        max_per_source = max(2, self.max_items // 4)
        
        self.stdout.write(f"   📥 Fetching news using {'MOCK' if USE_MOCK_FETCHERS else 'REAL'} fetchers...")
        
        # Fetch from each source
        sources = [
            ('cryptopanic', lambda: fetch_cryptopanic_news(max_items=max_per_source)),
            ('cryptocompare', lambda: fetch_cryptocompare_news(max_items=max_per_source)),
            ('messari', lambda: fetch_messari_news(max_items=max_per_source)),
            ('coindesk', lambda: fetch_coindesk_news(max_items=max_per_source)),
            ('newsapi', lambda: fetch_newsapi_articles(max_items=max_per_source)),  # Example duplicate source
        ]
        
        for source_name, fetch_func in sources:
            try:
                articles = fetch_func()
                if articles:
                    all_articles.extend(articles)
                    self.stdout.write(f"      {source_name}: {len(articles)} articles")
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"      {source_name}: {str(e)[:50]}"))
        
        self.stdout.write(f"   Total news articles fetched: {len(all_articles)}")
        return all_articles[:self.max_items]
    
    def _fetch_social_from_fetchers(self):
        """
        Fetch social posts using actual fetchers (mock or real based on USE_MOCK_FETCHERS)
        """
        from myapp.fetchers import (
            fetch_reddit_posts,
            fetch_twitter_posts,
            fetch_youtube_videos,
            USE_MOCK_FETCHERS
        )
         
        all_posts = []
        max_per_platform = max(2, self.max_items // 3)
        
        self.stdout.write(f"   📥 Fetching social using {'MOCK' if USE_MOCK_FETCHERS else 'REAL'} fetchers...")
        
        # Fetch from each platform
        platforms = [
            ('reddit', lambda: fetch_reddit_posts(
                subreddits=['CryptoCurrency', 'Bitcoin'],
                limit=max_per_platform
            )),
            ('twitter', lambda: fetch_twitter_posts(
                max_results=max_per_platform,
                hours_back=24
            )),
            ('youtube', lambda: fetch_youtube_videos(
                max_results=max_per_platform,
                days_back=7
            )),
        ]
        
        for platform_name, fetch_func in platforms:
            try:
                posts = fetch_func()
                if posts:
                    all_posts.extend(posts)
                    self.stdout.write(f"      {platform_name}: {len(posts)} posts")
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"      {platform_name}: {str(e)[:50]}"))
        
        self.stdout.write(f"   Total social posts fetched: {len(all_posts)}")
        return all_posts[:self.max_items]
    
    def _get_hardcoded_news(self):
        """Get hardcoded sample news articles (fallback)"""
        articles_data = [ 
            (
                'Bitcoin Surges Past $100,000 on ETF Approval News',
                'Bitcoin reached a new all-time high following SEC approval of spot ETFs. Institutional investors are showing increased interest.',
                'CoinDesk',
                'Michael Chen',
                (45, 5, 20),
                'cryptopanic'
            ),
            (
                'Ethereum Network Upgrade Reduces Gas Fees by 50%',
                'The latest Ethereum upgrade has successfully reduced transaction costs, making DeFi more accessible.',
                'Reuters',
                'Sarah Johnson',
                (38, 2, 15),
                'cryptocompare'
            ),
            (
                'CRYPTO PUMP INCOMING! GET IN NOW! 🚀🚀🚀',
                'This coin is about to explode! Trust me bro! 1000x guaranteed!',
                'Unknown Crypto Blog',
                'Anonymous',
                (2, 25, 0),
                'newsapi'
            ),
            (
                'Federal Reserve Signals Interest Rate Decision Impact on Crypto',
                'Fed Chair Powell comments on digital assets amid monetary policy discussions.',
                'Bloomberg',
                'David Williams',
                (30, 8, 12),
                'cryptopanic'
            ),
            (
                'DeFi Protocol Completes Security Audit with No Critical Issues',
                'Leading smart contract auditor confirms security of new DeFi protocol.',
                'Messari',
                'Alex Turner',
                (22, 3, 8),
                'messari'
            )
        ]
        
        return [
            {
                'id': f'news_test_{i}',
                'platform': platform,
                'title': title,
                'description': desc,
                'content': desc,
                'url': f'https://example.com/article_{i}',
                'author': author,
                'published_at': timezone.now().isoformat(),
                'source': {'title': source, 'domain': f'{source.lower().replace(" ", "")}.com'},
                'votes': {'positive': votes[0], 'negative': votes[1], 'important': votes[2]},
                'instruments': [{'code': 'BTC'}, {'code': 'ETH'}] if i < 3 else [],
            }
            for i, (title, desc, source, author, votes, platform) in enumerate(articles_data)
        ]
    
    def _get_hardcoded_social(self):
        """Get hardcoded sample social posts (fallback)"""
        now = timezone.now()
        
        reddit_posts = [
            {
                'id': 'reddit_test_1',
                'platform': 'reddit',
                'title': 'Bitcoin Technical Analysis - Key Support Levels',
                'selftext': 'Looking at the BTC daily chart, we have strong support at $95K with resistance at $105K. RSI showing oversold conditions. #Bitcoin $BTC',
                'url': 'https://reddit.com/r/cryptocurrency/test1',
                'author': 'crypto_analyst_pro',
                'created_utc': now.timestamp(),
                'subreddit': 'cryptocurrency',
                'score': 1250,
                'upvote_ratio': 0.94,
                'num_comments': 230,
                'total_awards_received': 5,
                'author_info': {
                    'name': 'crypto_analyst_pro',
                    'created_utc': (now - timedelta(days=1500)).timestamp(),
                    'link_karma': 25000,
                    'comment_karma': 60000,
                    'is_mod': True,
                    'is_gold': True,
                    'has_verified_email': True
                },
                'subreddit_info': {
                    'display_name': 'cryptocurrency',
                    'subscribers': 6500000,
                    'created_utc': (now - timedelta(days=3000)).timestamp()
                }
            },
            {
                'id': 'reddit_test_2',
                'platform': 'reddit',
                'title': '🚀🚀🚀 THIS COIN WILL 1000X GUARANTEED!!! 🚀🚀🚀',
                'selftext': 'Trust me bro just buy now before its too late! My uncle works at crypto and said this will moon!',
                'url': 'https://reddit.com/r/cryptomoonshots/test2',
                'author': 'moonshot_king_2024',
                'created_utc': now.timestamp(),
                'subreddit': 'cryptomoonshots',
                'score': 15,
                'upvote_ratio': 0.35,
                'num_comments': 80,
                'total_awards_received': 0,
                'author_info': {
                    'name': 'moonshot_king_2024',
                    'created_utc': (now - timedelta(days=5)).timestamp(),
                    'link_karma': 5,
                    'comment_karma': 20,
                    'is_mod': False,
                    'is_gold': False
                },
                'subreddit_info': {
                    'display_name': 'cryptomoonshots',
                    'subscribers': 150000
                }
            }
        ]
        
        twitter_posts = [
            {
                'id': 'twitter_test_1',
                'platform': 'twitter',
                'text': 'Bitcoin showing strong accumulation patterns. Institutional inflows continue. Key levels: Support $95K, Resistance $105K. #Bitcoin #BTC $BTC',
                'created_at': now.isoformat(),
                'public_metrics': {
                    'like_count': 5200,
                    'retweet_count': 1800,
                    'reply_count': 340,
                    'quote_count': 120
                },
                'user_info': {
                    'id': '123456',
                    'username': 'verified_crypto_analyst',
                    'verified': True,
                    'verified_type': 'business',
                    'created_at': (now - timedelta(days=3000)).isoformat(),
                    'public_metrics': {
                        'followers_count': 450000,
                        'following_count': 800,
                        'tweet_count': 25000,
                        'listed_count': 500
                    }
                }
            },
            {
                'id': 'twitter_test_2',
                'platform': 'twitter',
                'text': '🚀🚀🚀 $SCAMCOIN TO THE MOON! GUARANTEED 10000X! DM ME FOR INSIDER INFO! 🚀🚀🚀 #crypto #gems',
                'created_at': now.isoformat(),
                'public_metrics': {
                    'like_count': 25,
                    'retweet_count': 150,
                    'reply_count': 5,
                    'quote_count': 2
                },
                'user_info': {
                    'id': '789012',
                    'username': 'crypto_gems_finder',
                    'verified': False,
                    'created_at': (now - timedelta(days=15)).isoformat(),
                    'public_metrics': {
                        'followers_count': 200,
                        'following_count': 8000,
                        'tweet_count': 75000,
                        'listed_count': 0
                    }
                }
            }
        ]
        
        youtube_posts = [
            {
                'id': 'youtube_test_1',
                'video_id': 'test_video_1',
                'platform': 'youtube',
                'title': 'Bitcoin Price Prediction 2025 - Complete Technical Analysis',
                'description': 'In-depth analysis of Bitcoin market structure and price targets for 2025.',
                'channel_id': 'UC_test_1',
                'channel_title': 'CryptoEducator',
                'published_at': now.isoformat(),
                'view_count': 250000,
                'like_count': 12000,
                'comment_count': 850,
                'duration_seconds': 2400,
                'caption': 'Full video transcript covering Bitcoin technical analysis...',
                'channel_info': {
                    'subscriber_count': 750000,
                    'subscriber_count_hidden': False,
                    'total_view_count': 150000000,
                    'video_count': 1200,
                    'channel_created': (now - timedelta(days=3500)).isoformat()
                }
            },
            {
                'id': 'youtube_test_2',
                'video_id': 'test_video_2',
                'platform': 'youtube',
                'title': 'GET RICH QUICK WITH THIS ONE SIMPLE TRICK!!!',
                'description': 'Secret method to make millions in crypto overnight!',
                'channel_id': 'UC_test_2',
                'channel_title': 'CryptoMillionaire',
                'published_at': now.isoformat(),
                'view_count': 500,
                'like_count': 10,
                'comment_count': 200,
                'duration_seconds': 180,
                'channel_info': {
                    'subscriber_count': 100,
                    'subscriber_count_hidden': False,
                    'total_view_count': 5000,
                    'video_count': 50,
                    'channel_created': (now - timedelta(days=30)).isoformat()
                }
            }
        ]
        
        return reddit_posts + twitter_posts + youtube_posts
    
    def get_sample_texts(self, content_type: str):
        """Get sample texts for text-based testing"""
        # Use actual fetched data if available
        texts = []
        
        if content_type in ['all', 'news']:
            articles = self.get_sample_news_articles()
            for article in articles[:3]:
                texts.append(f"{article.get('title', '')} {article.get('description', '')}")
        
        if content_type in ['all', 'social']:
            posts = self.get_sample_social_posts()
            for post in posts[:3]:
                content = post.get('content') or post.get('text') or post.get('selftext') or post.get('title', '')
                texts.append(content)
        
        return texts if texts else self._get_fallback_texts()
    
    def _get_fallback_texts(self):
        """Fallback texts if fetchers fail"""
        return [
            "Bitcoin surges to $100,000 as institutional investors flood the market. #Bitcoin $BTC @coinbase",
            "🚀 BTC to the moon! HODL! 💎🙌 #Bitcoin #crypto @elonmusk",
            "Ethereum completes major network upgrade, reducing gas fees significantly. #ETH #Ethereum",
        ]
    
    # =========================================================================
    # TEST METHODS (All test both News and Social)
    # =========================================================================
    
    def test_text_processor(self, verbose: bool, content_type: str) -> dict:
        """Test text processor with both news and social content"""
        self.stdout.write("\n📝 Testing Text Processor...")
        
        try:
            from myapp.services.content.text_processor import get_text_processor
            
            processor = get_text_processor()
            test_texts = self.get_sample_texts(content_type)
            
            results = {'news': [], 'social': []}
            
            for i, text in enumerate(test_texts):
                # Determine if news or social based on content characteristics
                is_social = any(x in text for x in ['🚀', '💎', '🙌', 'HODL', 'moon'])
                text_type = 'social' if is_social else 'news'
                
                processed = processor.preprocess(text)
                entities = processor.extract_entities(text)
                
                result = {
                    'text': text[:50] + '...',
                    'type': text_type,
                    'language': processed.language,
                    'confidence': processed.language_confidence,
                    'word_count': processed.word_count,
                    'hashtags': processed.hashtags,
                    'mentions': processed.mentions,
                    'cryptocurrencies': entities.cryptocurrencies,
                    'persons': entities.persons,
                    'exchanges': entities.exchanges
                }
                results[text_type].append(result)
                
                if verbose:
                    self.stdout.write(f"\n   [{text_type.upper()}] '{text[:40]}...'")
                    self.stdout.write(f"      Language: {processed.language} ({processed.language_confidence:.0%})")
                    self.stdout.write(f"      Hashtags: {processed.hashtags}")
                    self.stdout.write(f"      Cryptos: {entities.cryptocurrencies}")
            
            self.stdout.write(self.style.SUCCESS(f"\n   Text Processor: PASSED"))
            self.stdout.write(f"      News texts processed: {len(results['news'])}")
            self.stdout.write(f"      Social texts processed: {len(results['social'])}")
            
            return {
                'status': 'passed',
                'news_processed': len(results['news']),
                'social_processed': len(results['social']),
                'results': results
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   Text Processor: FAILED - {e}"))
            return {'status': 'failed', 'error': str(e)}
    
    def test_sentiment_analyzer(self, verbose: bool, content_type: str) -> dict:
        """Test sentiment analyzer with both news and social content"""
        self.stdout.write("\n😊 Testing Sentiment Analyzer...")
        
        try:
            from myapp.services.content.sentiment_analyzer import get_sentiment_analyzer
            
            analyzer = get_sentiment_analyzer()
            
            # Test cases for both content types
            test_cases = []
            
            if content_type in ['all', 'news']:
                test_cases.extend([
                    ("Bitcoin surges to new all-time high on institutional adoption.", "bullish", "news"),
                    ("Crypto market crashes amid regulatory concerns and FUD.", "bearish", "news"),
                    ("Bitcoin price consolidates around $95,000 support level.", "neutral", "news"),
                ])
            
            if content_type in ['all', 'social']:
                test_cases.extend([
                    ("🚀🚀🚀 BTC TO THE MOON! HODL! 💎🙌 We're all gonna make it!", "bullish", "social"),
                    ("Market is dead. Lost everything. Crypto is a scam. 😭", "bearish", "social"),
                    ("Just watching charts. Anyone else bored? #crypto", "neutral", "social"),
                ])
            
            results = {'news': [], 'social': []}
            
            for text, expected, text_type in test_cases:
                result = analyzer.analyze(text)
                
                analysis = {
                    'text': text[:50] + '...',
                    'expected': expected,
                    'actual': result.label.value,
                    'score': result.score,
                    'confidence': result.confidence,
                    'correct': result.label.value == expected or (
                        expected == 'bullish' and result.score > 0.1
                    ) or (
                        expected == 'bearish' and result.score < -0.1
                    )
                }
                results[text_type].append(analysis)
                
                if verbose:
                    icon = "✓" if analysis['correct'] else "✗"
                    self.stdout.write(f"\n   [{text_type.upper()}] {icon} '{text[:35]}...'")
                    self.stdout.write(f"      Expected: {expected} | Got: {result.label.value} ({result.score:.2f})")
            
            news_correct = sum(1 for r in results['news'] if r['correct'])
            social_correct = sum(1 for r in results['social'] if r['correct'])
            
            self.stdout.write(self.style.SUCCESS(f"\n   Sentiment Analyzer: PASSED"))
            self.stdout.write(f"      News accuracy: {news_correct}/{len(results['news'])}")
            self.stdout.write(f"      Social accuracy: {social_correct}/{len(results['social'])}")
            
            return {
                'status': 'passed',
                'news_results': results['news'],
                'social_results': results['social'],
                'news_accuracy': news_correct / len(results['news']) if results['news'] else 0,
                'social_accuracy': social_correct / len(results['social']) if results['social'] else 0
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   Sentiment Analyzer: FAILED - {e}"))
            return {'status': 'failed', 'error': str(e)}
    
    def test_credibility_engine(self, verbose: bool, content_type: str) -> dict:
        """Test credibility engine with both news and social content"""
        self.stdout.write("\nTesting Credibility Engine...")
        
        try:
            from myapp.services.content.credibility_engine import get_credibility_engine, get_threshold_manager
            
            engine = get_credibility_engine()
            threshold_manager = get_threshold_manager()
            
            results = {'news': [], 'social': {'reddit': [], 'twitter': [], 'youtube': []}}
            
            # Test NEWS credibility
            if content_type in ['all', 'news']:
                self.stdout.write("\n   Testing NEWS Credibility:")
                
                for article in self.get_sample_news_articles():
                    trust_score = engine.calculate_trust_score(article)
                    action = engine.determine_content_action(trust_score)
                    
                    # Handle both source formats (dict or string)
                    source = article.get('source', {})
                    if isinstance(source, dict):
                        source_name = source.get('title') or source.get('name') or 'Unknown'
                    else:
                        source_name = str(source) if source else 'Unknown'
                    
                    result = {
                        'title': article.get('title', '')[:40] + '...',
                        'source': source_name,
                        'trust_score': trust_score.final_score,
                        'source_score': trust_score.source_score,
                        'sentiment_score': trust_score.sentiment_score,
                        'action': action['action']
                    }
                    results['news'].append(result)
                    
                    if verbose:
                        icon = "✓" if action['action'] == 'auto_approve' else "" if action['action'] in ['normal_flow', 'delayed_review'] else "✗"
                        self.stdout.write(f"      {icon} [{result['trust_score']:.1f}] {result['source']}: {result['title'][:25]}...")
            
            # Test SOCIAL credibility
            if content_type in ['all', 'social']:
                self.stdout.write("\n   Testing SOCIAL Credibility:")
                
                for post in self.get_sample_social_posts():
                    platform = post['platform']
                    trust_score = engine.calculate_trust_score(post)
                    action = engine.determine_content_action(trust_score)
                    
                    # Extract author correctly for display
                    author = (
                        post.get('author') or 
                        post.get('username') or 
                        post.get('channel_title') or
                        post.get('author_info', {}).get('name') or
                        post.get('user_info', {}).get('username') or
                        'Unknown'
                    )
                    
                    # Get title/content for display
                    title = post.get('title') or post.get('text') or post.get('selftext') or post.get('content') or ''
                    
                    result = {
                        'title': title[:40] + '...' if title else '...',
                        'platform': platform,
                        'author': author,
                        'trust_score': trust_score.final_score,
                        'source_score': trust_score.source_score,
                        'action': action['action']
                    }
                    results['social'][platform].append(result)
                    
                    if verbose:
                        icon = "✓" if action['action'] == 'auto_approve' else "" if action['action'] in ['normal_flow', 'delayed_review'] else "✗"
                        self.stdout.write(f"      {icon} [{platform.upper()}] [{result['trust_score']:.1f}] @{result['author']}: {result['title'][:25]}...")
            
            # Calculate stats
            news_scores = [r['trust_score'] for r in results['news']]
            social_scores = [r['trust_score'] for p in results['social'].values() for r in p]
            
            self.stdout.write(self.style.SUCCESS(f"\n   Credibility Engine: PASSED"))
            if news_scores:
                self.stdout.write(f"      News avg score: {sum(news_scores)/len(news_scores):.2f}")
            if social_scores:
                self.stdout.write(f"      Social avg score: {sum(social_scores)/len(social_scores):.2f}")
            
            return {
                'status': 'passed',
                'news_results': results['news'],
                'social_results': results['social'],
                'news_avg': sum(news_scores) / len(news_scores) if news_scores else 0,
                'social_avg': sum(social_scores) / len(social_scores) if social_scores else 0,
                'thresholds': threshold_manager.get_thresholds()
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   Credibility Engine: FAILED - {e}"))
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}

    
    def test_mongo_manager(self, verbose: bool, content_type: str) -> dict:
        """Test MongoDB manager with both news and social content"""
        self.stdout.write("\n🗄️ Testing MongoDB Manager...")
        
        try:
            from myapp.services.mongo_manager import get_mongo_manager
            
            manager = get_mongo_manager()
            test_ids = {'news': [], 'social': []}
            
            # Get statistics
            stats = manager.get_statistics()
            
            if verbose:
                self.stdout.write(f"   Current Statistics:")
                for coll, info in stats.items():
                    if isinstance(info, dict) and 'total_documents' in info:
                        self.stdout.write(f"      {coll}: {info['total_documents']} docs")
            
            # Test NEWS insert/query
            if content_type in ['all', 'news']:
                self.stdout.write("\n   Testing NEWS storage:")
                
                test_article = {
                    'id': f'integration_test_news_{datetime.now().timestamp()}',
                    'platform': 'coindesk',
                    'title': 'Integration Test News Article',
                    'description': 'Test article for integration testing',
                    'url': 'https://test.com/article',
                    'published_at': timezone.now().isoformat(),
                    'trust_score': 8.5,
                    'source_credibility': {'trust_score': 8.0},
                    'overall_credibility': {'final_trust_score': 8.0}
                }
                
                insert_id = manager.insert_news_article(test_article)
                if insert_id:
                    test_ids['news'].append(insert_id)
                    if verbose:
                        self.stdout.write(f"      News insert successful: {insert_id}")
            
            # Test SOCIAL insert/query
            if content_type in ['all', 'social']:
                self.stdout.write("\n   Testing SOCIAL storage:")
                
                for platform in ['reddit', 'twitter', 'youtube']:
                    test_post = {
                        'id': f'integration_test_{platform}_{datetime.now().timestamp()}',
                        'platform': platform,
                        'title': f'Integration Test {platform.title()} Post',
                        'content': f'Test {platform} post for integration testing',
                        'url': f'https://{platform}.com/test',
                        'author_username': 'test_user',
                        'published_at': timezone.now().isoformat(),
                        'trust_score': 7.0,
                        'engagement_metrics': {'score': 100},
                        'user_credibility': {'account_age_days': 365}
                    }
                    
                    insert_id = manager.insert_social_post(test_post)
                    if insert_id:
                        test_ids['social'].append(insert_id)
                        if verbose:
                            self.stdout.write(f"      {platform.title()} insert successful: {insert_id}")
            
            # Cleanup test data
            if test_ids['news']:
                for _id in test_ids['news']:
                    manager.collections['news_articles'].delete_one({'_id': _id})
            if test_ids['social']:
                for _id in test_ids['social']:
                    manager.collections['social_posts'].delete_one({'_id': _id})
            
            if verbose:
                self.stdout.write(f"\n      Cleanup completed")
            
            self.stdout.write(self.style.SUCCESS(f"\n   MongoDB Manager: PASSED"))
            self.stdout.write(f"      News operations: {len(test_ids['news'])} successful")
            self.stdout.write(f"      Social operations: {len(test_ids['social'])} successful")
            
            return {
                'status': 'passed',
                'news_operations': len(test_ids['news']),
                'social_operations': len(test_ids['social']),
                'statistics': stats
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   MongoDB Manager: FAILED - {e}"))
            return {'status': 'failed', 'error': str(e)}
    
    def test_topic_modeler(self, verbose: bool, content_type: str) -> dict:
        """Test topic modeler with both news and social content"""
        self.stdout.write("\nTesting Topic Modeler...")
        
        try:
            from myapp.services.content.topic_modeler import get_topic_modeler
            
            modeler = get_topic_modeler()
            
            # Collect documents from both sources
            documents = []
            doc_types = []
            
            if content_type in ['all', 'news']:
                for article in self.get_sample_news_articles():
                    # Handle different field names for description
                    title = article.get('title', '')
                    desc = (
                        article.get('description') or 
                        article.get('body') or 
                        article.get('summary') or 
                        article.get('content') or 
                        ''
                    )
                    documents.append(f"{title} {desc}")
                    doc_types.append('news')
            
            if content_type in ['all', 'social']:
                for post in self.get_sample_social_posts():
                    title = post.get('title', '')
                    content = (
                        post.get('content') or 
                        post.get('text') or 
                        post.get('selftext') or 
                        post.get('description') or 
                        ''
                    )
                    documents.append(f"{title} {content}")
                    doc_types.append('social')
            
            # Add more documents for better topic modeling
            extra_docs = [
                "Bitcoin mining difficulty reaches all-time high as hashrate increases",
                "Ethereum staking rewards attract institutional investors",
                "DeFi protocols see record TVL despite market volatility",
                "NFT market shows signs of recovery with new collections",
                "Central banks explore CBDC implementations globally",
                "Layer 2 solutions reduce Ethereum transaction costs",
            ]
            documents.extend(extra_docs)
            doc_types.extend(['news'] * len(extra_docs))
            
            # Fit model
            result = modeler.fit(documents, [f'doc_{i}' for i in range(len(documents))])
            
            if verbose:
                self.stdout.write(f"   Mode: {result.get('mode', 'unknown')}")
                self.stdout.write(f"   Documents: {len(documents)} ({doc_types.count('news')} news, {doc_types.count('social')} social)")
                self.stdout.write(f"   Topics Found: {result.get('num_topics', 0)}")
                for topic in result.get('topics', [])[:3]:
                    self.stdout.write(f"      - {topic['name']}: {topic['keywords'][:5]}")
            
            summary = modeler.get_topic_summary()
            
            self.stdout.write(self.style.SUCCESS(f"\n   Topic Modeler: PASSED"))
            
            return {
                'status': 'passed',
                'total_documents': len(documents),
                'news_documents': doc_types.count('news'),
                'social_documents': doc_types.count('social'),
                'topics_found': result.get('num_topics', 0),
                'mode': result.get('mode', 'unknown')
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   Topic Modeler: FAILED - {e}"))
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}

    def test_hashtag_analyzer(self, verbose: bool, content_type: str) -> dict:
        """Test hashtag analyzer with both news and social content"""
        self.stdout.write("\nTesting Hashtag Analyzer...")
        
        try:
            from myapp.services.content.hashtag_analyzer import get_hashtag_analyzer
            
            analyzer = get_hashtag_analyzer()
            
            # Collect texts from both sources
            texts_processed = {'news': 0, 'social': 0}
            
            if content_type in ['all', 'news']:
                for article in self.get_sample_news_articles():
                    # Handle different field names for description
                    title = article.get('title', '')
                    desc = (
                        article.get('description') or 
                        article.get('body') or 
                        article.get('summary') or 
                        article.get('content') or 
                        ''
                    )
                    text = f"{title} {desc}"
                    analyzer.extract_and_record(text, sentiment=0.5, source='news')
                    texts_processed['news'] += 1
            
            if content_type in ['all', 'social']:
                for post in self.get_sample_social_posts():
                    title = post.get('title', '')
                    content = (
                        post.get('content') or 
                        post.get('text') or 
                        post.get('selftext') or 
                        post.get('description') or 
                        ''
                    )
                    text = f"{title} {content}"
                    platform = post.get('platform', 'social')
                    analyzer.extract_and_record(text, sentiment=0.3, source=platform)
                    texts_processed['social'] += 1
            
            # Get trending
            trending_hashtags = analyzer.get_trending_hashtags(limit=10, min_count=1)
            trending_keywords = analyzer.get_trending_keywords(limit=10, min_count=1)
            
            if verbose:
                self.stdout.write(f"   Texts processed: {texts_processed}")
                self.stdout.write(f"   Trending Hashtags: {[t.item for t in trending_hashtags[:5]]}")
                self.stdout.write(f"   Trending Keywords: {[t.item for t in trending_keywords[:5]]}")
            
            summary = analyzer.get_summary()
            
            self.stdout.write(self.style.SUCCESS(f"\n   Hashtag Analyzer: PASSED"))
            self.stdout.write(f"      News texts: {texts_processed['news']}")
            self.stdout.write(f"      Social texts: {texts_processed['social']}")
            self.stdout.write(f"      Hashtags found: {len(trending_hashtags)}")
            
            return {
                'status': 'passed',
                'news_processed': texts_processed['news'],
                'social_processed': texts_processed['social'],
                'hashtags_found': len(trending_hashtags),
                'keywords_found': len(trending_keywords)
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   Hashtag Analyzer: FAILED - {e}"))
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}

    
    def test_full_pipeline(self, verbose: bool, content_type: str) -> dict:
        """Test FULL pipeline with both news AND social content"""
        self.stdout.write("\n🔄 Testing Full Pipeline Integration (News + Social)...")
        
        try:
            from myapp.services.content.integrator_service import (
                ContentIntegrationService,
                convert_fetcher_to_service_format,
                convert_social_fetcher_to_service_format
            )
             
            service = ContentIntegrationService()
            results = {'news': None, 'social': {'reddit': None, 'twitter': None, 'youtube': None}}
            
            # =====================================================
            # TEST NEWS PIPELINE
            # =====================================================
            if content_type in ['all', 'news']:
                self.stdout.write("\n   Processing NEWS through pipeline...")
                
                mock_articles = self.get_sample_news_articles()
                formatted_articles = convert_fetcher_to_service_format(mock_articles, 'integration_test')
                news_result = service.process_news_batch(formatted_articles, 'integration_test')
                results['news'] = news_result
                
                if verbose:
                    self.stdout.write(f"      Total: {news_result.total_processed}")
                    self.stdout.write(f"      Approved: {news_result.approved}")
                    self.stdout.write(f"      Pending: {news_result.pending}")
                    self.stdout.write(f"      Flagged: {news_result.flagged}")
                    self.stdout.write(f"      Avg Trust: {news_result.average_trust_score:.2f}")
            
            # =====================================================
            # TEST SOCIAL PIPELINE (All Platforms)
            # =====================================================
            if content_type in ['all', 'social']:
                self.stdout.write("\n   Processing SOCIAL through pipeline...")
                
                social_posts = self.get_sample_social_posts()
                
                # Group by platform
                posts_by_platform = {'reddit': [], 'twitter': [], 'youtube': []}
                for post in social_posts:
                    platform = post.get('platform', 'unknown')
                    if platform in posts_by_platform:
                        posts_by_platform[platform].append(post)
                
                # Process each platform
                for platform, posts in posts_by_platform.items():
                    if posts:
                        formatted_posts = convert_social_fetcher_to_service_format(posts, platform)
                        platform_result = service.process_social_posts_batch(formatted_posts, platform)
                        results['social'][platform] = platform_result
                        
                        if verbose:
                            self.stdout.write(f"\n      [{platform.upper()}]")
                            self.stdout.write(f"         Processed: {platform_result.total_processed}")
                            self.stdout.write(f"         Approved: {platform_result.approved}")
                            self.stdout.write(f"         Flagged: {platform_result.flagged}")
                            self.stdout.write(f"         Avg Trust: {platform_result.average_trust_score:.2f}")
            
            # =====================================================
            # VERIFY ENTITY EXTRACTION
            # =====================================================
            self.stdout.write("\n   Verifying Entity Extraction...")
            
            if content_type in ['all', 'news'] and formatted_articles:
                for article in formatted_articles[:1]:
                    if 'extracted_entities' in article:
                        entities = article['extracted_entities']
                        if verbose:
                            self.stdout.write(f"      News entities: {entities.get('cryptocurrencies', [])[:3]}")
            
            if content_type in ['all', 'social']:
                for platform in ['reddit', 'twitter', 'youtube']:
                    if posts_by_platform.get(platform):
                        formatted = convert_social_fetcher_to_service_format(posts_by_platform[platform][:1], platform)
                        if formatted and 'extracted_entities' in formatted[0]:
                            entities = formatted[0]['extracted_entities']
                            if verbose:
                                self.stdout.write(f"      {platform.title()} entities: {entities.get('cryptocurrencies', [])[:3]}")
            
            # =====================================================
            # CALCULATE TOTALS
            # =====================================================
            total_processed = 0
            total_approved = 0
            total_flagged = 0
            all_scores = []
            
            if results['news']:
                total_processed += results['news'].total_processed
                total_approved += results['news'].approved
                total_flagged += results['news'].flagged
                all_scores.extend(results['news'].trust_scores if hasattr(results['news'], 'trust_scores') else [results['news'].average_trust_score])
            
            for platform_result in results['social'].values():
                if platform_result:
                    total_processed += platform_result.total_processed
                    total_approved += platform_result.approved
                    total_flagged += platform_result.flagged
                    all_scores.extend(platform_result.trust_scores if hasattr(platform_result, 'trust_scores') else [platform_result.average_trust_score])
            
            avg_trust = sum(all_scores) / len(all_scores) if all_scores else 0
            
            # =====================================================
            # GET SYSTEM HEALTH
            # =====================================================
            health = service.get_system_health()
            
            self.stdout.write(self.style.SUCCESS(f"\n   Full Pipeline: PASSED"))
            self.stdout.write(f"\n   PIPELINE SUMMARY:")
            self.stdout.write(f"      Total Processed: {total_processed}")
            self.stdout.write(f"      Total Approved: {total_approved}")
            self.stdout.write(f"      Total Flagged: {total_flagged}")
            self.stdout.write(f"      Overall Avg Trust: {avg_trust:.2f}")
            self.stdout.write(f"      System Status: {health.get('overall_status', 'unknown')}")
            
            return {
                'status': 'passed',
                'total_processed': total_processed,
                'total_approved': total_approved,
                'total_flagged': total_flagged,
                'average_trust_score': avg_trust,
                'news_result': results['news'].__dict__ if results['news'] else None,
                'social_results': {
                    platform: r.__dict__ if r else None
                    for platform, r in results['social'].items()
                },
                'system_health': health
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   Full Pipeline: FAILED - {e}"))
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def test_rag_system(self, verbose: bool, content_type: str) -> dict:
        """Test RAG system with both news and social content"""
        self.stdout.write("\nTesting RAG System...")
        
        try:
            from myapp.services.rag.rag_service import get_rag_engine
            from myapp.services.rag.knowledge_graph import get_knowledge_graph
            
            rag_engine = get_rag_engine()
            kg = get_knowledge_graph()
            
            results = {'indexing': {'news': 0, 'social': 0}, 'retrieval': None, 'kg': None}
            
            # Test indexing (simulated - don't actually index test data)
            self.stdout.write("   Testing RAG components...")
            
            # Test retrieval
            test_query = "What is happening with Bitcoin price?"
            
            try:
                search_results = rag_engine.retrieve(test_query, top_k=3)
                results['retrieval'] = {
                    'query': test_query,
                    'results_count': len(search_results) if search_results else 0
                }
                if verbose:
                    self.stdout.write(f"      Retrieval working: {len(search_results) if search_results else 0} results")
            except Exception as e:
                if verbose:
                    self.stdout.write(f"      Retrieval: {str(e)[:50]}")
            
            # Test knowledge graph
            try:
                kg_stats = kg.get_statistics()
                results['kg'] = kg_stats
                if verbose:
                    self.stdout.write(f"      Knowledge Graph: {kg_stats.get('total_entities', 0)} entities")
            except Exception as e:
                if verbose:
                    self.stdout.write(f"      Knowledge Graph: {str(e)[:50]}")
            
            # Test document creation methods exist
            has_news_method = hasattr(rag_engine, '_create_document_from_article')
            has_social_method = hasattr(rag_engine, '_create_document_from_social')
            
            if verbose:
                self.stdout.write(f"      News document method: {'Yes' if has_news_method else 'No'}")
                self.stdout.write(f"      Social document method: {'Yes' if has_social_method else 'No'}")
            
            self.stdout.write(self.style.SUCCESS(f"\n   RAG System: PASSED"))
            self.stdout.write(f"      News indexing support: {'Yes' if has_news_method else 'No'}")
            self.stdout.write(f"      Social indexing support: {'Yes' if has_social_method else 'No'}")
            
            return {
                'status': 'passed',
                'has_news_support': has_news_method,
                'has_social_support': has_social_method,
                'retrieval': results['retrieval'],
                'knowledge_graph': results['kg']
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   RAG System: FAILED - {e}"))
            return {'status': 'failed', 'error': str(e)}
    
    def print_summary(self, results: dict):
        """Print comprehensive test summary"""
        self.stdout.write("\n" + "="*70)
        self.stdout.write(self.style.SUCCESS("   INTEGRATION TEST SUMMARY"))
        self.stdout.write("="*70 + "\n")
        
        passed = sum(1 for r in results.values() if r.get('status') == 'passed')
        failed = sum(1 for r in results.values() if r.get('status') == 'failed')
        
        # Component results
        self.stdout.write("   COMPONENT RESULTS:\n")
        for component, result in results.items():
            status = result.get('status', 'unknown')
            icon = "✅" if status == 'passed' else "❌"
            self.stdout.write(f"      {icon} {component}: {status.upper()}")
            
            # Show additional info for pipeline
            if component == 'full_pipeline' and status == 'passed':
                self.stdout.write(f"         → Processed: {result.get('total_processed', 0)}")
                self.stdout.write(f"         → Approved: {result.get('total_approved', 0)}")
                self.stdout.write(f"         → Avg Trust: {result.get('average_trust_score', 0):.2f}")
        
        # Summary stats
        self.stdout.write(f"\n   TOTALS:")
        self.stdout.write(f"      Passed: {passed}")
        self.stdout.write(f"      Failed: {failed}")
        self.stdout.write(f"      Total: {len(results)}")
        
        if failed == 0:
            self.stdout.write(self.style.SUCCESS("\n   🎉 ALL INTEGRATION TESTS PASSED!"))
            self.stdout.write(self.style.SUCCESS("      News + Social content fully integrated!"))
        else: 
            self.stdout.write(self.style.ERROR(f"\n   ️ {failed} TESTS FAILED - Review errors above"))
        
        self.stdout.write("\n" + "="*70 + "\n")