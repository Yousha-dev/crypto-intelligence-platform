from django.core.management.base import BaseCommand
from django.utils import timezone
from myapp.tasks.content_tasks import (
    fetch_and_process_news_articles,
    fetch_and_process_social_posts,
    comprehensive_content_update,
    scheduled_content_fetch
)
import json


class Command(BaseCommand):
    help = 'Fetch content from external sources and process through credibility pipeline'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--source',
            type=str,
            choices=['news', 'social', 'all'],
            default='all',
            help='Which type of sources to fetch from'
        ) 
        
        parser.add_argument(
            '--platforms',
            type=str,
            help='Comma-separated list of specific platforms/sources'
        )
        
        parser.add_argument(
            '--sync',
            action='store_true',
            help='Process synchronously instead of using Celery tasks'
        ) 
        
        parser.add_argument(
            '--max-items',
            type=int,
            default=30,
            help='Maximum items per source'
        )
        
        parser.add_argument(
            '--comprehensive',
            action='store_true',
            help='Run comprehensive update with all sources'
        )
        
        parser.add_argument(
            '--scheduled',
            action='store_true',
            help='Run scheduled update (lighter version)'
        )
        
        parser.add_argument(
            '--stats',
            action='store_true',
            help='Show social media statistics'
        )
        
        parser.add_argument(
            '--trending',
            action='store_true',
            help='Show trending social content'
        )
        
        parser.add_argument(
            '--platform-stats',
            type=str,
            choices=['reddit', 'twitter', 'youtube', 'all'],
            help='Show detailed statistics for specific platform'
        )
    
    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('🚀 Starting Content Fetching Process')
        )
        
        if options['stats']:
            self.show_social_statistics()
        elif options['trending']:
            self.show_trending_content()
        elif options.get('platform_stats'):
            self.show_platform_statistics(options['platform_stats'])
        elif options['comprehensive']:
            self.run_comprehensive_update()
        elif options['scheduled']:
            self.run_scheduled_update()
        else:
            self.run_custom_fetch(options)
    
    def show_social_statistics(self):
        """Show social media statistics"""
        self.stdout.write("\nSocial Media Statistics (Last 24 Hours)")
        self.stdout.write("=" * 60)
        
        try:
            from myapp.services.mongo_manager import get_mongo_manager
            
            mongo_manager = get_mongo_manager()
            stats = mongo_manager.get_social_statistics(hours_back=24)
            
            if not stats:
                self.stdout.write("No statistics available")
                return
            
            self.stdout.write(f"\nTotal Posts: {stats.get('total_posts', 0)}")
            
            # Platform breakdown
            self.stdout.write("\nPlatform Breakdown:")
            for platform, data in stats.get('platform_breakdown', {}).items():
                self.stdout.write(f"  {platform.upper()}:")
                self.stdout.write(f"    Posts: {data.get('count', 0)}")
                self.stdout.write(f"    Avg Trust Score: {data.get('avg_trust_score', 0):.2f}")
                self.stdout.write(f"    Avg Engagement: {data.get('avg_engagement', 0):.0f}")
            
            # Status breakdown
            self.stdout.write("\nStatus Breakdown:")
            for status, count in stats.get('status_breakdown', {}).items():
                emoji = {'approved': '✅', 'pending': '⏳', 'flagged': '🚩', 'rejected': '❌'}.get(status, '•')
                self.stdout.write(f"  {emoji} {status}: {count}")
            
            self.stdout.write(f"\n⏰ Generated at: {stats.get('generated_at', 'N/A')}")
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error getting statistics: {e}"))
    
    def show_platform_statistics(self, platform: str):
        """Show detailed statistics for a specific platform"""
        self.stdout.write(f"\nDetailed Statistics for {platform.upper()}")
        self.stdout.write("=" * 60)
        
        try:
            from myapp.services.mongo_manager import get_mongo_manager
            
            mongo_manager = get_mongo_manager()
            
            platforms = [platform] if platform != 'all' else ['reddit', 'twitter', 'youtube']
            
            for plat in platforms:
                posts = mongo_manager.get_social_posts(
                    platform=plat,
                    hours_back=24,
                    limit=100
                )
                
                if not posts:
                    self.stdout.write(f"\n{plat.upper()}: No posts found")
                    continue
                
                self.stdout.write(f"\n{plat.upper()} ({len(posts)} posts)")
                self.stdout.write("-" * 40)
                
                # Calculate metrics
                trust_scores = [p.get('trust_score', 0) for p in posts]
                avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0
                
                # Platform-specific metrics
                if plat == 'reddit':
                    self._show_reddit_stats(posts)
                elif plat == 'twitter':
                    self._show_twitter_stats(posts)
                elif plat == 'youtube':
                    self._show_youtube_stats(posts)
                
                self.stdout.write(f"  Average Trust Score: {avg_trust:.2f}")
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {e}"))
    
    def _show_reddit_stats(self, posts):
        """Show Reddit-specific statistics"""
        total_score = 0
        total_comments = 0
        total_awards = 0
        
        for post in posts:
            engagement = post.get('engagement_metrics', {})
            total_score += engagement.get('score', 0)
            total_comments += engagement.get('num_comments', 0)
            total_awards += engagement.get('total_awards_received', 0)
        
        self.stdout.write(f"  Total Score (upvotes): {total_score:,}")
        self.stdout.write(f"  Total Comments: {total_comments:,}")
        self.stdout.write(f"  Total Awards: {total_awards}")
    
    def _show_twitter_stats(self, posts):
        """Show Twitter-specific statistics"""
        total_likes = 0
        total_retweets = 0
        total_replies = 0
        verified_posts = 0
        
        for post in posts:
            engagement = post.get('engagement_metrics', {})
            user_cred = post.get('user_credibility', {})
            
            total_likes += engagement.get('like_count', 0)
            total_retweets += engagement.get('retweet_count', 0)
            total_replies += engagement.get('reply_count', 0)
            
            if user_cred.get('verified'):
                verified_posts += 1
        
        self.stdout.write(f"  Total Likes: {total_likes:,}")
        self.stdout.write(f"  Total Retweets: {total_retweets:,}")
        self.stdout.write(f"  Total Replies: {total_replies:,}")
        self.stdout.write(f"  Posts by Verified Users: {verified_posts}")
    
    def _show_youtube_stats(self, posts):
        """Show YouTube-specific statistics"""
        total_views = 0
        total_likes = 0
        total_comments = 0
        
        for post in posts:
            engagement = post.get('engagement_metrics', {})
            total_views += engagement.get('view_count', 0)
            total_likes += engagement.get('like_count', 0)
            total_comments += engagement.get('comment_count', 0)
        
        self.stdout.write(f"  Total Views: {total_views:,}")
        self.stdout.write(f"  Total Likes: {total_likes:,}")
        self.stdout.write(f"  Total Comments: {total_comments:,}")
    
    def show_trending_content(self):
        """Show trending social content"""
        self.stdout.write("\n🔥 Trending Social Content (Last 24 Hours)")
        self.stdout.write("=" * 60)
        
        try:
            from myapp.services.mongo_manager import get_mongo_manager
            
            mongo_manager = get_mongo_manager()
            trending = mongo_manager.get_trending_social_content(hours_back=24, limit=10)
            
            if not trending:
                self.stdout.write("No trending content found")
                return
            
            for i, post in enumerate(trending, 1):
                platform = post.get('platform', 'unknown').upper()
                title = post.get('title', post.get('content', ''))[:60]
                trust = post.get('trust_score', 0)
                engagement = post.get('engagement_score', post.get('engagement_metrics', {}).get('total_engagement', 0))
                
                self.stdout.write(f"\n{i}. [{platform}] {title}...")
                self.stdout.write(f"   Trust: {trust:.1f} | Engagement: {engagement:.0f}")
                self.stdout.write(f"   Author: @{post.get('author_username', post.get('username', 'unknown'))}")
                
                self._show_post_metrics(post, post.get('platform', 'unknown'))
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error getting trending content: {e}"))
    
    def _show_post_metrics(self, post, platform):
        """Show platform-specific metrics for a single post"""
        engagement = post.get('engagement_metrics', {})
        
        if platform == 'reddit':
            score = engagement.get('score', 0)
            ratio = engagement.get('upvote_ratio', 0)
            comments = engagement.get('num_comments', 0)
            self.stdout.write(f"   Score: {score} | Ratio: {ratio:.0%} | Comments: {comments}")
            
        elif platform == 'twitter':
            likes = engagement.get('like_count', 0)
            retweets = engagement.get('retweet_count', 0)
            self.stdout.write(f"   Likes: {likes:,} | RTs: {retweets:,}")
            
        elif platform == 'youtube':
            views = engagement.get('view_count', 0)
            likes = engagement.get('like_count', 0)
            self.stdout.write(f"   Views: {views:,} | Likes: {likes:,}")
    
    def run_comprehensive_update(self):
        """Run comprehensive content update"""
        self.stdout.write("Running comprehensive content update...")
        
        task = comprehensive_content_update.delay()
        self.stdout.write(f"Task queued with ID: {task.id}")
        
        try:
            result = task.get(timeout=1800)
            self.stdout.write(self.style.SUCCESS(f"Comprehensive update completed"))
            self.display_results(result)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {e}"))
     
    def run_scheduled_update(self):
        """Run scheduled content update"""
        self.stdout.write("Running scheduled content update...")
        
        task = scheduled_content_fetch.delay()
        self.stdout.write(f"Scheduled update queued with ID: {task.id}")
        
        result = {
            'status': 'queued',
            'task_id': task.id,
            'message': 'Task has been queued for processing'
        }
        self.display_results(result)
    
    def run_custom_fetch(self, options):
        """Run custom fetch based on options"""
        source_type = options['source']
        platforms = options['platforms'].split(',') if options['platforms'] else None
        sync = options['sync']
        max_items = options['max_items']
        
        results = {}
        
        if source_type in ['news', 'all']:
            news_config = self.build_news_config(platforms, max_items)
            self.stdout.write("Fetching news articles...")
              
            if sync:
                news_results = self.fetch_news_sync(news_config)
                results['news'] = news_results
            else:
                task = fetch_and_process_news_articles.delay(news_config, async_processing=True)
                results['news'] = {'task_id': task.id, 'status': 'queued'}
          
        if source_type in ['social', 'all']:
            social_config = self.build_social_config(platforms, max_items)
            self.stdout.write("Fetching social media posts...")
            
            if sync:
                social_results = self.fetch_social_sync(social_config)
                results['social'] = social_results
            else:
                task = fetch_and_process_social_posts.delay(social_config, async_processing=True)
                results['social'] = {'task_id': task.id, 'status': 'queued'}
        
        self.display_results(results)
    
    def build_news_config(self, platforms, max_items):
        """Build configuration for news sources"""
        all_sources = ['cryptopanic', 'cryptocompare', 'newsapi', 'messari', 'coindesk']
        selected_sources = platforms if platforms else all_sources
        
        config = {}
        for source in selected_sources:
            if source in all_sources:
                config[source] = {
                    'enabled': True,
                    'max_items': max_items
                }
                
                if source == 'cryptopanic':
                    config[source]['filter_type'] = 'important'
                elif source == 'cryptocompare':
                    config[source]['categories'] = 'BTC,ETH'
                elif source == 'newsapi':
                    config[source]['query'] = 'cryptocurrency bitcoin ethereum'
        
        return config
    
    def build_social_config(self, platforms, max_items):
        """Build configuration for social platforms"""
        all_platforms = ['twitter', 'reddit', 'youtube']
        selected_platforms = platforms if platforms else all_platforms
        
        config = {}
        for platform in selected_platforms:
            if platform in all_platforms:
                config[platform] = {
                    'enabled': True,
                    'max_items': max_items
                }
                
                if platform == 'twitter':
                    config[platform]['hours_back'] = 12
                elif platform == 'reddit':
                    config[platform]['subreddits'] = ['CryptoCurrency', 'Bitcoin', 'ethereum']
                elif platform == 'youtube':
                    config[platform]['days_back'] = 3
                    config[platform]['min_views'] = 1000
        
        return config
    
    def fetch_news_sync(self, config):
        """Fetch news synchronously - RAW DATA from fetchers, analysis in service"""
        from myapp.fetchers import (
            fetch_cryptopanic_news,
            fetch_cryptocompare_news,
            fetch_newsapi_articles,
            fetch_messari_news,
            fetch_coindesk_news
        )
        from myapp.services.content.integrator_service import ContentIntegrationService
        
        service = ContentIntegrationService()
        results = {}
        
        for source_name, source_config in config.items():
            if not source_config.get('enabled'):
                continue
                
            try:
                self.stdout.write(f"  Fetching from {source_name}...")
                articles = []
                
                # Fetch RAW data only
                if source_name == 'cryptopanic':
                    articles = fetch_cryptopanic_news(
                        filter_type=source_config.get('filter_type', 'important'),
                        max_items=source_config.get('max_items', 30)
                    )
                elif source_name == 'cryptocompare':
                    articles = fetch_cryptocompare_news(
                        categories=source_config.get('categories', 'BTC,ETH'),
                        max_items=source_config.get('max_items', 30)
                    )
                elif source_name == 'newsapi':
                    articles = fetch_newsapi_articles(
                        query=source_config.get('query', 'cryptocurrency'),
                        max_items=source_config.get('max_items', 30)
                    )
                elif source_name == 'messari':
                    articles = fetch_messari_news(
                        max_items=source_config.get('max_items', 30)
                    )
                elif source_name == 'coindesk':
                    articles = fetch_coindesk_news(
                        max_items=source_config.get('max_items', 30)
                    )
                 
                if articles:
                    self.stdout.write(f"    📥 Fetched {len(articles)} raw articles")
                    
                    # Process through integrator service (analysis happens here)
                    result = service.process_news_batch(articles, source_name)
                    results[source_name] = result.__dict__
                    self.stdout.write(f"    Processed {result.total_processed} articles")
                else:
                    results[source_name] = {'status': 'no_articles'}
                    self.stdout.write(f"    ️ No articles found")
                    
            except Exception as e:
                results[source_name] = {'status': 'error', 'error': str(e)}
                self.stdout.write(self.style.ERROR(f"    Error: {e}"))
        
        return results
    
    def fetch_social_sync(self, config):
        """Fetch social posts synchronously - RAW DATA from fetchers, analysis in service"""
        from myapp.fetchers import (
            fetch_twitter_posts,
            fetch_reddit_posts,
            fetch_youtube_videos
        )
        from myapp.services.content.integrator_service import ContentIntegrationService
        
        service = ContentIntegrationService()
        results = {}
        
        for platform_name, platform_config in config.items():
            if not platform_config.get('enabled'):
                continue
                
            try:
                self.stdout.write(f"  Fetching from {platform_name}...")
                posts = []
                
                # Fetch RAW data only
                if platform_name == 'twitter':
                    posts = fetch_twitter_posts(
                        max_results=platform_config.get('max_items', 30),
                        hours_back=platform_config.get('hours_back', 12)
                    )
                elif platform_name == 'reddit':
                    posts = fetch_reddit_posts(
                        subreddits=platform_config.get('subreddits', ['CryptoCurrency', 'Bitcoin', 'ethereum']),
                        limit=platform_config.get('max_items', 30)
                    )
                elif platform_name == 'youtube':
                    posts = fetch_youtube_videos(
                        max_results=platform_config.get('max_items', 20),
                        days_back=platform_config.get('days_back', 3),
                        min_views=platform_config.get('min_views', 1000)
                    )
                
                if posts:
                    self.stdout.write(f"    📥 Fetched {len(posts)} raw posts")
                    
                    # Display sample raw data
                    self._display_raw_sample(posts[0], platform_name)
                    
                    # Process through integrator service (analysis happens here)
                    result = service.process_social_posts_batch(posts, platform_name)
                    results[platform_name] = result.__dict__
                    self.stdout.write(f"    Processed {result.total_processed} posts")
                else:
                    results[platform_name] = {'status': 'no_posts'}
                    self.stdout.write(f"    ️ No posts found")
                    
            except Exception as e:
                results[platform_name] = {'status': 'error', 'error': str(e)}
                self.stdout.write(self.style.ERROR(f"    Error: {e}"))
                import traceback
                traceback.print_exc()
        
        return results
    
    def _display_raw_sample(self, post, platform):
        """Display sample raw data from fetcher"""
        self.stdout.write(f"\n    Sample Raw {platform.title()} Data:")
        
        if platform == 'reddit':
            self.stdout.write(f"       Title: {post.get('title', '')[:50]}...")
            self.stdout.write(f"       Author: u/{post.get('author', 'unknown')}")
            self.stdout.write(f"       Score: {post.get('score', 'N/A')}")
            self.stdout.write(f"       Upvote Ratio: {post.get('upvote_ratio', 'N/A')}")
            self.stdout.write(f"       Comments: {post.get('num_comments', 'N/A')}")
            self.stdout.write(f"       Subreddit: r/{post.get('subreddit', 'N/A')}")
            
            # Show author info if available
            author_info = post.get('author_info', {})
            if author_info:
                self.stdout.write(f"       Author Karma: {author_info.get('link_karma', 0) + author_info.get('comment_karma', 0):,}")
            
        elif platform == 'twitter':
            self.stdout.write(f"       Text: {post.get('text', '')[:50]}...")
            self.stdout.write(f"       Author: @{post.get('username', 'unknown')}")
            
            metrics = post.get('public_metrics', {})
            self.stdout.write(f"       Likes: {metrics.get('like_count', 'N/A')}")
            self.stdout.write(f"       Retweets: {metrics.get('retweet_count', 'N/A')}")
            self.stdout.write(f"       Replies: {metrics.get('reply_count', 'N/A')}")
            
            user_info = post.get('user_info', {})
            if user_info:
                self.stdout.write(f"       Verified: {user_info.get('verified', False)}")
                user_metrics = user_info.get('public_metrics', {})
                self.stdout.write(f"       Followers: {user_metrics.get('followers_count', 'N/A')}")
            
        elif platform == 'youtube':
            self.stdout.write(f"       Title: {post.get('title', '')[:50]}...")
            self.stdout.write(f"       Channel: {post.get('channel_title', 'unknown')}")
            self.stdout.write(f"       Views: {post.get('view_count', 'N/A'):,}" if isinstance(post.get('view_count'), int) else f"       Views: N/A")
            self.stdout.write(f"       Likes: {post.get('like_count', 'N/A')}")
            self.stdout.write(f"       Duration: {post.get('duration_seconds', 'N/A')}s")
            
            channel_info = post.get('channel_info', {})
            if channel_info:
                self.stdout.write(f"       Channel Subscribers: {channel_info.get('subscriber_count', 'N/A'):,}" if isinstance(channel_info.get('subscriber_count'), int) else f"       Channel Subscribers: N/A")
        
        self.stdout.write("")
    
    def display_results(self, results):
        """Display processing results"""
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("Processing Results")
        self.stdout.write("=" * 60)
        
        if isinstance(results, dict):
            for source, data in results.items():
                self.stdout.write(f"\n{source.upper()}:")
                if isinstance(data, dict):
                    if data.get('status') == 'queued':
                        self.stdout.write(f"  Status: Queued (Task ID: {data.get('task_id')})")
                    elif data.get('status') == 'error':
                        self.stdout.write(self.style.ERROR(f"  Error: {data.get('error')}"))
                    elif data.get('status') in ['no_articles', 'no_posts']:
                        self.stdout.write(self.style.WARNING(f"  No content found"))
                    else:
                        self.stdout.write(f"  Total Processed: {data.get('total_processed', 'N/A')}")
                        self.stdout.write(f"  Approved: {data.get('approved', 'N/A')}")
                        self.stdout.write(f"  Pending: {data.get('pending', 'N/A')}")
                        self.stdout.write(f"  Flagged: {data.get('flagged', 'N/A')}")
                        self.stdout.write(f"  Errors: {data.get('errors', 'N/A')}")
                        self.stdout.write(f"  Avg Trust Score: {data.get('average_trust_score', 0):.2f}")
                        self.stdout.write(f"  Processing Time: {data.get('processing_time_seconds', 0):.2f}s")
                else:
                    self.stdout.write(f"  {data}")
        else:
            self.stdout.write(str(results))
          
        self.stdout.write("\n" + "=" * 60) 
        self.stdout.write(self.style.SUCCESS("Fetch process completed"))