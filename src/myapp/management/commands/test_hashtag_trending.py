from django.core.management.base import BaseCommand
from datetime import datetime
from django.utils import timezone
from myapp.services.content.hashtag_analyzer import get_hashtag_analyzer


class Command(BaseCommand):
    help = 'Test hashtag trending functionality'
    
    def add_arguments(self, parser):
        parser.add_argument('--populate', action='store_true', help='Populate with sample data')
        parser.add_argument('--trending', action='store_true', help='Show trending hashtags')
        parser.add_argument('--keywords', action='store_true', help='Show trending keywords')
        parser.add_argument('--sentiment', action='store_true', help='Show sentiment by hashtag')
        parser.add_argument('--stats', type=str, help='Get stats for specific hashtag')
        parser.add_argument('--all', action='store_true', help='Run all tests')
        parser.add_argument(
            '--content-type',
            type=str,
            choices=['all', 'news', 'social'],
            default='all',
            help='Which content type to test'
        )
     
    def handle(self, *args, **options):
        analyzer = get_hashtag_analyzer()
        
        if options['all']:
            self.populate_sample_data(analyzer)
            self.show_trending_hashtags(analyzer)
            self.show_trending_keywords(analyzer)
            self.show_sentiment_by_hashtag(analyzer)
            self.show_summary(analyzer)
        elif options['populate']:
            self.populate_sample_data(analyzer)
            self.show_trending_hashtags(analyzer)
        elif options['trending']:
            if not analyzer.hashtag_occurrences:
                self.stdout.write(self.style.WARNING("No data found. Populating first...\n"))
                self.populate_sample_data(analyzer)
            self.show_trending_hashtags(analyzer)
        elif options['keywords']:
            if not analyzer.keyword_occurrences:
                self.stdout.write(self.style.WARNING("No data found. Populating first...\n"))
                self.populate_sample_data(analyzer)
            self.show_trending_keywords(analyzer)
        elif options['sentiment']:
            if not analyzer.hashtag_occurrences:
                self.populate_sample_data(analyzer)
            self.show_sentiment_by_hashtag(analyzer)
        elif options['stats']:
            if not analyzer.hashtag_occurrences:
                self.populate_sample_data(analyzer)
            self.show_hashtag_stats(analyzer, options['stats'])
        else:
            # Default: populate and show all
            self.populate_sample_data(analyzer)
            self.show_trending_hashtags(analyzer)
            self.show_trending_keywords(analyzer)
            self.show_summary(analyzer)
     
    def populate_sample_data(self, analyzer, content_type='all'):
        self.stdout.write(self.style.SUCCESS("\n=== POPULATING SAMPLE DATA ===\n"))
         
        # NEWS posts (formal)
        news_posts = [
            ("#Bitcoin price analysis shows strong support levels. $BTC", 0.5, "news"),
            ("#Ethereum gas fees drop after network upgrade. #ETH", 0.6, "news"),
            ("#Crypto regulation framework under SEC review.", -0.2, "news"),
            ("Bitcoin ETF approval expected. #BTC #ETF", 0.7, "news"),
            ("#DeFi protocol completes security audit. #crypto", 0.4, "news"),
        ]
        
        # SOCIAL posts (informal)
        social_posts = [
            ("#Bitcoin breaking out! $BTC to the moon 🚀", 0.8, "twitter"),
            ("#Ethereum gas fees dropping #ETH #DeFi", 0.5, "twitter"),
            ("#Crypto crash incoming? #BTC #bearish", -0.6, "twitter"),
            ("$BTC $ETH looking bullish #cryptocurrency", 0.7, "reddit"),
            ("#Bitcoin halving soon! #HODL", 0.6, "twitter"),
            ("#NFT sales exploding #OpenSea", 0.4, "twitter"),
            ("#DeFi yield farming #crypto", 0.3, "reddit"),
            ("#Bitcoin institutional adoption #BTC", 0.7, "twitter"),
            ("#Ethereum merge successful #ETH", 0.8, "twitter"),
            ("#Bitcoin ATH incoming #BTC #moon", 0.9, "twitter"),
            ("#Altseason starting #crypto #altcoins", 0.6, "reddit"),
            ("#Bitcoin $BTC pump incoming!", 0.85, "twitter"),
            ("💎🙌 Diamond hands! #HODL #WAGMI", 0.75, "twitter"),
        ]
        
        # Select posts based on content type
        sample_posts = []
        if content_type in ['all', 'news']:
            sample_posts.extend(news_posts)
        if content_type in ['all', 'social']:
            sample_posts.extend(social_posts)
        
        news_count = len([p for p in sample_posts if p[2] == 'news'])
        social_count = len(sample_posts) - news_count
        
        for text, sentiment, source in sample_posts:
            extracted = analyzer.extract_and_record(text, sentiment=sentiment, source=source)
            hashtags = extracted['hashtags']
            cashtags = extracted['cashtags']
            keywords = extracted['keywords']
            self.stdout.write(f"  [{source}] #{', #'.join(hashtags)} | ${', $'.join(cashtags)}")
        
        self.stdout.write(self.style.SUCCESS(f"\nPopulated {len(sample_posts)} sample posts"))
        self.stdout.write(f"  News: {news_count} | Social: {social_count}")
        self.stdout.write(f"  Total hashtags tracked: {len(analyzer.hashtag_occurrences)}")
        self.stdout.write(f"  Total keywords tracked: {len(analyzer.keyword_occurrences)}")
    
    def show_trending_hashtags(self, analyzer):
        self.stdout.write(self.style.SUCCESS("\n=== TRENDING HASHTAGS ===\n"))
        
        # Use min_count=1 for testing
        trending = analyzer.get_trending_hashtags(limit=15, min_count=1)
        
        if not trending:
            self.stdout.write("No trending hashtags found.")
            self.stdout.write(f"  Hashtags in memory: {len(analyzer.hashtag_occurrences)}")
            for tag, occs in list(analyzer.hashtag_occurrences.items())[:5]:
                self.stdout.write(f"    #{tag}: {len(occs)} occurrences")
            return
        
        self.stdout.write(f"Found {len(trending)} trending hashtags:\n")
        
        for item in trending:
            sentiment_emoji = "📈" if item.sentiment > 0.2 else "📉" if item.sentiment < -0.2 else "➡️"
            self.stdout.write(
                f"  #{item.rank} {item.item} | "
                f"Count: {item.count} | "
                f"Velocity: {item.velocity:.1f}x | "
                f"{sentiment_emoji} {item.sentiment:.2f} | "
                f"Score: {item.trend_score:.1f}"
            )
    
    def show_trending_keywords(self, analyzer):
        self.stdout.write(self.style.SUCCESS("\n=== TRENDING KEYWORDS ===\n"))
        
        trending = analyzer.get_trending_keywords(limit=15, min_count=1)
        
        if not trending:
            self.stdout.write("No trending keywords found.")
            self.stdout.write(f"  Keywords in memory: {len(analyzer.keyword_occurrences)}")
            return
        
        self.stdout.write(f"Found {len(trending)} trending keywords:\n")
        
        for item in trending:
            sentiment_emoji = "📈" if item.sentiment > 0.2 else "📉" if item.sentiment < -0.2 else "➡️"
            self.stdout.write(
                f"  #{item.rank} {item.item} | "
                f"Count: {item.count} | "
                f"Velocity: {item.velocity:.1f}x | "
                f"{sentiment_emoji} {item.sentiment:.2f} | "
                f"Score: {item.trend_score:.1f}"
            )
    
    def show_sentiment_by_hashtag(self, analyzer):
        self.stdout.write(self.style.SUCCESS("\n=== SENTIMENT BY HASHTAG ===\n"))
        
        sentiment_data = analyzer.get_sentiment_by_hashtag(hours_back=24)
        
        if not sentiment_data:
            self.stdout.write("No sentiment data available.")
            return
        
        self.stdout.write(f"Sentiment data for {len(sentiment_data)} hashtags:\n")
        
        for hashtag, stats in sorted(sentiment_data.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
            bullish_pct = stats['bullish_ratio'] * 100
            bearish_pct = stats['bearish_ratio'] * 100
            sentiment_bar = "🟢" * int(bullish_pct / 20) + "🔴" * int(bearish_pct / 20)
            self.stdout.write(f"\n  {hashtag}")
            self.stdout.write(f"    Count: {stats['count']} | Avg Sentiment: {stats['avg_sentiment']:.2f}")
            self.stdout.write(f"    Bullish: {bullish_pct:.0f}% | Bearish: {bearish_pct:.0f}% {sentiment_bar}")
    
    def show_hashtag_stats(self, analyzer, hashtag):
        self.stdout.write(self.style.SUCCESS(f"\n=== STATS FOR {hashtag} ===\n"))
        
        stats = analyzer.get_hashtag_stats(hashtag)
        
        if not stats:
            self.stdout.write(f"No data found for {hashtag}")
            self.stdout.write(f"Available hashtags: {list(analyzer.hashtag_occurrences.keys())[:10]}")
            return
        
        self.stdout.write(f"  Hashtag: {stats.hashtag}")
        self.stdout.write(f"  Count (1h): {stats.count_1h}")
        self.stdout.write(f"  Count (6h): {stats.count_6h}")
        self.stdout.write(f"  Count (24h): {stats.count_24h}")
        self.stdout.write(f"  Velocity (1h): {stats.velocity_1h:.2f}x")
        self.stdout.write(f"  Velocity (6h): {stats.velocity_6h:.2f}x")
        self.stdout.write(f"  Avg Sentiment: {stats.avg_sentiment:.2f}")
        self.stdout.write(f"  Sentiment Dist: {stats.sentiment_distribution}")
        self.stdout.write(f"  Is Trending: {'🔥 Yes' if stats.is_trending else 'No'}")
        self.stdout.write(f"  Trend Score: {stats.trend_score:.1f}")
    
    def show_summary(self, analyzer):
        self.stdout.write(self.style.SUCCESS("\n=== SUMMARY ===\n"))
        
        summary = analyzer.get_summary()
        
        for key, value in summary.items():
            self.stdout.write(f"  {key}: {value}")