from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from myapp.models import Trendinghashtag, Trendingkeyword, Trendingtopic
from myapp.services.content.hashtag_analyzer import get_hashtag_analyzer
from myapp.services.content.topic_modeler import get_topic_modeler

 
class Command(BaseCommand):
    help = 'Test persisting trending data to PostgreSQL'
     
    def add_arguments(self, parser):
        parser.add_argument('--persist', action='store_true', help='Persist current trending data')
        parser.add_argument('--query', action='store_true', help='Query persisted data')
        parser.add_argument('--cleanup', action='store_true', help='Cleanup old data')
        parser.add_argument('--populate', action='store_true', help='Populate sample data first')
        parser.add_argument('--all', action='store_true', help='Populate, persist, and query')
     
    def handle(self, *args, **options):
        if options['all']:
            self.populate_sample_data()
            self.persist_trending_data()
            self.query_persisted_data()
        elif options['populate']:
            self.populate_sample_data()
        elif options['persist']:
            self.persist_trending_data()
        elif options['query']:
            self.query_persisted_data()
        elif options['cleanup']:
            self.cleanup_old_data()
        else:
            # Default: populate, persist, and query
            self.populate_sample_data()
            self.persist_trending_data()
            self.query_persisted_data()
    
    def populate_sample_data(self):
        """Populate sample data for hashtags and topics"""
        self.stdout.write(self.style.SUCCESS("\n=== POPULATING SAMPLE DATA ===\n"))
        
        hashtag_analyzer = get_hashtag_analyzer()
        topic_modeler = get_topic_modeler()
        
        # Sample social posts for hashtag analyzer
        sample_posts = [
            ("#Bitcoin breaking out! $BTC to the moon 🚀", 0.8, "twitter"),
            ("#Ethereum gas fees dropping #ETH #DeFi", 0.5, "twitter"),
            ("#Crypto crash incoming? #BTC #bearish", -0.6, "twitter"),
            ("$BTC $ETH looking bullish #cryptocurrency", 0.7, "reddit"),
            ("#Bitcoin halving soon! #HODL", 0.6, "twitter"),
            ("#NFT sales exploding #OpenSea", 0.4, "twitter"),
            ("#DeFi yield farming #crypto", 0.3, "reddit"),
            ("#Bitcoin institutional adoption #BTC", 0.7, "twitter"),
            ("#Ethereum merge successful #ETH", 0.8, "twitter"),
            ("#Crypto regulation news #SEC", -0.3, "twitter"),
            ("#Bitcoin ATH incoming #BTC #moon", 0.9, "twitter"),
            ("#Altseason starting #crypto #altcoins", 0.6, "reddit"),
            ("#Bitcoin $BTC pump incoming!", 0.85, "twitter"),
            ("#Bitcoin dominance rising #BTC", 0.7, "twitter"),
            ("#Crypto market pumping hard!", 0.75, "telegram"),
            ("#DeFi summer is back #yield", 0.6, "twitter"),
            ("#ETH breaking resistance #Ethereum", 0.65, "twitter"),
            ("#Bitcoin #BTC new ATH soon?", 0.8, "reddit"),
        ]
        
        for text, sentiment, source in sample_posts:
            hashtag_analyzer.extract_and_record(text, sentiment=sentiment, source=source)
        
        self.stdout.write(f"  Populated {len(sample_posts)} social posts")
        self.stdout.write(f"  Hashtags tracked: {len(hashtag_analyzer.hashtag_occurrences)}")
        self.stdout.write(f"  Keywords tracked: {len(hashtag_analyzer.keyword_occurrences)}")
        
        # Sample documents for topic modeler
        documents = [
            "Bitcoin price surges past $50,000 amid institutional buying",
            "Ethereum gas fees drop following network upgrade",
            "SEC delays Bitcoin ETF decision again",
            "DeFi protocol launches new yield farming mechanism",
            "NFT marketplace sees record trading volume",
            "Binance faces regulatory scrutiny in multiple countries",
            "Bitcoin halving expected to impact mining profitability",
            "Ethereum staking rewards attract institutional investors",
            "Crypto market cap reaches new all-time high",
            "Regulatory clarity needed for crypto adoption",
            "Bitcoin ETF approval could bring billions in inflows",
            "DeFi lending protocols see surge in deposits",
            "NFT artists earning millions from digital art",
            "Exchange outages during market volatility",
            "Layer 2 solutions gaining traction for scaling",
            "Stablecoin regulations under discussion",
            "Bitcoin mining difficulty reaches record high",
            "Ethereum merge successfully completed",
            "Crypto winter fears as prices decline",
            "Institutional adoption accelerates despite volatility",
        ]
        
        doc_ids = [f"doc_{i}" for i in range(len(documents))]
        result = topic_modeler.fit(documents, doc_ids)
        
        if result['status'] == 'success':
            self.stdout.write(f"  Topic model fitted: {result['num_topics']} topics discovered")
            
            # Record occurrences for trending
            import random
            for assignment in result.get('topic_assignments', []):
                sentiment = random.uniform(-0.3, 0.8)
                topic_modeler.record_topic_occurrence(
                    assignment['topic_id'],
                    sentiment=sentiment
                )
        else:
            self.stdout.write(self.style.WARNING(f"  ️ Topic modeling: {result.get('message', 'failed')}"))
    
    def persist_trending_data(self):
        self.stdout.write(self.style.SUCCESS("\n=== PERSISTING TRENDING DATA ===\n"))
        
        hashtag_analyzer = get_hashtag_analyzer()
        topic_modeler = get_topic_modeler()
        
        # Check if we have data
        if not hashtag_analyzer.hashtag_occurrences:
            self.stdout.write(self.style.WARNING("  No hashtag data in memory. Run with --populate or --all first."))
            return
        
        # Persist hashtags
        trending_hashtags = hashtag_analyzer.get_trending_hashtags(limit=20, min_count=1)
        hashtag_count = 0
        
        self.stdout.write(f"  Found {len(trending_hashtags)} trending hashtags to persist")
        
        for item in trending_hashtags:
            hashtag_name = item.item.lstrip('#')
            stats = hashtag_analyzer.get_hashtag_stats(hashtag_name)
            if stats:
                Trendinghashtag.objects.create(
                    hashtag=item.item,
                    count_1h=stats.count_1h,
                    count_6h=stats.count_6h,
                    count_24h=stats.count_24h,
                    velocity=stats.velocity_1h,
                    avg_sentiment=stats.avg_sentiment,
                    trend_score=item.trend_score
                )
                hashtag_count += 1
        
        self.stdout.write(f"  Persisted {hashtag_count} trending hashtags")
        
        # Persist keywords
        trending_keywords = hashtag_analyzer.get_trending_keywords(limit=20, min_count=1)
        keyword_count = 0
        
        self.stdout.write(f"  Found {len(trending_keywords)} trending keywords to persist")
        
        for item in trending_keywords:
            stats = hashtag_analyzer.get_keyword_stats(item.item)
            if stats:
                Trendingkeyword.objects.create(
                    keyword=item.item,
                    count_1h=stats.count_1h,
                    count_6h=stats.count_6h,
                    count_24h=stats.count_24h,
                    velocity=item.velocity,
                    avg_sentiment=stats.avg_sentiment,
                    sources=list(stats.sources) if stats.sources else []
                )
                keyword_count += 1
        
        self.stdout.write(f"  Persisted {keyword_count} trending keywords")
        
        # Persist topics
        trending_topics = topic_modeler.get_trending_topics(hours_back=24)
        spikes = topic_modeler.detect_topic_spikes()
        spike_ids = {s['topic_id'] for s in spikes}
        topic_count = 0
        
        self.stdout.write(f"  Found {len(trending_topics)} trending topics to persist")
        
        for topic in trending_topics[:10]:
            Trendingtopic.objects.create(
                topic_id=topic.topic_id,
                topic_name=topic.name,
                keywords=topic.keywords if topic.keywords else [],
                document_count=topic.document_count,
                velocity=topic.velocity,
                avg_sentiment=topic.avg_sentiment,
                is_spike=topic.topic_id in spike_ids
            )
            topic_count += 1
        
        self.stdout.write(f"  Persisted {topic_count} trending topics")
        self.stdout.write(self.style.SUCCESS(f"\nTotal: {hashtag_count + keyword_count + topic_count} records saved to PostgreSQL"))
    
    def query_persisted_data(self):
        self.stdout.write(self.style.SUCCESS("\n=== QUERYING PERSISTED DATA ===\n"))
        
        # Query hashtags
        total_hashtags = Trendinghashtag.objects.count()
        hashtags = Trendinghashtag.objects.order_by('-trend_score')[:10]
        self.stdout.write(f"Top Hashtags ({total_hashtags} total in PostgreSQL):")
        
        if hashtags:
            for h in hashtags:
                sentiment_emoji = "📈" if h.avg_sentiment > 0.2 else "📉" if h.avg_sentiment < -0.2 else "➡️"
                self.stdout.write(f"  {h.hashtag} | Score: {h.trend_score:.1f} | {sentiment_emoji} {h.avg_sentiment:.2f} | Count: {h.count_24h}")
        else:
            self.stdout.write("  No hashtags found")
        
        # Query keywords
        total_keywords = Trendingkeyword.objects.count()
        keywords = Trendingkeyword.objects.order_by('-velocity')[:10]
        self.stdout.write(f"\nTop Keywords ({total_keywords} total in PostgreSQL):")
        
        if keywords:
            for k in keywords:
                sentiment_emoji = "📈" if k.avg_sentiment > 0.2 else "📉" if k.avg_sentiment < -0.2 else "➡️"
                self.stdout.write(f"  {k.keyword} | Velocity: {k.velocity:.1f}x | {sentiment_emoji} {k.avg_sentiment:.2f} | Count: {k.count_24h}")
        else:
            self.stdout.write("  No keywords found")
        
        # Query topics
        total_topics = Trendingtopic.objects.count()
        topics = Trendingtopic.objects.order_by('-velocity')[:10]
        self.stdout.write(f"\nTop Topics ({total_topics} total in PostgreSQL):")
        
        if topics:
            for t in topics:
                spike = "🔥" if t.is_spike else "  "
                self.stdout.write(f"  {spike} {t.topic_name} | Velocity: {t.velocity:.1f}x | Docs: {t.document_count} | Keywords: {t.keywords[:3] if t.keywords else []}")
        else:
            self.stdout.write("  No topics found")
        
        self.stdout.write(self.style.SUCCESS(f"\nTotal records in PostgreSQL: {total_hashtags + total_keywords + total_topics}"))
    
    def cleanup_old_data(self):
        self.stdout.write(self.style.SUCCESS("\n=== CLEANING UP OLD DATA ===\n"))
        
        cutoff = timezone.now() - timedelta(days=30)
        
        h_deleted, _ = Trendinghashtag.objects.filter(timestamp__lt=cutoff).delete()
        k_deleted, _ = Trendingkeyword.objects.filter(timestamp__lt=cutoff).delete()
        t_deleted, _ = Trendingtopic.objects.filter(timestamp__lt=cutoff).delete()
        
        self.stdout.write(f"  Deleted {h_deleted} old hashtag records")
        self.stdout.write(f"  Deleted {k_deleted} old keyword records")
        self.stdout.write(f"  Deleted {t_deleted} old topic records")
        self.stdout.write(self.style.SUCCESS(f"\nCleanup complete"))