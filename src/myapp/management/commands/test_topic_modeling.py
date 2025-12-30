from django.core.management.base import BaseCommand
from myapp.services.content.topic_modeler import get_topic_modeler


class Command(BaseCommand):
    help = 'Test topic modeling functionality'
    
    def add_arguments(self, parser):
        parser.add_argument('--fit', action='store_true', help='Fit model on sample data')
        parser.add_argument('--trending', action='store_true', help='Show trending topics')
        parser.add_argument('--spikes', action='store_true', help='Detect topic spikes')
        parser.add_argument('--summary', action='store_true', help='Show topic summary')
        parser.add_argument('--all', action='store_true', help='Run all tests')
        parser.add_argument(
            '--content-type',
            type=str,
            choices=['all', 'news', 'social'],
            default='all',
            help='Which content type to test'
        )
    
    def handle(self, *args, **options):
        topic_modeler = get_topic_modeler()
        content_type = options.get('content_type', 'all')
         
        if options['all']:
            self.test_fit(topic_modeler, content_type)
            self.show_trending(topic_modeler)
            self.detect_spikes(topic_modeler)
            self.show_summary(topic_modeler)
        elif options['fit']:
            self.test_fit(topic_modeler, content_type)
            self.show_trending(topic_modeler)
        elif options['trending']:
            if not topic_modeler.topic_history:
                self.stdout.write(self.style.WARNING("No topic data found. Running fit first...\n"))
                self.test_fit(topic_modeler, content_type)
            self.show_trending(topic_modeler)
        elif options['spikes']:
            if not topic_modeler.topic_history:
                self.stdout.write(self.style.WARNING("No topic data found. Running fit first...\n"))
                self.test_fit(topic_modeler, content_type)
            self.detect_spikes(topic_modeler)
        elif options['summary']:
            self.show_summary(topic_modeler)
        else:
            self.test_fit(topic_modeler, content_type)
            self.show_trending(topic_modeler)
            self.show_summary(topic_modeler)
    
    def test_fit(self, topic_modeler, content_type='all'):
        self.stdout.write(self.style.SUCCESS("\n=== TOPIC MODELING TEST ===\n"))
        
        # NEWS documents (formal)
        news_documents = [
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
        ]
        
        # SOCIAL documents (informal/crypto slang)
        social_documents = [
            "🚀🚀🚀 BTC TO THE MOON! HODL! 💎🙌",
            "Just bought the dip! $ETH looking bullish af",
            "Who else is stacking sats? This is the way! #Bitcoin",
            "NFA but this altcoin gonna pump hard 📈",
            "Bear market? More like buying opportunity! #crypto",
            "Wen lambo? Asking for a friend 😂 #HODL",
            "Diamond hands only! Paper hands stay away 💎",
            "This dip is nothing, we've seen worse. Stay strong!",
            "Market looking oversold, time to accumulate",
            "WAGMI! Keep holding anon 🦍",
        ]
        
        # Select documents based on content type
        documents = []
        doc_types = []
        
        if content_type in ['all', 'news']:
            documents.extend(news_documents)
            doc_types.extend(['news'] * len(news_documents))
        
        if content_type in ['all', 'social']:
            documents.extend(social_documents)
            doc_types.extend(['social'] * len(social_documents))
        
        doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.stdout.write(f"Fitting model on {len(documents)} documents...")
        self.stdout.write(f"  News: {doc_types.count('news')} | Social: {doc_types.count('social')}")
        
        result = topic_modeler.fit(documents, doc_ids)
        
        if result['status'] == 'success':
            self.stdout.write(self.style.SUCCESS(f"Model fitted successfully!"))
            self.stdout.write(f"  Mode: {result.get('mode', 'unknown')}")
            self.stdout.write(f"  Topics discovered: {result['num_topics']}")
            self.stdout.write(f"\nTopics:")
            for topic in result['topics'][:10]:
                self.stdout.write(f"  Topic {topic['topic_id']}: {topic['keywords'][:5]}")
            
            # Record some sentiment data for trending
            import random
            from datetime import datetime
            from django.utils import timezone
            now = timezone.now()
            
            for assignment in result.get('topic_assignments', []):
                sentiment = random.uniform(-0.5, 0.8)
                topic_modeler.record_topic_occurrence(
                    assignment['topic_id'], 
                    sentiment=sentiment,
                    timestamp=now
                )
            
            self.stdout.write(f"\n  Recorded {len(result.get('topic_assignments', []))} topic occurrences")
        else:
            self.stdout.write(self.style.WARNING(f"️ {result.get('message', result['status'])}"))
    
    def show_trending(self, topic_modeler):
        self.stdout.write(self.style.SUCCESS("\n=== TRENDING TOPICS ===\n"))
        
        trending = topic_modeler.get_trending_topics(hours_back=24)
         
        if not trending:
            self.stdout.write("No trending topics found.")
            self.stdout.write(f"  Topic history entries: {len(topic_modeler.topic_history)}")
            return
         
        self.stdout.write(f"Found {len(trending)} topics:\n")
        
        for topic in trending[:10]:
            status = "🔥 TRENDING" if topic.is_trending else "📊"
            self.stdout.write(f"{status} Topic {topic.topic_id}: {topic.name}")
            if topic.keywords:
                self.stdout.write(f"   Keywords: {topic.keywords[:5]}")
            self.stdout.write(f"   Velocity: {topic.velocity:.2f}x")
            self.stdout.write(f"   Documents: {topic.document_count}")
            self.stdout.write(f"   Sentiment: {topic.avg_sentiment:.2f}")
            self.stdout.write("")
    
    def detect_spikes(self, topic_modeler):
        self.stdout.write(self.style.SUCCESS("\n=== TOPIC SPIKES ===\n"))
        
        spikes = topic_modeler.detect_topic_spikes(threshold_multiplier=1.5)
        
        if not spikes:
            self.stdout.write("No topic spikes detected.")
            self.stdout.write("  (Spikes are detected when current hour count exceeds average by threshold)")
            return
        
        for spike in spikes:
            self.stdout.write(f"🚨 SPIKE: Topic {spike['topic_id']}")
            self.stdout.write(f"   Keywords: {spike['keywords']}")
            self.stdout.write(f"   Current: {spike['current_count']} (avg: {spike['average_count']:.1f})")
            self.stdout.write(f"   Multiplier: {spike['spike_multiplier']:.1f}x")
            self.stdout.write("")
    
    def show_summary(self, topic_modeler):
        self.stdout.write(self.style.SUCCESS("\n=== TOPIC MODEL SUMMARY ===\n"))
        
        summary = topic_modeler.get_topic_summary()
        
        for key, value in summary.items():
            self.stdout.write(f"  {key}: {value}")
        
        self.stdout.write(f"\n  Topic history entries: {len(topic_modeler.topic_history)}")
        for topic_id, history in list(topic_modeler.topic_history.items())[:5]:
            self.stdout.write(f"    Topic {topic_id}: {len(history)} occurrences")