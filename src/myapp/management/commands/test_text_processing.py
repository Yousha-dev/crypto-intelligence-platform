from django.core.management.base import BaseCommand
from myapp.services.content.text_processor import get_text_processor
from myapp.services.content.hashtag_analyzer import get_hashtag_analyzer


class Command(BaseCommand):
    help = 'Test text processing and NER functionality'
     
    def add_arguments(self, parser):
        parser.add_argument('--text', type=str, help='Text to process')
        parser.add_argument('--ner', action='store_true', help='Test NER extraction')
        parser.add_argument('--hashtags', action='store_true', help='Test hashtag extraction')
        parser.add_argument('--full', action='store_true', help='Run all tests')
        parser.add_argument(
            '--content-type', 
            type=str,
            choices=['all', 'news', 'social'],
            default='all',
            help='Which content type to test'
        )
    
    def handle(self, *args, **options):
        text_processor = get_text_processor()
        hashtag_analyzer = get_hashtag_analyzer()
        content_type = options.get('content_type', 'all')
         
        if options['text']:
            self.test_single_text(options['text'], text_processor, hashtag_analyzer)
        elif options['full']:
            self.run_full_test(text_processor, hashtag_analyzer, content_type)
        else:
            self.run_full_test(text_processor, hashtag_analyzer, content_type)
    
    def test_single_text(self, text, text_processor, hashtag_analyzer):
        self.stdout.write(self.style.SUCCESS(f"\n=== PROCESSING TEXT ===\n"))
        self.stdout.write(f"Input: {text[:100]}...")
        
        # Preprocess
        processed = text_processor.preprocess(text)
        self.stdout.write(f"\n📝 PREPROCESSING:")
        self.stdout.write(f"  Language: {processed.language} ({processed.language_confidence:.0%})")
        self.stdout.write(f"  Is English: {processed.is_english}")
        self.stdout.write(f"  Word Count: {processed.word_count}")
        self.stdout.write(f"  Hashtags: {processed.hashtags}")
        self.stdout.write(f"  Mentions: {processed.mentions}")
        self.stdout.write(f"  URLs: {len(processed.urls)} found")
        
        # NER
        entities = text_processor.extract_entities(text)
        self.stdout.write(f"\n🏷️ NAMED ENTITIES:")
        self.stdout.write(f"  Cryptocurrencies: {entities.cryptocurrencies}")
        self.stdout.write(f"  Exchanges: {entities.exchanges}")
        self.stdout.write(f"  Persons: {entities.persons}")
        self.stdout.write(f"  Organizations: {entities.organizations}")
        self.stdout.write(f"  Money Amounts: {entities.money_amounts}")
        self.stdout.write(f"  Dates: {entities.dates}")
        self.stdout.write(f"  Locations: {entities.locations}")
        
        # Hashtag analysis
        extracted = hashtag_analyzer.extract_and_record(text, sentiment=0.5, source='test')
        self.stdout.write(f"\nHASHTAG ANALYSIS:")
        self.stdout.write(f"  Hashtags: {extracted['hashtags']}")
        self.stdout.write(f"  Cashtags: {extracted['cashtags']}")
        self.stdout.write(f"  Keywords: {extracted['keywords']}")
    
    def run_full_test(self, text_processor, hashtag_analyzer, content_type='all'):
        # NEWS texts (formal)
        news_texts = [
            "Bitcoin surges to $50,000 as institutional investors increase exposure. #Bitcoin #Crypto",
            "SEC announces investigation into Binance. Gary Gensler warns investors.",
            "Ethereum's Vitalik Buterin discusses the merge upgrade at ETH Denver.",
            "Breaking: FTX files for bankruptcy. Sam Bankman-Fried resigns as CEO.",
            "DeFi protocol hacked for $100 million. Users urged to revoke approvals.",
        ]
        
        # SOCIAL texts (informal/crypto slang)
        social_texts = [
            "🚀🚀🚀 BTC TO THE MOON! HODL! 💎🙌 #Bitcoin @elonmusk",
            "Just bought the dip on $ETH! Wen lambo? 😂 #Ethereum #DeFi",
            "Diamond hands only! Paper hands stay away 💎 #crypto #WAGMI",
            "NFA but this altcoin gonna pump hard 📈 $SOL $AVAX",
            "Bearish on $DOGE, bullish on $BTC. This is the way! @CryptoAnalyst",
        ]
        
        # Select texts based on content type
        test_texts = []
        text_types = []
        
        if content_type in ['all', 'news']:
            test_texts.extend(news_texts)
            text_types.extend(['news'] * len(news_texts))
        
        if content_type in ['all', 'social']:
            test_texts.extend(social_texts)
            text_types.extend(['social'] * len(social_texts))
        
        self.stdout.write(self.style.SUCCESS("\n=== FULL TEXT PROCESSING TEST ===\n"))
        self.stdout.write(f"Testing {len(test_texts)} texts (News: {text_types.count('news')}, Social: {text_types.count('social')})")
        
        results = {'news': [], 'social': []}
        
        for i, (text, text_type) in enumerate(zip(test_texts, text_types), 1):
            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(f"Test {i} [{text_type.upper()}]: {text[:50]}...")
            self.test_single_text(text, text_processor, hashtag_analyzer)
            
            # Collect results
            processed = text_processor.preprocess(text)
            entities = text_processor.extract_entities(text)
            results[text_type].append({
                'text': text[:30],
                'hashtags': processed.hashtags,
                'mentions': processed.mentions,
                'cryptos': entities.cryptocurrencies
            })
        
        # Summary
        self.stdout.write(self.style.SUCCESS(f"\n{'='*60}"))
        self.stdout.write(self.style.SUCCESS("=== TEST SUMMARY ==="))
        
        for text_type in ['news', 'social']:
            if results[text_type]:
                self.stdout.write(f"\n{text_type.upper()} Content ({len(results[text_type])} texts):")
                total_hashtags = sum(len(r['hashtags']) for r in results[text_type])
                total_mentions = sum(len(r['mentions']) for r in results[text_type])
                total_cryptos = sum(len(r['cryptos']) for r in results[text_type])
                self.stdout.write(f"  Total Hashtags Found: {total_hashtags}")
                self.stdout.write(f"  Total Mentions Found: {total_mentions}")
                self.stdout.write(f"  Total Cryptos Found: {total_cryptos}")
        
        self.stdout.write(self.style.SUCCESS("\nText processing test completed!"))
        
        summary = hashtag_analyzer.get_summary()
        self.stdout.write(f"\nHashtag Analyzer Summary:")
        self.stdout.write(f"  Total Hashtags Tracked: {summary['total_hashtags_tracked']}")
        self.stdout.write(f"  Total Keywords Tracked: {summary['total_keywords_tracked']}")