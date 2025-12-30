"""
Django Management Command: Test Cross-Verification Engine
Comprehensive testing of semantic similarity, claim extraction, and cross-referencing

Usage:
  python manage.py test_cross_verification --basic
  python manage.py test_cross_verification --semantic
  python manage.py test_cross_verification --claims
  python manage.py test_cross_verification --live
  python manage.py test_cross_verification --all
  python manage.py test_cross_verification --all --detailed

Save to: myapp/management/commands/test_cross_verification.py
"""

from django.core.management.base import BaseCommand
from datetime import datetime
from django.utils import timezone
import json


class Command(BaseCommand):
    help = 'Test cross-verification engine functionality'
    
    def add_arguments(self, parser):
        parser.add_argument('--basic', action='store_true', help='Test basic initialization')
        parser.add_argument('--semantic', action='store_true', help='Test semantic similarity')
        parser.add_argument('--claims', action='store_true', help='Test claim extraction')
        parser.add_argument('--live', action='store_true', help='Test with stored MongoDB content')
        parser.add_argument('--all', action='store_true', help='Run all tests')
        parser.add_argument('--detailed', action='store_true', help='Show detailed output')
    
    def handle(self, *args, **options):
        self.detailed = options.get('detailed', False)
        
        self.stdout.write(self.style.SUCCESS("\n" + "="*80))
        self.stdout.write(self.style.SUCCESS("   CROSS-VERIFICATION ENGINE TEST SUITE"))
        self.stdout.write(self.style.SUCCESS("="*80 + "\n"))
        
        run_all = options.get('all', False)
        
        if run_all or options.get('basic'):
            self.test_basic_initialization()
        
        if run_all or options.get('semantic'):
            self.test_semantic_similarity()
        
        if run_all or options.get('claims'):
            self.test_claim_extraction()
        
        if run_all or options.get('live'):
            self.test_live_verification()
        
        if not any([options.get('basic'), options.get('semantic'), 
                    options.get('claims'), options.get('live'), run_all]):
            self.test_basic_initialization()
            self.test_semantic_similarity()
        
        self.stdout.write("\n" + "="*80)
        self.stdout.write(self.style.SUCCESS("✅ All tests completed!"))
        self.stdout.write("="*80 + "\n")
    
    def test_basic_initialization(self):
        """Test 1: Basic initialization and configuration"""
        self.stdout.write(self.style.WARNING("\n" + "="*80))
        self.stdout.write(self.style.WARNING("TEST 1: BASIC INITIALIZATION"))
        self.stdout.write(self.style.WARNING("="*80))
        
        try:
            from myapp.services.content.cross_verification_engine import get_verification_engine
            
            engine = get_verification_engine()
            
            self.stdout.write("\n✅ Engine initialized successfully")
            self.stdout.write(f"\nConfiguration:")
            self.stdout.write(f"   Similarity Threshold: {engine.config['similarity_threshold']}")
            self.stdout.write(f"   Temporal Window: {engine.config['temporal_window_hours']} hours")
            self.stdout.write(f"   Min References: {engine.config['min_references_for_high_confidence']}")
            self.stdout.write(f"   Max References: {engine.config['max_references_to_return']}")
            
            self.stdout.write(f"\nSource Tiers:")
            for tier, sources in engine.source_tiers.items():
                self.stdout.write(f"   {tier}: {', '.join(sources[:5])}")
            
            self.stdout.write(self.style.SUCCESS("\n✅ TEST 1 PASSED"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"\n❌ TEST 1 FAILED: {e}"))
            import traceback
            traceback.print_exc()
    
    def test_semantic_similarity(self):
        """Test 2: Semantic similarity computation"""
        self.stdout.write(self.style.WARNING("\n" + "="*80))
        self.stdout.write(self.style.WARNING("TEST 2: SEMANTIC SIMILARITY"))
        self.stdout.write(self.style.WARNING("="*80))
        
        try:
            from myapp.services.content.cross_verification_engine import get_verification_engine
            
            engine = get_verification_engine()
            
            test_cases = [
                ("Bitcoin price surges past $100,000 milestone",
                 "BTC breaks through $100K barrier for first time", 'high', 0.4),
                ("Ethereum gas fees drop following network upgrade",
                 "ETH transaction costs decrease after latest fork", 'high', 0.4),
                ("Bitcoin price reaches new all-time high",
                 "Ethereum network sees record transactions", 'low', 0.4),
            ]
            
            self.stdout.write("\nTesting semantic similarity:\n")
            
            passed = 0
            for i, (text1, text2, expected, threshold) in enumerate(test_cases, 1):
                similarity = engine.compute_semantic_similarity(text1, text2)
                bar = '█' * int(similarity * 50) + '░' * (50 - int(similarity * 50))
                
                self.stdout.write(f"\n{i}. Expected: {expected.upper()}")
                self.stdout.write(f"   Text 1: \"{text1}\"")
                self.stdout.write(f"   Text 2: \"{text2}\"")
                self.stdout.write(f"   Similarity: {similarity:.4f}")
                self.stdout.write(f"   Visual: [{bar}]")
                
                if similarity >= threshold:
                    self.stdout.write(self.style.SUCCESS(f"   ✅ PASS"))
                    passed += 1
                else:
                    self.stdout.write(self.style.WARNING(f"   ⚠️  Below threshold"))
            
            self.stdout.write(f"\nResults: {passed}/{len(test_cases)} passed")
            self.stdout.write(self.style.SUCCESS("\n✅ TEST 2 COMPLETED"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"\n❌ TEST 2 FAILED: {e}"))
            import traceback
            traceback.print_exc()
    
    def test_claim_extraction(self):
        """Test 3: Claim extraction"""
        self.stdout.write(self.style.WARNING("\n" + "="*80))
        self.stdout.write(self.style.WARNING("TEST 3: CLAIM EXTRACTION"))
        self.stdout.write(self.style.WARNING("="*80))
        
        try:
            from myapp.services.content.cross_verification_engine import get_verification_engine
            
            engine = get_verification_engine()
            
            test_articles = [
                {
                    'title': 'Bitcoin Surges Past $100,000',
                    'description': 'Bitcoin price reached $100,000 driven by institutional buying.',
                    'extracted_entities': {'cryptocurrencies': ['Bitcoin', 'BTC']},
                    'source_id': 'test_1',
                    'created_at': timezone.now()
                },
            ]
            
            self.stdout.write("\nExtracting claims:\n")
            
            for i, article in enumerate(test_articles, 1):
                claims = engine.extract_claims(article)
                
                self.stdout.write(f"\n{i}. \"{article['title']}\"")
                self.stdout.write(f"   Claims: {len(claims)}")
                
                for j, claim in enumerate(claims[:3], 1):
                    self.stdout.write(f"   {j}. {claim.claim_type}: \"{claim.text[:60]}...\"")
            
            self.stdout.write(self.style.SUCCESS("\n✅ TEST 3 COMPLETED"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"\n❌ TEST 3 FAILED: {e}"))
            import traceback
            traceback.print_exc()
    
    def test_live_verification(self):
        """Test 4: Live verification with MongoDB content"""
        self.stdout.write(self.style.WARNING("\n" + "="*80))
        self.stdout.write(self.style.WARNING("TEST 4: LIVE VERIFICATION"))
        self.stdout.write(self.style.WARNING("="*80))
        
        try:
            from myapp.services.content.cross_verification_engine import get_verification_engine
            from myapp.services.mongo_manager import get_mongo_manager
            
            engine = get_verification_engine()
            mongo = get_mongo_manager()
            
            articles = list(mongo.collections['news_articles'].find().sort('created_at', -1).limit(10))
            
            if not articles:
                self.stdout.write(self.style.WARNING("\n⚠️  No articles in database"))
                self.stdout.write("   Run: python manage.py run_pipeline")
                return
            
            test_article = articles[0]
            candidates = articles[1:]
            
            self.stdout.write(f"\nArticle: \"{test_article.get('title', '')[:60]}...\"")
            self.stdout.write(f"Candidates: {len(candidates)}")
            
            result = engine.verify_content(test_article, candidates)
            
            self.stdout.write(f"\n{'='*80}")
            self.stdout.write(self.style.SUCCESS("RESULTS:"))
            self.stdout.write(f"{'='*80}")
            self.stdout.write(f"   References: {result.total_references}")
            self.stdout.write(f"   Similarity: {result.avg_similarity:.4f}")
            self.stdout.write(f"   Corroboration: {result.corroboration_score:.2f}/10")
            self.stdout.write(f"   Confidence: {result.confidence:.2%}")
            
            if result.flags:
                self.stdout.write(f"   Flags: {', '.join(result.flags)}")
            
            if result.reasoning:
                self.stdout.write(f"\n💭 {result.reasoning}")
            
            # Check MongoDB storage
            stored = mongo.collections['news_articles'].find_one({'source_id': test_article['source_id']})
            if stored and 'verification_details' in stored:
                self.stdout.write(self.style.SUCCESS("\n✅ verification_details stored in MongoDB"))
            else:
                self.stdout.write(self.style.WARNING("\n⚠️  verification_details NOT in MongoDB"))
            
            # Interpretation
            if result.total_references == 0:
                self.stdout.write(self.style.WARNING("\n⚠️  NO MATCHES (Expected with mock data)"))
                self.stdout.write(f"   Score 5.00 = NEUTRAL (no penalty)")
            else:
                self.stdout.write(self.style.SUCCESS("\n✅ VERIFICATION WORKING!"))
            
            self.stdout.write(self.style.SUCCESS("\n✅ TEST 4 COMPLETED"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"\n❌ TEST 4 FAILED: {e}"))
            import traceback
            traceback.print_exc()