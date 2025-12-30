# management/commands/test_credibility_system.py
"""
Test the credibility scoring system with RAW data format (matching fetcher output)
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
import json
import logging
import random
from typing import Dict, List
from collections import Counter
from datetime import timedelta

from myapp.services.mongo_manager import get_mongo_manager
from myapp.services.content.credibility_engine import get_credibility_engine, get_threshold_manager

logger = logging.getLogger(__name__)
  
class Command(BaseCommand):
    help = 'Test the credibility scoring system with RAW data (matching fetcher output)'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--action',
            type=str,
            choices=[
                'test_scoring', 
                'test_storage', 
                'test_full_pipeline', 
                'stats', 
                'test_formula',
                'test_source_history',
                'test_cross_reference',
                'test_extremity',
                'test_social_scoring',
                'test_social_pipeline',
                'test_all'
            ],
            default='test_full_pipeline',
            help='Which test to run'
        )
        
        parser.add_argument(
            '--sample-size',
            type=int,
            default=10,
            help='Number of sample articles to test with'
        )
        
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed score breakdown'
        )
    
    def handle(self, *args, **options):
        action = options['action']
        sample_size = options['sample_size']
        verbose = options['verbose']
        
        self.stdout.write(
            self.style.SUCCESS(f'Running credibility system test: {action}')
        )
        self.stdout.write(
            self.style.WARNING('Using RAW data format (matching fetcher output)\n')
        )
        
        if action == 'test_scoring':
            self.test_credibility_scoring(sample_size, verbose)
        elif action == 'test_storage':
            self.test_mongodb_storage(sample_size)
        elif action == 'test_full_pipeline':
            self.test_full_pipeline(sample_size, verbose)
        elif action == 'stats':
            self.show_statistics()
        elif action == 'test_formula':
            self.test_formula_weights()
        elif action == 'test_source_history':
            self.test_source_history()
        elif action == 'test_cross_reference':
            self.test_cross_reference()
        elif action == 'test_extremity':
            self.test_sentiment_extremity()
        elif action == 'test_social_scoring':
            self.test_social_scoring(verbose)
        elif action == 'test_social_pipeline':
            self.test_social_pipeline(sample_size, verbose)
        elif action == 'test_all':
            self.test_all_features(sample_size, verbose)

    def test_all_features(self, sample_size: int, verbose: bool):
        """Run all feature tests including social posts"""
        self.stdout.write(self.style.SUCCESS("\n" + "="*70))
        self.stdout.write(self.style.SUCCESS("   COMPREHENSIVE CREDIBILITY SYSTEM TEST (RAW DATA)"))
        self.stdout.write(self.style.SUCCESS("="*70 + "\n"))
        
        self.test_formula_weights()
        self.test_source_history()
        self.test_cross_reference()
        self.test_sentiment_extremity()
        self.test_credibility_scoring(sample_size, verbose)
        self.test_social_scoring(verbose)
        self.test_social_pipeline(sample_size, verbose)
        
        self.stdout.write(self.style.SUCCESS("\n" + "="*70))
        self.stdout.write(self.style.SUCCESS("   ALL TESTS COMPLETED"))
        self.stdout.write(self.style.SUCCESS("="*70 + "\n"))
    
    def test_source_history(self):
        """Test source historical accuracy tracking feature"""
        self.stdout.write(self.style.SUCCESS("\n" + "="*60))
        self.stdout.write(self.style.SUCCESS("=== TESTING SOURCE HISTORY TRACKING ==="))
        self.stdout.write(self.style.SUCCESS("="*60 + "\n"))
        
        engine = get_credibility_engine()
        
        # Show current source history state
        self.stdout.write("Current Source History State:")
        if engine.source_history:
            for source, record in engine.source_history.items():
                self.stdout.write(f"   {source}: {record.total_articles} articles, "
                                f"{record.accuracy_rate:.1%} accuracy, "
                                f"reliability: {record.reliability_score:.2f}")
        else:
            self.stdout.write("   No source history recorded yet")
        
        # Simulate source history updates
        self.stdout.write("\n📝 Simulating source history updates...")
        
        test_sources = [
            ('coindesk', True, False, False, 8.5),
            ('coindesk', True, False, False, 9.0),
            ('coindesk', True, False, False, 8.8),
            ('coindesk', True, False, False, 8.7),
            ('coindesk', False, True, False, 5.0),
            ('coindesk', True, False, False, 8.9),
            ('unknown_blog', True, False, False, 4.0),
            ('unknown_blog', False, True, False, 3.0),
            ('unknown_blog', False, True, False, 2.5),
            ('unknown_blog', False, False, True, 1.0),
            ('unknown_blog', True, False, False, 4.5),
            ('unknown_blog', False, True, False, 2.0),
            ('reuters', True, False, False, 9.5),
            ('reuters', True, False, False, 9.2),
            ('reuters', True, False, False, 9.4),
            ('reuters', True, False, False, 9.3),
            ('reuters', True, False, False, 9.1),
            ('reuters', True, False, False, 9.6),
            ('crypto_pump_daily', False, True, False, 2.0),
            ('crypto_pump_daily', False, True, False, 1.5),
            ('crypto_pump_daily', False, False, True, 0.5),
            ('crypto_pump_daily', False, True, False, 1.0),
            ('crypto_pump_daily', False, False, True, 0.5),
        ]
        
        for source, accurate, flagged, retracted, score in test_sources:
            engine.update_source_history(
                source_name=source,
                was_accurate=accurate,
                was_flagged=flagged,
                was_retracted=retracted,
                trust_score=score
            )
        
        self.stdout.write(self.style.SUCCESS("   Simulated history updates for 4 sources"))
        
        # Show updated source history
        self.stdout.write("\nUpdated Source History:")
        self.stdout.write("-" * 80)
        self.stdout.write(f"{'Source':<25} {'Articles':>10} {'Accurate':>10} {'Flagged':>10} {'Retracted':>10} {'Reliability':>12}")
        self.stdout.write("-" * 80)
        
        for source, record in sorted(engine.source_history.items()):
            self.stdout.write(
                f"{source:<25} "
                f"{record.total_articles:>10} "
                f"{record.accurate_articles:>10} "
                f"{record.flagged_articles:>10} "
                f"{record.retracted_articles:>10} "
                f"{record.reliability_score:>12.2f}"
            )
        
        self.stdout.write("-" * 80)
        
        # Test scoring with historical sources - NOW USING RAW FORMAT
        self.stdout.write("\n🧪 Testing Score Calculation with Source History (RAW DATA):")
        
        test_articles = [
            {
                'id': 'history_test_1',
                'platform': 'coindesk',
                'title': 'Bitcoin Analysis: Market Shows Stability',
                'description': 'Professional market analysis',
                'content': 'Bitcoin market analysis shows continued stability...',
                'source': {'title': 'CoinDesk', 'domain': 'coindesk.com'},
                'published_at': timezone.now().isoformat(),
            },
            {
                'id': 'history_test_2',
                'platform': 'unknown',
                'title': 'Bitcoin Prediction',
                'description': 'Analysis from unknown source',
                'content': 'Bitcoin price prediction...',
                'source': {'title': 'Unknown Blog', 'domain': 'unknown-blog.com'},
                'published_at': timezone.now().isoformat(),
            },
            {
                'id': 'history_test_3',
                'platform': 'reuters',
                'title': 'Cryptocurrency Regulation Update',
                'description': 'Official regulatory news',
                'content': 'Regulatory update from official sources...',
                'source': {'title': 'Reuters', 'domain': 'reuters.com'},
                'published_at': timezone.now().isoformat(),
            },
            {
                'id': 'history_test_4',
                'platform': 'unknown',
                'title': 'GET RICH QUICK WITH CRYPTO!!!',
                'description': 'Suspicious pump content',
                'content': 'Buy now before its too late! 100x guaranteed!',
                'source': {'title': 'Crypto Pump Daily', 'domain': 'pumpdaily.com'},
                'published_at': timezone.now().isoformat(),
            },
            {
                'id': 'history_test_5',
                'platform': 'unknown',
                'title': 'New DeFi Protocol Analysis',
                'description': 'Analysis from new source',
                'content': 'New DeFi protocol launches with innovative features...',
                'source': {'title': 'New Crypto Blog', 'domain': 'newcryptoblog.com'},
                'published_at': timezone.now().isoformat(),
            }
        ]
        
        self.stdout.write("\n" + "-" * 100)
        self.stdout.write(f"{'Source':<25} {'History Score':>15} {'Final Score':>12} {'Action':<20} {'History Impact':<20}")
        self.stdout.write("-" * 100)
        
        for article in test_articles:
            trust_score = engine.calculate_trust_score(article)
            action = engine.determine_content_action(trust_score)
            
            source_name = article['source']['title']
            
            if source_name.lower() in engine.source_history:
                record = engine.source_history[source_name.lower()]
                if record.total_articles >= 5:
                    history_impact = f"Based on {record.total_articles} articles"
                else:
                    history_impact = "Insufficient data"
            else:
                history_impact = "No history (penalty)"
            
            self.stdout.write(
                f"{source_name:<25} "
                f"{trust_score.source_history_score:>15.2f} "
                f"{trust_score.final_score:>12.2f} "
                f"{action['action']:<20} "
                f"{history_impact:<20}"
            )
        
        self.stdout.write("-" * 100)
        self.stdout.write(self.style.SUCCESS("\nSource History Tracking Test Completed"))
    
    def test_cross_reference(self):
        """Test cross-reference validation feature with RAW data"""
        self.stdout.write(self.style.SUCCESS("\n" + "="*60))
        self.stdout.write(self.style.SUCCESS("=== TESTING CROSS-REFERENCE VALIDATION (RAW DATA) ==="))
        self.stdout.write(self.style.SUCCESS("="*60 + "\n"))
        
        engine = get_credibility_engine()
        
        self.stdout.write("📝 Creating test scenario: Multiple sources reporting same event")
        self.stdout.write("   Event: 'SEC Approves Bitcoin ETF'\n")
        
        # Cross-verification data - RAW FORMAT
        cross_check_articles = [
            {
                'id': 'cross_1',
                'title': 'SEC Gives Green Light to Bitcoin ETF Applications',
                'description': 'The Securities and Exchange Commission has approved Bitcoin ETF.',
                'content': 'SEC approves Bitcoin ETF after years of applications...',
                'source': {'title': 'Reuters', 'domain': 'reuters.com'},
                'platform': 'reuters',
                'published_at': timezone.now().isoformat()
            },
            {
                'id': 'cross_2',
                'title': 'Bitcoin ETF Finally Approved by SEC',
                'description': 'SEC approves first Bitcoin exchange-traded fund.',
                'content': 'Historic decision as SEC approves Bitcoin ETF...',
                'source': {'title': 'Bloomberg', 'domain': 'bloomberg.com'},
                'platform': 'bloomberg',
                'published_at': timezone.now().isoformat()
            },
            {
                'id': 'cross_3',
                'title': 'Breaking: SEC Approves Bitcoin ETF - Historic Decision',
                'description': 'Federal regulators approve Bitcoin ETF after years of applications.',
                'content': 'The SEC has finally approved Bitcoin ETF products...',
                'source': {'title': 'CoinDesk', 'domain': 'coindesk.com'},
                'platform': 'coindesk',
                'published_at': timezone.now().isoformat()
            },
            {
                'id': 'cross_4',
                'title': 'Bitcoin ETF Gets Regulatory Approval',
                'description': 'SEC announces approval of Bitcoin ETF products.',
                'content': 'Bitcoin ETF approval marks new era for crypto...',
                'source': {'title': 'The Block', 'domain': 'theblock.co'},
                'platform': 'theblock',
                'published_at': timezone.now().isoformat()
            }
        ]
        
        # Test articles - RAW FORMAT
        test_cases = [
            { 
                'name': 'Article with strong cross-reference (same event)',
                'article': {
                    'id': 'test_cross_1',
                    'title': 'SEC Finally Approves Bitcoin ETF After Long Wait',
                    'description': 'The SEC has approved Bitcoin ETF applications.',
                    'content': 'Bitcoin ETF approval marks historic moment...',
                    'source': {'title': 'Crypto News', 'domain': 'cryptonews.com'},
                    'platform': 'cryptonews',
                    'published_at': timezone.now().isoformat(),
                },
                'cross_data': cross_check_articles
            },
            {
                'name': 'Article with partial cross-reference',
                'article': {
                    'id': 'test_cross_2',
                    'title': 'Bitcoin Regulation News: SEC Makes Decision',
                    'description': 'Regulatory update regarding Bitcoin and SEC.',
                    'content': 'SEC regulatory decision impacts Bitcoin market...',
                    'source': {'title': 'Crypto Slate', 'domain': 'cryptoslate.com'},
                    'platform': 'cryptoslate',
                    'published_at': timezone.now().isoformat(),
                },
                'cross_data': cross_check_articles[:2]
            },
            {
                'name': 'Article with NO cross-reference (unique claim)',
                'article': {
                    'id': 'test_cross_3',
                    'title': 'Exclusive: Secret Bitcoin Whale Sells Everything',
                    'description': 'A mysterious whale has sold all their Bitcoin holdings.',
                    'content': 'Exclusive report on mysterious Bitcoin whale...',
                    'source': {'title': 'Unknown Blog', 'domain': 'unknownblog.com'},
                    'platform': 'unknown',
                    'published_at': timezone.now().isoformat(),
                },
                'cross_data': []
            },
            {
                'name': 'Article with unrelated cross-reference data',
                'article': {
                    'id': 'test_cross_4',
                    'title': 'Ethereum 2.0 Staking Rewards Increase',
                    'description': 'Ethereum staking yields have gone up significantly.',
                    'content': 'Ethereum staking rewards see increase...',
                    'source': {'title': 'DeFi News', 'domain': 'definews.com'},
                    'platform': 'defi',
                    'published_at': timezone.now().isoformat(),
                },
                'cross_data': cross_check_articles
            }
        ]
        
        self.stdout.write("-" * 110)
        self.stdout.write(f"{'Test Case':<45} {'Cross Score':>12} {'Matches':>10} {'Sources':>25} {'Bonus':>15}")
        self.stdout.write("-" * 110)
        
        for case in test_cases:
            trust_score = engine.calculate_trust_score(
                case['article'], 
                cross_check_data=case['cross_data']
            )
            
            sources_str = ', '.join(trust_score.corroboration_sources[:2]) if trust_score.corroboration_sources else 'None'
            if len(trust_score.corroboration_sources) > 2:
                sources_str += f' +{len(trust_score.corroboration_sources)-2}'
            
            if trust_score.cross_reference_matches >= 3:
                bonus = "Strong (+2.0)"
            elif trust_score.cross_reference_matches >= 2:
                bonus = "Moderate (+1.5)"
            elif trust_score.cross_reference_matches >= 1:
                bonus = "Weak (+0.5)"
            else:
                bonus = "None"
            
            self.stdout.write(
                f"{case['name']:<45} "
                f"{trust_score.cross_check_score:>12.2f} "
                f"{trust_score.cross_reference_matches:>10} "
                f"{sources_str:>25} "
                f"{bonus:>15}"
            )
        
        self.stdout.write("-" * 110)
        self.stdout.write(self.style.SUCCESS("\nCross-Reference Validation Test Completed"))
    
    def test_sentiment_extremity(self):
        """Test sentiment extremity penalty feature with RAW data"""
        self.stdout.write(self.style.SUCCESS("\n" + "="*60))
        self.stdout.write(self.style.SUCCESS("=== TESTING SENTIMENT EXTREMITY PENALTY (RAW DATA) ==="))
        self.stdout.write(self.style.SUCCESS("="*60 + "\n"))
        
        engine = get_credibility_engine()
        
        test_cases = [
            {'name': 'Professional Analysis', 'title': 'Bitcoin Price Analysis: Technical Indicators Show Consolidation Phase', 'expected': 'Low/No penalty'},
            {'name': 'Neutral News', 'title': 'Ethereum Network Successfully Completes Scheduled Upgrade', 'expected': 'Low/No penalty'},
            {'name': 'Positive News', 'title': 'Bitcoin Adoption Growing Steadily Among Institutional Investors', 'expected': 'Low penalty'},
            {'name': 'Negative News', 'title': 'Cryptocurrency Market Faces Challenges Amid Regulatory Uncertainty', 'expected': 'Low penalty'},
            {'name': 'Hype Content', 'title': 'This Altcoin Will Make You Rich - Dont Miss The Opportunity!', 'expected': 'Moderate penalty'},
            {'name': 'Extreme Hype', 'title': 'BITCOIN TO THE MOON! 100X GAINS GUARANTEED! GET IN NOW!', 'expected': 'High penalty'},
            {'name': 'Pump Scheme', 'title': '🚀🚀🚀 MASSIVE PUMP INCOMING! EASY MONEY! GUARANTEED PROFITS! 🚀🚀🚀', 'expected': 'Very high penalty'},
            {'name': 'Fear Content', 'title': 'Warning: Bitcoin Could Drop Significantly in Coming Weeks', 'expected': 'Low/Moderate penalty'},
            {'name': 'Panic Content', 'title': 'URGENT: Crypto Market Crash Imminent - Sell Before Its Too Late!', 'expected': 'High penalty'},
            {'name': 'Extreme Panic', 'title': 'EMERGENCY!!! CRYPTO COLLAPSE! EVERYTHING IS CRASHING! SELL NOW!!!', 'expected': 'Very high penalty'}
        ]
        
        self.stdout.write("Testing sentiment extremity detection and penalties:\n")
        self.stdout.write("-" * 120)
        self.stdout.write(f"{'Type':<20} {'Sentiment Score':>15} {'Extremity Penalty':>18} {'Expected':>20} {'Flags':<45}")
        self.stdout.write("-" * 120)
        
        for case in test_cases:
            # RAW FORMAT - no pre-analyzed fields
            article = {
                'id': f"extremity_test_{case['name']}",
                'title': case['title'],
                'description': f"Test content for {case['name']}",
                'content': f"Content related to {case['title']}",
                'source': {'title': 'Test Source', 'domain': 'test.com'},
                'platform': 'test',
                'published_at': timezone.now().isoformat(),
            }
            
            trust_score = engine.calculate_trust_score(article)
            
            extremity_flags = [f for f in trust_score.flags if 'extremity' in f or 'extreme' in f or 'hype' in f or 'panic' in f]
            flags_str = ', '.join(extremity_flags[:3]) if extremity_flags else 'None'
            
            penalty = trust_score.sentiment_extremity_penalty
            if penalty >= 5.0:
                penalty_level = "🔴 Very High"
            elif penalty >= 3.0:
                penalty_level = "🟠 High"
            elif penalty >= 1.5:
                penalty_level = "🟡 Moderate"
            elif penalty > 0:
                penalty_level = "🟢 Low"
            else:
                penalty_level = "⚪ None"
            
            self.stdout.write(
                f"{case['name']:<20} "
                f"{trust_score.sentiment_score:>15.2f} "
                f"{penalty:>10.1f} ({penalty_level}) "
                f"{case['expected']:>20} "
                f"{flags_str:<45}"
            )
        
        self.stdout.write("-" * 120)
        self.stdout.write(self.style.SUCCESS("\nSentiment Extremity Penalty Test Completed"))
    
    def test_formula_weights(self):
        """Test that the formula weights are correctly configured"""
        self.stdout.write(self.style.SUCCESS("\n" + "="*60))
        self.stdout.write(self.style.SUCCESS("=== FORMULA WEIGHT VERIFICATION ==="))
        self.stdout.write(self.style.SUCCESS("="*60 + "\n"))
        
        engine = get_credibility_engine()
        
        expected_weights = {
            'source': 0.35,
            'content': 0.20,
            'sentiment': 0.15,
            'cross_check': 0.15,
            'source_history': 0.10,
            'recency': 0.05
        }
        
        self.stdout.write("Expected Formula:")
        self.stdout.write("  Final = 0.35(source) + 0.20(content) + 0.15(sentiment) + 0.15(cross_ref) + 0.10(history) + 0.05(recency)")
        self.stdout.write("")
        
        self.stdout.write("Configured Weights:")
        total_weight = 0
        all_correct = True
        
        for component, expected in expected_weights.items():
            actual = engine.weights.get(component, 0)
            total_weight += actual
            status = "✅" if abs(actual - expected) < 0.001 else "❌"
            if abs(actual - expected) >= 0.001:
                all_correct = False
            self.stdout.write(f"  {component}: {actual:.2f} (expected: {expected:.2f}) {status}")
        
        self.stdout.write(f"\n  Total Weight: {total_weight:.2f} (should be 1.00)")
        
        if all_correct and abs(total_weight - 1.0) < 0.001:
            self.stdout.write(self.style.SUCCESS("\nFormula weights are correctly configured!"))
        else:
            self.stdout.write(self.style.ERROR("\nFormula weights need adjustment!"))
    
    def test_credibility_scoring(self, sample_size: int, verbose: bool = False):
        """Test credibility scoring engine with RAW sample data"""
        self.stdout.write(self.style.SUCCESS("\n" + "="*60))
        self.stdout.write(self.style.SUCCESS("=== TESTING CREDIBILITY SCORING ENGINE (RAW DATA) ==="))
        self.stdout.write(self.style.SUCCESS("="*60 + "\n"))
        
        self.stdout.write("Formula: Final = 0.35(source) + 0.20(content) + 0.15(sentiment) + 0.15(cross_ref) + 0.10(history) + 0.05(recency)\n")
        
        sample_articles = self.generate_sample_articles(sample_size)
        
        engine = get_credibility_engine()
        
        results = []
        for i, article in enumerate(sample_articles, 1):
            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(f"Article {i}/{len(sample_articles)}: {article['title'][:50]}...")
            
            trust_score = engine.calculate_trust_score(article)
            action_info = engine.determine_content_action(trust_score)
            
            result = {
                'title': article['title'][:50] + '...',
                'platform': article['platform'],
                'trust_score': round(trust_score.final_score, 2),
                'source_score': round(trust_score.source_score, 2),
                'content_score': round(trust_score.content_score, 2),
                'sentiment_score': round(trust_score.sentiment_score, 2),
                'cross_check_score': round(trust_score.cross_check_score, 2),
                'source_history_score': round(trust_score.source_history_score, 2),
                'recency_score': round(trust_score.recency_score, 2),
                'action': action_info['action'],
                'flags': trust_score.flags,
                'extremity_penalty': trust_score.sentiment_extremity_penalty,
                'cross_ref_matches': trust_score.cross_reference_matches,
                'reasoning': trust_score.reasoning
            }
            results.append(result)
            
            self.stdout.write(f"\n  SCORE BREAKDOWN:")
            self.stdout.write(f"     Source Score:        {result['source_score']:>5.2f} × 0.35 = {result['source_score'] * 0.35:.2f}")
            self.stdout.write(f"     Content Score:       {result['content_score']:>5.2f} × 0.20 = {result['content_score'] * 0.20:.2f}")
            self.stdout.write(f"     Sentiment Score:     {result['sentiment_score']:>5.2f} × 0.15 = {result['sentiment_score'] * 0.15:.2f}")
            self.stdout.write(f"     Cross-Ref Score:     {result['cross_check_score']:>5.2f} × 0.15 = {result['cross_check_score'] * 0.15:.2f}")
            self.stdout.write(f"     Source History:      {result['source_history_score']:>5.2f} × 0.10 = {result['source_history_score'] * 0.10:.2f}")
            self.stdout.write(f"     Recency Score:       {result['recency_score']:>5.2f} × 0.05 = {result['recency_score'] * 0.05:.2f}")
            self.stdout.write(f"     {'─'*45}")
            self.stdout.write(f"     FINAL SCORE:         {result['trust_score']:>5.2f}/10")
            
            self.stdout.write(f"\n  Action: {result['action']}")
            
            if result['extremity_penalty'] > 0:
                self.stdout.write(f"  ️  Sentiment Extremity Penalty: -{result['extremity_penalty']:.1f}")
            
            if result['cross_ref_matches'] > 0:
                self.stdout.write(f"  Cross-Reference Matches: {result['cross_ref_matches']}")
            
            if result['flags']:
                self.stdout.write(f"  🚩 Flags: {', '.join(result['flags'])}")
            
            if verbose:
                self.stdout.write(f"  💭 Reasoning: {result['reasoning']}")
        
        # Summary
        self._print_scoring_summary(results)
    
    def _print_scoring_summary(self, results: List[Dict]):
        """Print scoring summary"""
        self.stdout.write(self.style.SUCCESS(f"\n{'='*60}"))
        self.stdout.write(self.style.SUCCESS("=== SCORING SUMMARY ==="))
        self.stdout.write(f"{'='*60}\n")
        
        avg_score = sum(r['trust_score'] for r in results) / len(results)
        avg_source = sum(r['source_score'] for r in results) / len(results)
        avg_content = sum(r['content_score'] for r in results) / len(results)
        avg_sentiment = sum(r['sentiment_score'] for r in results) / len(results)
        avg_cross = sum(r['cross_check_score'] for r in results) / len(results)
        avg_history = sum(r['source_history_score'] for r in results) / len(results)
        avg_recency = sum(r['recency_score'] for r in results) / len(results)
        
        self.stdout.write(f"Average Scores:")
        self.stdout.write(f"  Final Trust Score:    {avg_score:.2f}/10")
        self.stdout.write(f"  Source Score:         {avg_source:.2f}/10")
        self.stdout.write(f"  Content Score:        {avg_content:.2f}/10")
        self.stdout.write(f"  Sentiment Score:      {avg_sentiment:.2f}/10")
        self.stdout.write(f"  Cross-Ref Score:      {avg_cross:.2f}/10")
        self.stdout.write(f"  Source History Score: {avg_history:.2f}/10")
        self.stdout.write(f"  Recency Score:        {avg_recency:.2f}/10")
        
        actions_count = {}
        for r in results:
            actions_count[r['action']] = actions_count.get(r['action'], 0) + 1
        
        self.stdout.write(f"\nAction Distribution:")
        for action, count in sorted(actions_count.items()):
            pct = (count / len(results)) * 100
            bar = '█' * int(pct / 5)
            self.stdout.write(f"  {action:20s}: {count:3d} ({pct:5.1f}%) {bar}")
        
        high_extremity = sum(1 for r in results if r['extremity_penalty'] > 2.0)
        moderate_extremity = sum(1 for r in results if 0 < r['extremity_penalty'] <= 2.0)
        
        self.stdout.write(f"\nSentiment Extremity:")
        self.stdout.write(f"  High Extremity (penalty > 2.0):     {high_extremity}")
        self.stdout.write(f"  Moderate Extremity (penalty 0-2.0): {moderate_extremity}")
        self.stdout.write(f"  Normal Sentiment:                   {len(results) - high_extremity - moderate_extremity}")
        
        all_flags = [f for r in results for f in r['flags']]
        if all_flags:
            flag_counts = Counter(all_flags)
            self.stdout.write(f"\nFlag Frequency:")
            for flag, count in flag_counts.most_common(10):
                self.stdout.write(f"  {flag}: {count}")
    
    def test_social_scoring(self, verbose: bool = False):
        """Test credibility scoring for social media posts with RAW data"""
        self.stdout.write(self.style.SUCCESS("\n" + "="*60))
        self.stdout.write(self.style.SUCCESS("=== TESTING SOCIAL POST CREDIBILITY SCORING (RAW DATA) ==="))
        self.stdout.write(self.style.SUCCESS("="*60 + "\n"))
        
        engine = get_credibility_engine()
        
        # Reddit posts - RAW FORMAT (matching fetcher output)
        reddit_posts = [
            {
                'name': 'High Credibility Reddit (Mod + High Karma)',
                'post': {
                    'id': 'reddit_high_cred',
                    'platform': 'reddit',
                    'title': 'Bitcoin Technical Analysis - Support Levels',
                    'selftext': 'Detailed analysis of BTC support and resistance levels with chart patterns...',
                    'subreddit': 'cryptocurrency',
                    'author': 'crypto_analyst_pro',
                    'created_utc': timezone.now().timestamp(),
                    'score': 1500,
                    'upvote_ratio': 0.96,
                    'num_comments': 250,
                    'total_awards_received': 5,
                    'author_info': {
                        'name': 'crypto_analyst_pro',
                        'created_utc': (timezone.now() - timedelta(days=2000)).timestamp(),
                        'link_karma': 30000,
                        'comment_karma': 70000,
                        'is_mod': True,
                        'is_gold': True,
                        'has_verified_email': True
                    },
                    'subreddit_info': {
                        'display_name': 'cryptocurrency',
                        'subscribers': 6000000,
                        'created_utc': (timezone.now() - timedelta(days=3000)).timestamp()
                    }
                }
            },
            {
                'name': 'Medium Credibility Reddit',
                'post': {
                    'id': 'reddit_med_cred',
                    'platform': 'reddit',
                    'title': 'What do you think about ETH?',
                    'selftext': 'Just bought some ETH, thoughts?',
                    'subreddit': 'ethtrader',
                    'author': 'eth_holder_123',
                    'created_utc': timezone.now().timestamp(),
                    'score': 50,
                    'upvote_ratio': 0.75,
                    'num_comments': 20,
                    'total_awards_received': 0,
                    'author_info': {
                        'name': 'eth_holder_123',
                        'created_utc': (timezone.now() - timedelta(days=180)).timestamp(),
                        'link_karma': 500,
                        'comment_karma': 1000,
                        'is_mod': False,
                        'is_gold': False
                    },
                    'subreddit_info': {
                        'display_name': 'ethtrader',
                        'subscribers': 500000
                    }
                }
            },
            {
                'name': 'Low Credibility Reddit (New Account)',
                'post': {
                    'id': 'reddit_low_cred',
                    'platform': 'reddit',
                    'title': 'THIS COIN WILL 100X!!!',
                    'selftext': 'Trust me bro, buy now before its too late!',
                    'subreddit': 'cryptomoonshots',
                    'author': 'moonshot_king',
                    'created_utc': timezone.now().timestamp(),
                    'score': 5,
                    'upvote_ratio': 0.45,
                    'num_comments': 30,
                    'total_awards_received': 0,
                    'author_info': {
                        'name': 'moonshot_king',
                        'created_utc': (timezone.now() - timedelta(days=7)).timestamp(),
                        'link_karma': 10,
                        'comment_karma': 40,
                        'is_mod': False,
                        'is_gold': False
                    },
                    'subreddit_info': {
                        'display_name': 'cryptomoonshots',
                        'subscribers': 100000
                    }
                }
            }
        ]
        
        # Twitter posts - RAW FORMAT
        twitter_posts = [
            {
                'name': 'High Credibility Twitter (Verified + High Followers)',
                'post': {
                    'id': 'twitter_high_cred',
                    'platform': 'twitter',
                    'text': 'Bitcoin showing strong support at $95K. Key levels to watch for the week ahead.',
                    'created_at': timezone.now().isoformat(),
                    'public_metrics': {
                        'like_count': 5000,
                        'retweet_count': 1500,
                        'reply_count': 300,
                        'quote_count': 100
                    },
                    'user_info': {
                        'id': '123456',
                        'username': 'verified_analyst',
                        'verified': True,
                        'verified_type': 'business',
                        'created_at': (timezone.now() - timedelta(days=3000)).isoformat(),
                        'public_metrics': {
                            'followers_count': 500000,
                            'following_count': 1000,
                            'tweet_count': 10000,
                            'listed_count': 500
                        }
                    }
                }
            },
            {
                'name': 'Medium Credibility Twitter',
                'post': {
                    'id': 'twitter_med_cred',
                    'platform': 'twitter',
                    'text': 'ETH looking bullish today',
                    'created_at': timezone.now().isoformat(),
                    'public_metrics': {
                        'like_count': 50,
                        'retweet_count': 10,
                        'reply_count': 5,
                        'quote_count': 2
                    },
                    'user_info': {
                        'id': '789012',
                        'username': 'crypto_trader',
                        'verified': False,
                        'created_at': (timezone.now() - timedelta(days=500)).isoformat(),
                        'public_metrics': {
                            'followers_count': 5000,
                            'following_count': 2000,
                            'tweet_count': 3000,
                            'listed_count': 10
                        }
                    }
                }
            },
            {
                'name': 'Low Credibility Twitter (Bot-like)',
                'post': {
                    'id': 'twitter_low_cred',
                    'platform': 'twitter',
                    'text': '🚀🚀🚀 $SCAM COIN TO THE MOON! 1000X GUARANTEED! 🚀🚀🚀',
                    'created_at': timezone.now().isoformat(),
                    'public_metrics': {
                        'like_count': 10,
                        'retweet_count': 50,
                        'reply_count': 2,
                        'quote_count': 0
                    },
                    'user_info': {
                        'id': '345678',
                        'username': 'crypto_pumper',
                        'verified': False,
                        'created_at': (timezone.now() - timedelta(days=30)).isoformat(),
                        'public_metrics': {
                            'followers_count': 100,
                            'following_count': 5000,
                            'tweet_count': 50000,
                            'listed_count': 0
                        }
                    }
                }
            }
        ]
        
        # YouTube posts - RAW FORMAT
        youtube_posts = [
            {
                'name': 'High Credibility YouTube (Large Channel)',
                'post': {
                    'id': 'youtube_high_cred',
                    'video_id': 'abc123',
                    'platform': 'youtube',
                    'title': 'Bitcoin Technical Analysis - Full Breakdown',
                    'description': 'Comprehensive analysis of Bitcoin price action...',
                    'channel_id': 'UC123',
                    'channel_title': 'CryptoEducator',
                    'published_at': timezone.now().isoformat(),
                    'view_count': 500000,
                    'like_count': 25000,
                    'comment_count': 2000,
                    'duration_seconds': 1800,
                    'caption': 'Full transcript of technical analysis...',
                    'channel_info': {
                        'subscriber_count': 1000000,
                        'subscriber_count_hidden': False,
                        'total_view_count': 100000000,
                        'video_count': 500,
                        'channel_created': (timezone.now() - timedelta(days=2500)).isoformat()
                    }
                }
            },
            {
                'name': 'Low Credibility YouTube (Small Channel, Short Video)',
                'post': {
                    'id': 'youtube_low_cred',
                    'video_id': 'xyz789',
                    'platform': 'youtube',
                    'title': 'GET RICH QUICK WITH CRYPTO!!!',
                    'description': 'This coin will make you rich...',
                    'channel_id': 'UC456',
                    'channel_title': 'CryptoMillionaire',
                    'published_at': timezone.now().isoformat(),
                    'view_count': 500,
                    'like_count': 20,
                    'comment_count': 5,
                    'duration_seconds': 45,
                    'channel_info': {
                        'subscriber_count': 100,
                        'subscriber_count_hidden': False,
                        'total_view_count': 5000,
                        'video_count': 10,
                        'channel_created': (timezone.now() - timedelta(days=30)).isoformat()
                    }
                }
            }
        ]
        
        all_posts = reddit_posts + twitter_posts + youtube_posts
        
        self.stdout.write("-" * 100)
        self.stdout.write(f"{'Platform':<10} {'Test Case':<45} {'Source':>8} {'Content':>8} {'Sentiment':>10} {'Final':>8} {'Action':<15}")
        self.stdout.write("-" * 100)
        
        for test in all_posts:
            trust_score = engine.calculate_trust_score(test['post'])
            action = engine.determine_content_action(trust_score)
            
            platform = test['post']['platform'].upper()
            
            self.stdout.write(
                f"{platform:<10} "
                f"{test['name']:<45} "
                f"{trust_score.source_score:>8.2f} "
                f"{trust_score.content_score:>8.2f} "
                f"{trust_score.sentiment_score:>10.2f} "
                f"{trust_score.final_score:>8.2f} "
                f"{action['action']:<15}"
            )
            
            if verbose:
                self.stdout.write(f"           Reasoning: {trust_score.reasoning[:80]}...")
                if trust_score.flags:
                    self.stdout.write(f"           Flags: {', '.join(trust_score.flags[:3])}")
        
        self.stdout.write("-" * 100)
        
        self.stdout.write("\n📖 Social Platform Scoring Explanation:")
        self.stdout.write("\n  REDDIT (RAW fields used):")
        self.stdout.write("    • author_info.link_karma, comment_karma → karma score")
        self.stdout.write("    • author_info.created_utc → account age")
        self.stdout.write("    • author_info.is_mod → moderator bonus")
        self.stdout.write("    • score, upvote_ratio, num_comments → engagement")
        
        self.stdout.write("\n  TWITTER (RAW fields used):")
        self.stdout.write("    • user_info.verified, verified_type → verification bonus")
        self.stdout.write("    • user_info.public_metrics.followers_count → follower score")
        self.stdout.write("    • public_metrics.like_count, retweet_count → engagement")
        
        self.stdout.write("\n  YOUTUBE (RAW fields used):")
        self.stdout.write("    • channel_info.subscriber_count → channel credibility")
        self.stdout.write("    • view_count, like_count, comment_count → engagement")
        self.stdout.write("    • duration_seconds → content depth")
        self.stdout.write("    • caption/transcript → quality indicator")
        
        self.stdout.write(self.style.SUCCESS("\nSocial Post Scoring Test Completed"))

    def test_social_pipeline(self, sample_size: int, verbose: bool = False):
        """Test complete pipeline for social posts with RAW data"""
        self.stdout.write(self.style.SUCCESS("\n" + "="*60))
        self.stdout.write(self.style.SUCCESS("=== TESTING SOCIAL POST FULL PIPELINE (RAW DATA) ==="))
        self.stdout.write(self.style.SUCCESS("="*60 + "\n"))
        
        from myapp.services.content.integrator_service import ContentIntegrationService
        
        service = ContentIntegrationService()
        sample_posts = self.generate_sample_social_posts(sample_size)
        
        self.stdout.write(f"Processing {len(sample_posts)} social posts through full pipeline...")
        
        reddit_posts = [p for p in sample_posts if p['platform'] == 'reddit']
        twitter_posts = [p for p in sample_posts if p['platform'] == 'twitter']
        youtube_posts = [p for p in sample_posts if p['platform'] == 'youtube']
        
        results = {}
        
        if reddit_posts:
            self.stdout.write(f"\nProcessing {len(reddit_posts)} Reddit posts...")
            result = service.process_social_posts_batch(reddit_posts, 'reddit')
            results['reddit'] = result
            self._display_batch_result(result, 'Reddit', verbose)
        
        if twitter_posts:
            self.stdout.write(f"\nProcessing {len(twitter_posts)} Twitter posts...")
            result = service.process_social_posts_batch(twitter_posts, 'twitter')
            results['twitter'] = result
            self._display_batch_result(result, 'Twitter', verbose)
        
        if youtube_posts:
            self.stdout.write(f"\nProcessing {len(youtube_posts)} YouTube posts...")
            result = service.process_social_posts_batch(youtube_posts, 'youtube')
            results['youtube'] = result
            self._display_batch_result(result, 'YouTube', verbose)
        
        # Summary
        self.stdout.write(self.style.SUCCESS(f"\n{'='*60}"))
        self.stdout.write(self.style.SUCCESS("=== SOCIAL PIPELINE SUMMARY ==="))
        self.stdout.write(f"{'='*60}\n")
        
        total_processed = sum(r.total_processed for r in results.values())
        total_approved = sum(r.approved for r in results.values())
        total_pending = sum(r.pending for r in results.values())
        total_flagged = sum(r.flagged for r in results.values())
        
        self.stdout.write(f"Total Processed: {total_processed}")
        self.stdout.write(f"Total Approved: {total_approved} ({total_approved/max(total_processed,1)*100:.1f}%)")
        self.stdout.write(f"Total Pending: {total_pending} ({total_pending/max(total_processed,1)*100:.1f}%)")
        self.stdout.write(f"Total Flagged: {total_flagged} ({total_flagged/max(total_processed,1)*100:.1f}%)")
        
        self.stdout.write(self.style.SUCCESS("\nSocial Pipeline Test Completed"))
    
    def _display_batch_result(self, result, platform: str, verbose: bool):
        """Display batch processing result"""
        self.stdout.write(f"  Processed: {result.total_processed}")
        self.stdout.write(f"  Approved: {result.approved}")
        self.stdout.write(f"  Pending: {result.pending}")
        self.stdout.write(f"  Flagged: {result.flagged}")
        self.stdout.write(f"  Errors: {result.errors}")
        self.stdout.write(f"  Avg Trust Score: {result.average_trust_score:.2f}")
        self.stdout.write(f"  Processing Time: {result.processing_time_seconds:.2f}s")
        
    def generate_sample_social_posts(self, count: int) -> list:
        """Generate sample social posts in RAW format (matching fetcher output)"""
        posts = []
        platforms = ['reddit', 'twitter', 'youtube']
        
        for i in range(count):
            platform = random.choice(platforms)
            
            if platform == 'reddit':
                post = self._generate_reddit_post(i)
            elif platform == 'twitter':
                post = self._generate_twitter_post(i)
            else:
                post = self._generate_youtube_post(i)
            
            posts.append(post)
        
        return posts
    
    def _generate_reddit_post(self, index: int) -> dict:
        """Generate a sample Reddit post in RAW format"""
        karma_tiers = [(100, 50), (5000, 2000), (50000, 30000)]
        karma = random.choice(karma_tiers)
        is_mod = random.random() > 0.9
        upvote_ratio = random.uniform(0.5, 0.98)
        account_age_days = random.randint(7, 2000)
        
        titles = [
            'Bitcoin Analysis - Key Support Levels',
            'What do you think about ETH?',
            'THIS COIN WILL MOON!!!',
            'Serious Discussion: Crypto Regulations',
            'Market Update: BTC Price Action'
        ]
        
        return {
            'id': f'reddit_sample_{index}',
            'platform': 'reddit',
            'title': random.choice(titles),
            'selftext': 'Sample content for testing...',
            'subreddit': random.choice(['cryptocurrency', 'bitcoin', 'ethtrader', 'cryptomoonshots']),
            'author': f'user_{index}',
            'created_utc': (timezone.now() - timedelta(hours=random.randint(1, 48))).timestamp(),
            'score': random.randint(10, 2000),
            'upvote_ratio': upvote_ratio,
            'num_comments': random.randint(5, 500),
            'total_awards_received': random.randint(0, 10),
            'author_info': {
                'name': f'user_{index}',
                'created_utc': (timezone.now() - timedelta(days=account_age_days)).timestamp(),
                'link_karma': karma[1],
                'comment_karma': karma[0] - karma[1],
                'is_mod': is_mod,
                'is_gold': random.random() > 0.8,
                'has_verified_email': random.random() > 0.3
            },
            'subreddit_info': {
                'display_name': random.choice(['cryptocurrency', 'bitcoin', 'ethtrader']),
                'subscribers': random.randint(50000, 6000000)
            }
        }
    
    def _generate_twitter_post(self, index: int) -> dict:
        """Generate a sample Twitter post in RAW format"""
        follower_tiers = [100, 5000, 50000, 500000]
        followers = random.choice(follower_tiers)
        verified = followers > 100000 and random.random() > 0.5
        account_age_days = random.randint(30, 3000)
        
        texts = [
            'BTC looking strong today 📈',
            '🚀🚀🚀 MOON SOON!!! 🚀🚀🚀',
            'Market analysis thread 🧵',
            'Interesting development in DeFi...',
            'URGENT: Sell everything now!!!'
        ]
        
        return {
            'id': f'twitter_sample_{index}',
            'platform': 'twitter',
            'text': random.choice(texts),
            'created_at': (timezone.now() - timedelta(hours=random.randint(1, 48))).isoformat(),
            'public_metrics': {
                'like_count': random.randint(10, 10000),
                'retweet_count': random.randint(5, 2000),
                'reply_count': random.randint(2, 500),
                'quote_count': random.randint(0, 100)
            },
            'user_info': {
                'id': f'user_{index}',
                'username': f'crypto_user_{index}',
                'verified': verified,
                'verified_type': 'blue' if verified else None,
                'created_at': (timezone.now() - timedelta(days=account_age_days)).isoformat(),
                'public_metrics': {
                    'followers_count': followers,
                    'following_count': random.randint(100, followers * 2),
                    'tweet_count': random.randint(100, 50000),
                    'listed_count': random.randint(0, 100)
                }
            }
        }
    
    def _generate_youtube_post(self, index: int) -> dict:
        """Generate a sample YouTube post in RAW format"""
        sub_tiers = [100, 10000, 100000, 1000000]
        subscribers = random.choice(sub_tiers)
        channel_age_days = random.randint(30, 2500)
        
        titles = [
            'Bitcoin Price Prediction 2025',
            'EMERGENCY: Crypto Market Update',
            'Technical Analysis - BTC, ETH, SOL',
            'How to Get Rich with Crypto!!!',
            'Crypto News Daily Recap'
        ]
        
        return {
            'id': f'youtube_sample_{index}',
            'video_id': f'vid_{index}',
            'platform': 'youtube',
            'title': random.choice(titles),
            'description': 'Video description...',
            'channel_id': f'UC{index}',
            'channel_title': f'CryptoChannel_{index}',
            'published_at': (timezone.now() - timedelta(hours=random.randint(1, 48))).isoformat(),
            'view_count': random.randint(100, 500000),
            'like_count': random.randint(10, 25000),
            'comment_count': random.randint(5, 2000),
            'duration_seconds': random.choice([45, 300, 900, 1800, 3600]),
            'caption': 'Video transcript...' if random.random() > 0.3 else None,
            'channel_info': {
                'subscriber_count': subscribers,
                'subscriber_count_hidden': False,
                'total_view_count': subscribers * random.randint(50, 200),
                'video_count': random.randint(10, 500),
                'channel_created': (timezone.now() - timedelta(days=channel_age_days)).isoformat()
            }
        }
        
    def test_mongodb_storage(self, sample_size: int):
        """Test MongoDB storage functionality with RAW data"""
        self.stdout.write(self.style.SUCCESS("\n=== TESTING MONGODB STORAGE (RAW DATA) ===\n"))
        
        mongo_manager = get_mongo_manager()
        
        try:
            stats = mongo_manager.get_statistics()
            self.stdout.write(f"MongoDB connected. Current stats: {stats}")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"MongoDB connection failed: {e}"))
            return
        
        sample_articles = self.generate_sample_articles(sample_size)
        
        self.stdout.write(f"Inserting {len(sample_articles)} sample articles...")
        
        insert_stats = mongo_manager.bulk_insert_articles(sample_articles)
        self.stdout.write(f"Insert results: {insert_stats}")
        
        self.stdout.write("Testing queries...")
        
        high_cred_articles = mongo_manager.get_high_credibility_articles(
            trust_score_threshold=7.0,
            limit=10
        ) 
        self.stdout.write(f"Found {len(high_cred_articles)} high credibility articles")
        
        pending_content = mongo_manager.get_pending_content_for_review(limit=5)
        self.stdout.write(f"Pending content: {pending_content['total']} items")
        
        topic_content = mongo_manager.get_content_by_topic(['bitcoin', 'ethereum'])
        self.stdout.write(f"Topic search found {topic_content['total']} items")
        
        self.stdout.write(self.style.SUCCESS("MongoDB storage test completed"))
    
    def test_full_pipeline(self, sample_size: int, verbose: bool = False):
        """Test the complete pipeline from scoring to storage"""
        self.stdout.write(self.style.SUCCESS("\n=== TESTING FULL PIPELINE ===\n"))
        
        # First verify formula
        self.test_formula_weights()
        
        # Initialize components
        mongo_manager = get_mongo_manager()
        credibility_engine = get_credibility_engine()
        
        # Generate sample data
        sample_articles = self.generate_sample_articles(sample_size)
        
        self.stdout.write(f"\nProcessing {len(sample_articles)} articles through full pipeline...")
        
        processed_articles = []
        trust_scores = []
        
        for i, article in enumerate(sample_articles, 1):
            self.stdout.write(f"\nProcessing article {i}/{len(sample_articles)}: {article['title'][:40]}...")
            
            try:
                # Step 1: Calculate trust score
                trust_score = credibility_engine.calculate_trust_score(article)
                trust_scores.append(trust_score)
                
                # Step 2: Determine action
                action_info = credibility_engine.determine_content_action(trust_score)
                
                # Step 3: Add trust score to article data (with new fields)
                article['trust_score'] = trust_score.final_score
                article['credibility_analysis'] = {
                    'trust_score_breakdown': {
                        'source_score': trust_score.source_score,
                        'content_score': trust_score.content_score,
                        'sentiment_score': trust_score.sentiment_score,
                        'cross_check_score': trust_score.cross_check_score,
                        'source_history_score': trust_score.source_history_score,
                        'recency_score': trust_score.recency_score
                    },
                    'weights': credibility_engine.weights,  # Use actual engine weights
                    'flags': trust_score.flags,
                    'reasoning': trust_score.reasoning,
                    'confidence': trust_score.confidence,
                    'sentiment_extremity_penalty': trust_score.sentiment_extremity_penalty,
                    'cross_reference_matches': trust_score.cross_reference_matches,
                    'corroboration_sources': trust_score.corroboration_sources,
                    'recommended_action': action_info
                }
                
                # Step 4: Set status based on action
                if action_info['action'] == 'auto_approve':
                    article['status'] = 'approved'
                elif action_info['action'] == 'manual_review':
                    article['status'] = 'flagged'
                else:
                    article['status'] = 'pending'
                
                processed_articles.append(article)
                
                # Show brief result (updated to include content score)
                self.stdout.write(
                    f"  Score: {trust_score.final_score:.2f} | "
                    f"Src:{trust_score.source_score:.1f} Cnt:{trust_score.content_score:.1f} "
                    f"Snt:{trust_score.sentiment_score:.1f} Xref:{trust_score.cross_check_score:.1f} "
                    f"Hist:{trust_score.source_history_score:.1f} Rec:{trust_score.recency_score:.1f} | "
                    f"Action: {action_info['action']}"
                )
                
            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(f"Error processing article {i}: {e}")
                )
        
        # Step 5: Bulk insert to MongoDB
        self.stdout.write("\nStoring processed articles in MongoDB...")
        insert_stats = mongo_manager.bulk_insert_articles(processed_articles)
        self.stdout.write(f"Storage results: {insert_stats}")
        
        # Step 6: Generate comprehensive statistics
        self.stdout.write(self.style.SUCCESS(f"\n{'='*60}"))
        self.stdout.write(self.style.SUCCESS("=== PIPELINE RESULTS ==="))
        self.stdout.write(f"{'='*60}\n")
        
        # Use the engine's statistics method
        stats = credibility_engine.get_scoring_statistics(trust_scores)
        
        self.stdout.write(f"Score Distribution:")
        self.stdout.write(f"  Mean:   {stats['score_distribution']['mean']:.2f}")
        self.stdout.write(f"  Min:    {stats['score_distribution']['min']:.2f}")
        self.stdout.write(f"  Max:    {stats['score_distribution']['max']:.2f}")
        self.stdout.write(f"  Median: {stats['score_distribution']['median']:.2f}")
        
        self.stdout.write(f"\nThreshold Distribution:")
        for threshold, count in stats['threshold_distribution'].items():
            pct = (count / len(trust_scores)) * 100
            self.stdout.write(f"  {threshold:15s}: {count:3d} ({pct:5.1f}%)")
        
        self.stdout.write(f"\nCross-Reference Stats:")
        self.stdout.write(f"  Avg Matches:      {stats['cross_reference_stats']['avg_matches']:.2f}")
        self.stdout.write(f"  Verified Content: {stats['cross_reference_stats']['verified_content']}")
        
        self.stdout.write(f"\nExtremity Stats:")
        self.stdout.write(f"  High Extremity:     {stats['extremity_stats']['high_extremity']}")
        self.stdout.write(f"  Moderate Extremity: {stats['extremity_stats']['moderate_extremity']}")
        
        self.stdout.write(f"\nWeights Used: {stats['weights_used']}")
        
        # Status distribution
        status_counts = {}
        for article in processed_articles:
            status = article.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        self.stdout.write(f"\nStatus Distribution:")
        for status, count in status_counts.items():
            pct = (count / len(processed_articles)) * 100
            self.stdout.write(f"  {status}: {count} ({pct:.1f}%)")
        
        # Score distribution by platform
        platform_scores = {}
        for article in processed_articles:
            platform = article.get('platform', 'unknown')
            score = article.get('trust_score', 0)
            if platform not in platform_scores:
                platform_scores[platform] = []
            platform_scores[platform].append(score)
        
        self.stdout.write(f"\nAverage Scores by Platform:")
        for platform, scores in sorted(platform_scores.items()):
            avg_score = sum(scores) / len(scores)
            self.stdout.write(f"  {platform}: {avg_score:.2f}")
        
        self.stdout.write(self.style.SUCCESS("\nFull pipeline test completed"))
    
    def show_statistics(self):
        """Show current system statistics"""
        self.stdout.write(self.style.SUCCESS("\n=== SYSTEM STATISTICS ===\n"))
        
        try:
            mongo_manager = get_mongo_manager()
            stats = mongo_manager.get_statistics()
            
            self.stdout.write("DATABASE STATISTICS:")
            for collection, info in stats.items():
                if isinstance(info, dict) and 'total_documents' in info:
                    self.stdout.write(
                        f"  {collection}: {info['total_documents']} documents, "
                        f"{info['indexes']} indexes"
                    )
            
            if 'recent_activity' in stats:
                activity = stats['recent_activity']
                self.stdout.write(f"\nRecent Activity (24h):")
                self.stdout.write(f"  Articles: {activity.get('articles_24h', 0)}")
                self.stdout.write(f"  Posts: {activity.get('posts_24h', 0)}")
            
            # Show credibility engine stats
            engine = get_credibility_engine()
            self.stdout.write(f"\nCREDIBILITY ENGINE:")
            self.stdout.write(f"  Weights: {engine.weights}")
            self.stdout.write(f"  Source History Tracked: {len(engine.source_history)} sources")
            self.stdout.write(f"  Entity Cache Size: {sum(len(v) for v in engine.entity_event_cache.values())} entries")
            
            # Show source history details
            if engine.source_history:
                self.stdout.write(f"\n  Source History Details:")
                for source, record in sorted(engine.source_history.items())[:10]:
                    self.stdout.write(
                        f"    {source}: {record.total_articles} articles, "
                        f"{record.accuracy_rate:.1%} accuracy, "
                        f"reliability: {record.reliability_score:.2f}"
                    )
        
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error getting statistics: {e}"))
    
    def generate_sample_articles(self, count: int) -> List[Dict]:
        """Generate sample articles in RAW format (matching fetcher output)"""
        import random
        from datetime import datetime, timedelta
        
        platforms = ['cryptopanic', 'cryptocompare', 'newsapi', 'messari', 'coindesk']
        
        # Source info (raw format - no pre-calculated scores)
        sources = [
            {'title': 'CoinDesk', 'domain': 'coindesk.com'},
            {'title': 'Reuters', 'domain': 'reuters.com'},
            {'title': 'Bloomberg', 'domain': 'bloomberg.com'},
            {'title': 'Messari', 'domain': 'messari.io'},
            {'title': 'CryptoSlate', 'domain': 'cryptoslate.com'},
            {'title': 'Bitcoin Magazine', 'domain': 'bitcoinmagazine.com'},
            {'title': 'NewsbtC', 'domain': 'newsbtc.com'},
            {'title': 'Unknown Blog', 'domain': 'unknownblog.com'},
            {'title': 'Crypto Pump News', 'domain': 'pumpdaily.com'},
        ]
        
        # Sample titles with different sentiment patterns
        titles_data = [
            # Neutral/Professional
            ("Bitcoin Price Analysis: Technical Indicators Show Consolidation", "Detailed analysis of BTC price action and market structure."),
            ("Ethereum Network Upgrade Successfully Implemented", "The Ethereum network has completed its scheduled upgrade."),
            ("Regulatory Framework for Digital Assets Under Review", "Regulators are examining new frameworks for cryptocurrency."),
            ("Institutional Adoption of Cryptocurrency Continues to Grow", "Major institutions increase crypto holdings."),
            ("DeFi Protocol Launches New Staking Mechanism", "New staking features announced by DeFi protocol."),
            
            # Positive/Hype
            ("BITCOIN TO THE MOON! 100X GAINS GUARANTEED!!!", "This is your chance to get rich quick with Bitcoin!"),
            ("This Altcoin Will Make You Rich - Don't Miss Out!", "Secret gem that will pump 1000x guaranteed!"),
            ("MASSIVE PUMP INCOMING - GET IN NOW!", "Buy now before it's too late!"),
            
            # Negative/Panic
            ("CRYPTO MARKET CRASH - SELL EVERYTHING NOW!", "Market collapse imminent, sell all holdings immediately!"),
            ("URGENT: Major Exchange Hacked - Funds at Risk!", "Breaking news about exchange security breach."),
            ("Bitcoin Dead? Experts Predict Total Collapse", "Analysts warn of complete market failure."),
            
            # Balanced Analysis
            ("Market Analysis: Bitcoin Shows Mixed Signals Amid Uncertainty", "Technical analysis shows both bullish and bearish indicators."),
            ("Ethereum Gas Fees Drop Following Network Optimization", "Network improvements reduce transaction costs."),
            ("Central Bank Digital Currency Pilot Program Shows Promise", "CBDC testing reveals potential benefits."),
        ]
        
        articles = []
        
        for i in range(count):
            source = random.choice(sources)
            title, description = random.choice(titles_data)
            platform = random.choice(platforms)
            
            # Generate article in RAW format (matching fetcher output)
            article = {
                # Core fields (raw from API)
                'id': f'test_article_{i}_{random.randint(1000,9999)}',
                'title': title,
                'description': description,
                'content': f"{description} More detailed content about {title[:30]}...",
                'url': f'https://example.com/article_{i}',
                'author': f'Author {i}',
                
                # Raw source info (like CryptoPanic returns)
                'source': source,
                'platform': platform,
                
                # Raw timestamp (various formats like real APIs)
                'published_at': (timezone.now() - timedelta(hours=random.randint(1, 48))).isoformat(),
                
                # CryptoPanic raw fields
                'votes': {
                    'positive': random.randint(0, 50),
                    'negative': random.randint(0, 20),
                    'important': random.randint(0, 10),
                } if platform == 'cryptopanic' else {},
                'instruments': [
                    {'code': random.choice(['BTC', 'ETH', 'SOL', 'ADA'])}
                ] if random.random() > 0.3 else [],
                'kind': random.choice(['news', 'media']) if platform == 'cryptopanic' else '',
                
                # CryptoCompare raw fields
                'upvotes': random.randint(0, 100) if platform == 'cryptocompare' else 0,
                'downvotes': random.randint(0, 30) if platform == 'cryptocompare' else 0,
                'tags': 'BTC|ETH|ALTCOIN' if platform == 'cryptocompare' else '',
                'categories': 'Trading|Analysis' if platform == 'cryptocompare' else '',
                
                # Messari raw fields
                'references': [
                    {'url': 'https://source1.com'},
                    {'url': 'https://source2.com'}
                ] if platform == 'messari' else [],
            }
            
            articles.append(article)
        
        return articles