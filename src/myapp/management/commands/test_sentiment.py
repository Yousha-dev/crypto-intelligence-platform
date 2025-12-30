import warnings
warnings.filterwarnings("ignore", message=".*module compiled against ABI version.*")

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Test the sentiment analysis model'
    
    def add_arguments(self, parser):
        parser.add_argument('--text', type=str, help='Text to analyze')
        parser.add_argument('--batch', action='store_true', help='Run batch test')
        parser.add_argument('--debug', action='store_true', help='Show debug info')
        parser.add_argument('--reset', action='store_true', help='Reset singleton')
        parser.add_argument('--no-cache', action='store_true', help='Skip cache')
    
    def handle(self, *args, **options):
        if options.get('reset'):
            from myapp.services.content.sentiment_analyzer import reset_sentiment_analyzer
            reset_sentiment_analyzer()
            self.stdout.write(self.style.WARNING("Singleton reset"))
        
        if options['debug']:
            self.show_debug_info()
            return
        
        from myapp.services.content.sentiment_analyzer import get_sentiment_analyzer
        
        self.stdout.write("Getting sentiment analyzer...")
        analyzer = get_sentiment_analyzer()
        
        self.stdout.write(f"Models loaded: {analyzer.models_loaded}")
        self.stdout.write(f"Models initialized: {analyzer._models_initialized}")
        
        skip_cache = options.get('no_cache', False)
        
        if options['text']:
            self.analyze_single(analyzer, options['text'], skip_cache)
        else:
            self.run_batch_test(analyzer, skip_cache)
    
    def show_debug_info(self):
        """Show debug information"""
        self.stdout.write(self.style.SUCCESS("\n=== DEPENDENCY DEBUG INFO ===\n"))
        
        try:
            import numpy as np
            self.stdout.write(f"NumPy version: {np.__version__}")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"NumPy error: {e}"))
        
        try:
            import torch
            self.stdout.write(f"PyTorch version: {torch.__version__}")
            self.stdout.write(f"CUDA available: {torch.cuda.is_available()}")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"PyTorch error: {e}"))
        
        try:
            import transformers
            self.stdout.write(f"Transformers version: {transformers.__version__}")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Transformers error: {e}"))
        
        try:
            from textblob import TextBlob
            self.stdout.write("TextBlob: OK")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"TextBlob error: {e}"))
        
        self.stdout.write(self.style.SUCCESS("\n=== FINBERT LOADING TEST ===\n"))
        try:
            from transformers import pipeline
            
            self.stdout.write("Creating pipeline...")
            pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            self.stdout.write(self.style.SUCCESS("Pipeline created"))
            
            self.stdout.write("Testing inference...")
            result = pipe("Bitcoin price is increasing rapidly")
            self.stdout.write(self.style.SUCCESS(f"Result: {result}"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"FinBERT failed: {e}"))
            import traceback
            self.stdout.write(traceback.format_exc())
    
    def _get_label_value(self, label) -> str:
        from myapp.services.content.sentiment_analyzer import SentimentLabel
        if isinstance(label, SentimentLabel):
            return label.value
        return str(label)
     
    def analyze_single(self, analyzer, text, skip_cache=False):
        """Analyze a single text"""
        self.stdout.write(f"\nAnalyzing: {text[:100]}...")
        
        result = analyzer.analyze(text, skip_cache=skip_cache)
        
        label_value = self._get_label_value(result.label)
        
        self.stdout.write(self.style.SUCCESS(f"\n=== SENTIMENT ANALYSIS RESULT ==="))
        self.stdout.write(f"Label: {label_value}")
        self.stdout.write(f"Score: {result.score:.3f}")
        self.stdout.write(f"Confidence: {result.confidence:.2%}")
        self.stdout.write(f"\nProbabilities:")
        self.stdout.write(f"  Bullish: {result.bullish_probability:.2%}")
        self.stdout.write(f"  Bearish: {result.bearish_probability:.2%}")
        self.stdout.write(f"  Neutral: {result.neutral_probability:.2%}")
        
        if result.aspect_sentiments:
            self.stdout.write(f"\nAspect Sentiments:")
            for crypto, score in result.aspect_sentiments.items():
                self.stdout.write(f"  {crypto}: {score:.3f}")
        
        if result.emotions:
            self.stdout.write(f"\nEmotions:")
            for emotion, score in result.emotions.items():
                if score > 0:
                    self.stdout.write(f"  {emotion}: {score:.2%}")
        
        self.stdout.write(f"\nMarket Impact: {result.predicted_market_impact} ({result.impact_confidence:.2%})")
        
        if result.flags:
            self.stdout.write(f"\nFlags: {', '.join(result.flags)}")
        
        self.stdout.write(f"\nModel Scores:")
        for model, score in result.model_scores.items():
            # For FinBERT, check if it actually ran
            if model == 'finbert':
                raw_result = getattr(analyzer, '_last_finbert_raw', None)
                if raw_result and 'label' in raw_result:
                    status = "✓" if analyzer.models_loaded else "✗"
                    raw_label = raw_result['label']
                    raw_score = raw_result['score']
                    self.stdout.write(f"  {status} {model}: {score:.3f} (raw: {raw_label} @ {raw_score:.2%})")
                else:
                    self.stdout.write(f"  {model}: {score:.3f} (not loaded)")
            else:
                status = "✓" if score != 0 else "○"
                self.stdout.write(f"  {status} {model}: {score:.3f}")
        
        # Show explanation if FinBERT returned neutral
        raw_result = getattr(analyzer, '_last_finbert_raw', None)
        if raw_result and raw_result.get('label') == 'neutral':
            self.stdout.write(self.style.WARNING(
                f"\nℹ️  FinBERT classified this as 'neutral' ({raw_result['score']:.0%} confidence)."
                f"\n    This is expected - FinBERT is trained on formal financial news, not crypto slang."
                f"\n    The crypto_lexicon model compensates for this."
            ))
    
    def run_batch_test(self, analyzer, skip_cache=False):
        """Run batch test with news and social content"""
        
        # NEWS texts (formal financial language - FinBERT excels here)
        news_texts = [
            ("Stock prices surged following positive earnings report.", "formal_bullish", "news"),
            ("The company reported strong quarterly revenue growth.", "formal_bullish", "news"),
            ("Market crash imminent as recession fears grow.", "bearish", "news"),
            ("Company files for bankruptcy amid fraud investigation.", "bearish", "news"),
            ("Bitcoin price remains stable around $50,000.", "neutral", "news"),
            ("Regulatory framework expected to bring clarity to markets.", "neutral", "news"),
        ]
        
        # SOCIAL texts (crypto slang - crypto_lexicon compensates)
        social_texts = [
            ("🚀🚀🚀 BTC TO THE MOON! HODL! 💎🙌 WAGMI!", "crypto_bullish", "social"),
            ("Just bought the dip! This is the way! LFG! 📈", "crypto_bullish", "social"),
            ("Diamond hands only! We're all gonna make it! 🦍", "crypto_bullish", "social"),
            ("Market is dead. Lost everything. Crypto is a scam. 😭", "bearish", "social"),
            ("Rug pulled again. Never trusting these devs. 💀", "bearish", "social"),
            ("Just watching charts. Anyone else bored? #crypto", "neutral", "social"),
        ]
        
        test_texts = news_texts + social_texts
        
        self.stdout.write(self.style.SUCCESS("\n=== BATCH TEST (News + Social) ===\n"))
        self.stdout.write(f"Testing {len(news_texts)} news texts and {len(social_texts)} social texts\n")
        
        results = {'news': {'correct': 0, 'total': 0}, 'social': {'correct': 0, 'total': 0}}
        
        for text, expected_type, content_type in test_texts:
            result = analyzer.analyze(text, skip_cache=skip_cache)
            label_value = self._get_label_value(result.label)
            
            raw_result = getattr(analyzer, '_last_finbert_raw', {})
            finbert_label = raw_result.get('label', 'N/A')
            finbert_conf = raw_result.get('score', 0)
            
            # Determine if prediction is correct
            is_correct = False
            if expected_type in ['formal_bullish', 'crypto_bullish'] and result.score > 0:
                is_correct = True
            elif expected_type == 'bearish' and result.score < 0:
                is_correct = True
            elif expected_type == 'neutral' and -0.2 < result.score < 0.2:
                is_correct = True
            
            results[content_type]['total'] += 1
            if is_correct:
                results[content_type]['correct'] += 1
            
            # Color based on sentiment
            if label_value in ['very_bullish', 'bullish']:
                style = self.style.SUCCESS
            elif label_value in ['very_bearish', 'bearish']:
                style = self.style.ERROR
            else:
                style = self.style.WARNING
            
            icon = "✓" if is_correct else "✗"
            self.stdout.write(f"[{content_type.upper()}] {icon} {text[:45]}...")
            self.stdout.write(style(f"   → {label_value.upper()} (score: {result.score:.2f})"))
            self.stdout.write(f"   FinBERT: {finbert_label} ({finbert_conf:.0%}) | Lexicon: {result.model_scores.get('crypto_lexicon', 0):.2f}")
            self.stdout.write("")
        
        # Summary
        self.stdout.write(self.style.SUCCESS("\n=== ACCURACY SUMMARY ==="))
        for content_type, data in results.items():
            accuracy = data['correct'] / data['total'] * 100 if data['total'] > 0 else 0
            self.stdout.write(f"  {content_type.upper()}: {data['correct']}/{data['total']} ({accuracy:.0f}%)")
         
        total_correct = sum(d['correct'] for d in results.values())
        total_tests = sum(d['total'] for d in results.values())
        overall_accuracy = total_correct / total_tests * 100 if total_tests > 0 else 0
        self.stdout.write(f"  OVERALL: {total_correct}/{total_tests} ({overall_accuracy:.0f}%)")
         
        self.stdout.write(self.style.SUCCESS("\nBatch test completed!"))
        self.stdout.write("\nNote: FinBERT works best on formal financial language.")
        self.stdout.write("Crypto slang is handled by the crypto_lexicon model.")