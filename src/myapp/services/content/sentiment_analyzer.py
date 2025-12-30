"""
Advanced Sentiment Analysis for Cryptocurrency Content
Implements ensemble approach: FinBERT + Crypto Lexicon + Rule-based
"""

import warnings
warnings.filterwarnings("ignore", message=".*module compiled against ABI version.*")

import os
import re
import logging
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib

# Suppress TensorFlow warnings BEFORE importing torch/transformers
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from textblob import TextBlob
from django.core.cache import cache
from django.conf import settings
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

logger = logging.getLogger(__name__)


class SentimentLabel(Enum):
    """Sentiment classification labels"""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class SentimentResult:
    """Comprehensive sentiment analysis result"""
    label: SentimentLabel
    score: float
    confidence: float
    bullish_probability: float
    bearish_probability: float
    neutral_probability: float
    predicted_market_impact: str
    impact_confidence: float
    aspect_sentiments: Dict[str, float] = field(default_factory=dict)
    emotions: Dict[str, float] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    model_scores: Dict[str, float] = field(default_factory=dict)
  
     
class CryptoSentimentAnalyzer:
    """
    Advanced sentiment analyzer for cryptocurrency content
    Uses ensemble of models for robust predictions
    """
    
    def __init__(self, use_gpu: bool = True):
        self._use_gpu = use_gpu
        self._models_initialized = False
        self._init_lock = threading.Lock()
        
        self.device = None
        self.finbert_tokenizer = None
        self.finbert_model = None
        self.finbert_pipeline = None
        self.models_loaded = False
        self._last_finbert_raw = None
        
        self.ensemble_weights = {
            'finbert': 0.35,
            'crypto_lexicon': 0.35,
            'rule_based': 0.20,
            'textblob': 0.10
        }
        
        self._init_crypto_lexicon()
        self.cache_timeout = 3600
        
        logger.info("CryptoSentimentAnalyzer created (models will load on first use)")
    
    def _ensure_models_loaded(self):
        """Lazy load transformer models on first use - thread safe"""
        if self._models_initialized:
            return
        
        with self._init_lock:
            if self._models_initialized:
                return
            
            try:
                import torch
                self.device = "cuda" if self._use_gpu and torch.cuda.is_available() else "cpu"
                logger.info(f"Initializing sentiment models on device: {self.device}")
                
                self._init_models()
                
            except Exception as e:
                logger.error(f"Failed to initialize models: {e}")
                self.models_loaded = False
            finally:
                self._models_initialized = True
    
    def _init_crypto_lexicon(self):
        """Initialize cryptocurrency-specific sentiment lexicon - ENHANCED"""
        self.bullish_lexicon = {
            # Extreme bullish (0.85-1.0)
            'moon': 0.9, 'mooning': 0.95, 'to the moon': 0.95, 'moonshot': 0.9,
            'rocket': 0.85, '🚀': 0.9, 'lambo': 0.8, 'lambos': 0.8, 'when lambo': 0.85,
            'parabolic': 0.95, 'going parabolic': 0.95, 'exponential': 0.85,
            'explode': 0.85, 'exploding': 0.9, 'skyrocket': 0.9, 'skyrocketing': 0.95,
            'ath': 0.8, 'all time high': 0.8, 'new ath': 0.85, 'breaking ath': 0.9,
            
            # Strong bullish (0.7-0.85)
            'bullish': 0.85, 'bull run': 0.9, 'bull market': 0.85, 'super bullish': 0.9,
            'mega bullish': 0.9, 'ultra bullish': 0.9, 'extremely bullish': 0.9,
            'breakout': 0.75, 'breaking out': 0.8, 'massive breakout': 0.85,
            'hodl': 0.7, 'hold': 0.5, 'accumulate': 0.7, 'accumulating': 0.75,
            'buy the dip': 0.75, 'btd': 0.75, 'buying the dip': 0.75,
            'undervalued': 0.7, 'gem': 0.75, 'hidden gem': 0.8, 'absolute gem': 0.85,
            'diamond hands': 0.75, '💎': 0.7, '💎🙌': 0.75, 'diamond hand': 0.75,
            'pump': 0.6, 'pumping': 0.7, 'pump it': 0.7, 'mega pump': 0.8,
            '100x': 0.85, '10x': 0.8, '50x': 0.85, '1000x': 0.9, 'moonbag': 0.75,
            'generational wealth': 0.85, 'life changing': 0.8, 'retire early': 0.8,
            
            # Moderate-strong bullish (0.6-0.7)
            'gains': 0.65, 'profit': 0.6, 'winning': 0.6, 'massive gains': 0.75,
            'huge gains': 0.7, 'insane gains': 0.75, 'printing money': 0.7,
            'adoption': 0.7, 'mainstream': 0.65, 'mass adoption': 0.75,
            'institutional': 0.7, 'whale buying': 0.75, 'whales accumulating': 0.8,
            'smart money': 0.7, 'institutions buying': 0.75,
            'incredible': 0.7, 'amazing': 0.65, 'awesome': 0.6, 'fantastic': 0.65,
            'outstanding': 0.7, 'excellent': 0.65, 'perfect': 0.65,
            'buy signal': 0.7, 'strong buy': 0.75, 'buy now': 0.65,
            'golden cross': 0.75, 'reversal': 0.6, 'bottom in': 0.7,
            'early': 0.7, 'get in early': 0.75, 'still early': 0.7,
            
            # Moderate bullish (0.4-0.6)
            'support': 0.5, 'recovery': 0.55, 'bounce': 0.5, 'bouncing': 0.55,
            'uptrend': 0.6, 'higher high': 0.55, 'higher low': 0.55,
            'accumulation': 0.6, 'consolidation': 0.4, 'base building': 0.5,
            'promising': 0.55, 'potential': 0.5, 'opportunity': 0.55,
            'growth': 0.55, 'expansion': 0.5, 'scaling': 0.55,
            'surged': 0.7, 'surge': 0.65, 'soared': 0.7, 'rallied': 0.6,
            'rally': 0.6, 'rising': 0.5, 'climbing': 0.5, 'green': 0.5,
            'bullish divergence': 0.6, 'oversold': 0.55, 'undervalued': 0.6,
            'strong fundamentals': 0.65, 'solid project': 0.6, 'legit': 0.55,
            'partnership': 0.6, 'collaboration': 0.55, 'integration': 0.6,
            'upgrade': 0.6, 'improvement': 0.5, 'innovation': 0.6,
            'staking rewards': 0.5, 'yield': 0.5, 'apy': 0.45,
            'defi summer': 0.7, 'alt season': 0.75, 'altseason': 0.75,
            'fomo': 0.4, 'dont miss': 0.6, 'last chance': 0.55,
            'cheap': 0.5, 'discounted': 0.55, 'sale': 0.5, 'bargain': 0.6,
        }
        
        self.bearish_lexicon = {
            # Extreme bearish (-0.85 to -1.0)
            'crash': -0.9, 'crashing': -0.95, 'crashed': -0.9, 'mega crash': -0.95,
            'collapsed': -0.9, 'collapse': -0.85, 'collapsing': -0.9,
            'dump': -0.8, 'dumping': -0.85, 'dumped': -0.8, 'mass dump': -0.9,
            'rug pull': -0.95, 'rug': -0.85, 'rugged': -0.95, 'rugpull': -0.95,
            'scam': -0.95, 'fraud': -0.95, 'ponzi': -0.95, 'ponzi scheme': -0.95,
            'exit scam': -0.95, 'honeypot': -0.9, 'fake': -0.85,
            'dead': -0.85, 'rip': -0.8, 'rekt': -0.85, 'wrecked': -0.85,
            'liquidation': -0.8, 'liquidated': -0.85, 'liq': -0.8, 'mass liquidation': -0.9,
            'bankruptcy': -0.95, 'insolvent': -0.9, 'bankrupt': -0.95,
            'going to zero': -0.95, 'zero': -0.7, 'worthless': -0.9,
            'plunge': -0.85, 'plunging': -0.9, 'plummeting': -0.9, 'tanking': -0.85,
            
            # Strong bearish (-0.7 to -0.85)
            'bearish': -0.8, 'bear market': -0.85, 'bear trap': -0.7,
            'super bearish': -0.85, 'extremely bearish': -0.85,
            'hack': -0.9, 'hacked': -0.9, 'exploit': -0.85, 'exploited': -0.9,
            'vulnerability': -0.75, 'security breach': -0.85, 'breach': -0.8,
            'sell off': -0.75, 'selloff': -0.75, 'panic sell': -0.85, 'panic selling': -0.85,
            'capitulation': -0.8, 'capitulating': -0.85, 'giving up': -0.75,
            'bloodbath': -0.85, 'blood': -0.7, 'bleeding': -0.7, 'blood red': -0.8,
            'death cross': -0.8, 'dead cat bounce': -0.75, 'bull trap': -0.75,
            'free fall': -0.85, 'freefall': -0.85, 'falling knife': -0.8,
            'paper hands': -0.7, '📉': -0.7, 'ngmi': -0.75, 'rip bozo': -0.75,
            
            # Moderate-strong bearish (-0.5 to -0.7)
            'fud': -0.6, 'fear': -0.5, 'uncertainty': -0.4, 'doubt': -0.5,
            'panic': -0.7, 'panicking': -0.75, 'scared': -0.6, 'terrified': -0.7,
            'regulation': -0.4, 'ban': -0.7, 'banned': -0.75, 'illegal': -0.8,
            'sec lawsuit': -0.75, 'lawsuit': -0.65, 'investigation': -0.6,
            'delisted': -0.8, 'delisting': -0.75, 'suspended': -0.7,
            'whale dumping': -0.75, 'whales selling': -0.7, 'insider selling': -0.7,
            'overvalued': -0.5, 'bubble': -0.6, 'bubble pop': -0.75, 'popping': -0.65,
            'recession': -0.7, 'inflation': -0.5, 'interest rates': -0.4,
            'contagion': -0.75, 'systemic risk': -0.7, 'cascade': -0.65,
            
            # Moderate bearish (-0.3 to -0.5)
            'resistance': -0.4, 'rejection': -0.5, 'rejected': -0.55,
            'downtrend': -0.6, 'lower low': -0.55, 'lower high': -0.55,
            'correction': -0.5, 'pullback': -0.45, 'retracement': -0.45,
            'selling pressure': -0.55, 'heavy selling': -0.6, 'distribution': -0.5,
            'warning': -0.5, 'risk': -0.4, 'caution': -0.4, 'careful': -0.35,
            'overbought': -0.4, 'extended': -0.4, 'toppy': -0.45,
            'bearish divergence': -0.55, 'weakness': -0.5, 'losing steam': -0.5,
            'bag holder': -0.6, 'bagholder': -0.6, 'holding bags': -0.6,
            'cope': -0.5, 'copium': -0.5, 'hopium': -0.45,
            'red flag': -0.6, 'concerning': -0.5, 'suspicious': -0.6,
            'slow': -0.3, 'stagnant': -0.4, 'dying': -0.7, 'dead coin': -0.8,
        }
        
        self.neutral_lexicon = {
            'hold': 0.0, 'holding': 0.0, 'neutral': 0.0, 'sideways': 0.0,
            'consolidation': 0.0, 'consolidating': 0.0, 'waiting': 0.0,
            'monitor': 0.0, 'observe': 0.0, 'watching': 0.0, 'observing': 0.0,
            'ranging': 0.0, 'range bound': 0.0, 'unclear': 0.0, 
            'unknown': 0.0, 'maybe': 0.0, 'possibly': 0.0, 'might': 0.0,
            'watch': 0.0, 'flat': 0.0, 'no change': 0.0, 'stable': 0.0,
            'steady': 0.0, 'calm': 0.0, 'quiet': 0.0, 'balanced': 0.0,
            'indecisive': 0.0, 'mixed signals': 0.0, 'choppy': 0.0,
        }
        
        self.crypto_mentions = {
            'bitcoin': ['bitcoin', 'btc', 'sats', 'satoshi', 'satoshis'],
            'ethereum': ['ethereum', 'eth', 'ether', 'vitalik'],
            'solana': ['solana', 'sol', 'phantom'],
            'cardano': ['cardano', 'ada', 'hoskinson'],
            'xrp': ['xrp', 'ripple'],
            'dogecoin': ['doge', 'dogecoin', 'shiba'],
            'bnb': ['bnb', 'binance coin', 'binance'],
            'polygon': ['polygon', 'matic'],
            'avalanche': ['avalanche', 'avax'],
            'chainlink': ['chainlink', 'link'],
            'polkadot': ['polkadot', 'dot'],
            'uniswap': ['uniswap', 'uni'],
            'litecoin': ['litecoin', 'ltc'],
            'cosmos': ['cosmos', 'atom'],
            'algorand': ['algorand', 'algo'],
            'tron': ['tron', 'trx'],
            'stellar': ['stellar', 'xlm'],
            'monero': ['monero', 'xmr'],
            'tether': ['tether', 'usdt'],
            'usdc': ['usdc', 'circle'],
        }
        
        self.emotion_lexicon = {
            'fear': [
                # Basic fear emotions
                'fear', 'scared', 'worried', 'worry', 'panic', 'panicking', 'anxious', 
                'anxiety', 'terrified', 'nervous', 'afraid', 'frightened',
                # Crypto-specific fear
                'crash', 'crashing', 'crashed', 'dump', 'dumping', 'dumped',
                'liquidat', 'rekt', 'wrecked', 'bear market', 'correction',
                'sell off', 'selloff', 'blood', 'bloodbath', 'bleeding',
                'capitulation', 'dead cat', 'death cross', 'free fall',
                'panic sell', 'paper hands', 'baghold', 'trapped',
                'rug pull', 'rugged', 'exit scam', 'hack', 'exploit',
                'going to zero', 'worthless', 'dead', 'over',
            ],
            'greed': [
                'greed', 'greedy', 'fomo', 'cant miss', "can't miss", "don't miss", 
                'guaranteed', 'easy money', 'quick profit', 'get rich',
                'moon', 'mooning', 'lambo', 'lambos', 'when lambo',
                'to the moon', '🚀', 'rocket', 'parabolic',
                'ath', 'all time high', 'new ath', 'breaking ath',
                '100x', '10x', '50x', '1000x', 'moonshot',
                'generational wealth', 'life changing', 'early', 'get in early',
                'gem', 'hidden gem', 'absolute gem', 'diamond hands',
                'buy now', 'last chance', 'dont miss out', 'all in',
                'massive gains', 'insane gains', 'printing money',
                'inflow', 'inflows', 'buying', 'accumulation',
                'record purchase', 'record buy', 'big buy',
                'institutional buying', 'whale buy',
            ],
            'uncertainty': [
                # Decision uncertainty
                'uncertain', 'uncertainty', 'unclear', 'unknown', 'maybe', 
                'possibly', 'might', 'could', 'unsure', 'confused',
                'doubt', 'doubtful', 'questioning', 'skeptical',
                # Market uncertainty
                'consolidat', 'sideways', 'ranging', 'choppy',
                'waiting', 'watch', 'watching', 'monitor', 'observ',
                'mixed signals', 'indecisive', 'what to do',
                'hold or sell', 'buy or sell', 'confused',
            ],
            'excitement': [
                'excited', 'exciting', 'amazing', 'incredible', 'awesome', 
                'fantastic', 'wonderful', 'huge', 'massive', 'insane',
                'unbelievable', 'wow', 'omg', 'holy',
                'record', 'record-breaking', 'historic', 'milestone',
                'breakthrough', 'unprecedented', 'extraordinary',
                'billion', 'billions', 'trillion',
                'surge', 'surging', 'soaring', 'exploding',
                'hit new', 'reaches', 'breaks', 'crosses',
                'all-time', 'ath', 'new high',
                'breakout', 'breaking out', 'pump', 'pumping', 'surge', 'surging',
                'rocket', 'rocketing', 'explod', 'skyrocket', 'parabolic',
                'mooning', 'moon', 'gains', 'profit', 'winning',
                'lfg', 'lets go', "let's go", 'gm', 'wagmi',
            ],
            'confidence': [
                # Basic confidence
                'confident', 'confidence', 'certain', 'certainly', 'sure', 'surely',
                'definitely', 'clearly', 'obviously', 'absolutely',
                'no doubt', 'guaranteed', 'conviction', 'believe',
                # Trading confidence
                'accumulation', 'accumulating', 'strong support', 'floor',
                'bullish', 'buy signal', 'buying', 'hodl', 'holding',
                'confirmed', 'confirmation', 'trend confirmed',
                'smart money', 'institutional', 'whale', 'diamond hands',
                'confirmed', 'verification', 'proof',
                'data shows', 'numbers show', 'report confirms',
                'official', 'announced', 'released',
            ],
            'euphoria': [
                # Extreme positive emotions
                'euphoria', 'euphoric', 'ecstatic', 'amazing', 'incredible',
                'best ever', 'unstoppable', 'invincible',
                # Crypto euphoria
                'parabolic', 'going parabolic', 'to the moon',
                'new paradigm', 'this time different', 'cant stop',
                'price discovery', 'blow off top', 'face melting',
                'generational', 'retire early', 'financial freedom',
            ],
            'despair': [
                # Extreme negative emotions
                'despair', 'hopeless', 'helpless', 'defeated', 'crushed',
                'devastated', 'destroyed', 'ruined',
                # Crypto despair
                'its over', 'game over', 'dead', 'rip', 'rekt',
                'wiped out', 'lost everything', 'going to zero',
                'capitulation', 'giving up', 'sell everything',
                'never coming back', 'done with crypto',
            ]
        }
        
        self.high_impact_terms = [
            # Regulatory
            'sec', 'regulation', 'regulatory', 'cftc', 'fda', 'government',
            'etf', 'etf approval', 'etf rejection', 'approval', 'rejected',
            'ban', 'banned', 'illegal', 'lawsuit', 'investigation',
            # Security
            'hack', 'hacked', 'exploit', 'exploited', 'vulnerability',
            'security breach', 'breach', 'stolen', 'compromised',
            # Corporate
            'bankruptcy', 'insolvent', 'bankrupt', 'liquidation',
            'acquisition', 'merger', 'partnership', 'collaboration',
            'institutional', 'institutions', 'institutional adoption',
            # Technical
            'halving', 'bitcoin halving', 'upgrade', 'hard fork', 'fork',
            'merge', 'ethereum merge', 'mainnet', 'launch', 'deployment',
            # Market structure
            'delisted', 'delisting', 'listing', 'exchange listing',
            'futures', 'options', 'derivatives', 'spot etf',
            # Macro
            'fed', 'federal reserve', 'interest rates', 'inflation',
            'recession', 'economic crisis', 'bank failure',
        ]
    
    def _init_models(self):
        """Initialize transformer models - simplified"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            
            model_cache = getattr(settings, 'MODEL_CACHE_DIR', None)
            
            # Load tokenizer
            self.finbert_tokenizer = self._load_tokenizer(model_cache)
            
            # Load model
            self.finbert_model = self._load_model(model_cache)
            
            # Create pipeline
            self.finbert_pipeline = self._create_pipeline()
            
            # Verify
            self._verify_models()
            
            self.models_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self.models_loaded = False
            raise

    def _load_tokenizer(self, cache_dir):
        """Load tokenizer - single responsibility"""
        logger.info("Loading FinBERT tokenizer...")
        return AutoTokenizer.from_pretrained(
            "ProsusAI/finbert",
            cache_dir=cache_dir
        )

    def _load_model(self, cache_dir):
        """Load and configure model - single responsibility"""
        logger.info("Loading FinBERT model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert",
            cache_dir=cache_dir
        )
        model = model.to(self.device)
        model.eval()
        return model

    def _create_pipeline(self):
        """Create inference pipeline - single responsibility"""
        logger.info("Creating FinBERT pipeline...")
        device_id = 0 if self.device == "cuda" else -1
        return pipeline(
            "sentiment-analysis",
            model=self.finbert_model,
            tokenizer=self.finbert_tokenizer,
            device=device_id,
            max_length=512,
            truncation=True
        )

    def _verify_models(self):
        """Verify models work - single responsibility"""
        test_result = self.finbert_pipeline("Stock price increased")
        logger.info(f"FinBERT verified successfully. Test: {test_result}")

    
    def analyze(self, text: str, metadata: Optional[Dict] = None, skip_cache: bool = False) -> SentimentResult:
        """Perform comprehensive sentiment analysis"""
        self._ensure_models_loaded()
        
        # Check cache
        cache_key = self._get_cache_key(text)
        if not skip_cache:
            cached_result = cache.get(cache_key)
            if cached_result:
                return self._dict_to_result(cached_result)
        else:
            cache.delete(cache_key)
        
        original_text = text
        cleaned_text = self._preprocess_text(text)
        
        if not cleaned_text or len(cleaned_text.strip()) < 5:
            return self._create_neutral_result()
        
        model_scores = {}
        
        # 1. FinBERT
        if self.models_loaded and self.finbert_pipeline:
            model_scores['finbert'] = self._get_finbert_score(cleaned_text)
        else:
            model_scores['finbert'] = 0.0
        
        # 2. Crypto Lexicon
        model_scores['crypto_lexicon'] = self._get_lexicon_score(cleaned_text)
        
        # 3. Rule-based
        model_scores['rule_based'] = self._get_rule_based_score(original_text, cleaned_text)
        
        # 4. TextBlob
        model_scores['textblob'] = self._get_textblob_score(cleaned_text)
        
        ensemble_score = self._calculate_ensemble_score(model_scores)
        probabilities = self._calculate_probabilities(ensemble_score, model_scores)
        label = self._score_to_label(ensemble_score)
        confidence = self._calculate_confidence(model_scores, probabilities)
        aspect_sentiments = self._get_aspect_sentiments(cleaned_text)
        emotions = self._analyze_emotions(cleaned_text, ensemble_score)
        impact, impact_conf = self._predict_market_impact(cleaned_text, ensemble_score, metadata)
        flags = self._identify_flags(cleaned_text, ensemble_score, emotions)
        
        result = SentimentResult(
            label=label,
            score=ensemble_score,
            confidence=confidence,
            bullish_probability=probabilities['bullish'],
            bearish_probability=probabilities['bearish'],
            neutral_probability=probabilities['neutral'],
            predicted_market_impact=impact,
            impact_confidence=impact_conf,
            aspect_sentiments=aspect_sentiments,
            emotions=emotions,
            flags=flags,
            model_scores=model_scores
        )
        
        cache.set(cache_key, self._result_to_dict(result), self.cache_timeout)
        
        return result
    
    def analyze_batch(self, texts: List[str], metadata_list: Optional[List[Dict]] = None) -> List[SentimentResult]:
        """Analyze multiple texts efficiently"""
        self._ensure_models_loaded()
        results = []
        metadata_list = metadata_list or [None] * len(texts)
        for text, metadata in zip(texts, metadata_list):
            result = self.analyze(text, metadata)
            results.append(result)
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        if not text:
            return ""
        text_lower = text.lower()
        text_lower = re.sub(r'http\S+|www\.\S+', '', text_lower)
        text_lower = re.sub(r'@\w+', '', text_lower)
        text_lower = re.sub(r'\$([A-Za-z]+)', r'\1', text_lower)
        text_lower = re.sub(r'([!?.]){3,}', r'\1\1', text_lower)
        text_lower = re.sub(r'\s+', ' ', text_lower).strip()
        return text_lower
    
    def _get_finbert_score(self, text: str) -> float:
        """Get sentiment score from FinBERT"""
        try:
            if not self.models_loaded or not self.finbert_pipeline:
                return 0.0
            
            if len(text) > 500:
                text = text[:500]
            
            result = self.finbert_pipeline(text)[0]
            label = result['label'].lower()
            score = result['score']
            
            # Store raw result for debugging
            self._last_finbert_raw = {'label': label, 'score': score}
            
            if label == 'positive':
                return score
            elif label == 'negative':
                return -score
            else:  # neutral
                return 0.0
                
        except Exception as e:
            logger.warning(f"FinBERT error: {e}")
            self._last_finbert_raw = {'error': str(e)}
            return 0.0
    
    def _get_lexicon_score(self, text: str) -> float:
        """Calculate sentiment score using crypto lexicon"""
        words = text.split()
        text_lower = text.lower()
        
        bullish_score = 0.0
        bearish_score = 0.0
        matches = 0
        
        for word in words:
            word_clean = word.strip('.,!?')
            if word_clean in self.bullish_lexicon:
                bullish_score += self.bullish_lexicon[word_clean]
                matches += 1
            if word_clean in self.bearish_lexicon:
                bearish_score += abs(self.bearish_lexicon[word_clean])
                matches += 1
        
        if '🚀' in text:
            bullish_score += self.bullish_lexicon.get('🚀', 0.9)
            matches += 1
        
        for phrase, score in self.bullish_lexicon.items():
            if ' ' in phrase and phrase in text_lower:
                bullish_score += score
                matches += 1
        
        for phrase, score in self.bearish_lexicon.items():
            if ' ' in phrase and phrase in text_lower:
                bearish_score += abs(score)
                matches += 1
        
        if matches == 0:
            return 0.0
        
        net_score = (bullish_score - bearish_score) / max(matches, 1)
        return max(-1.0, min(1.0, net_score))
    
    def _get_rule_based_score(self, original_text: str, cleaned_text: str) -> float:
        """Apply rule-based sentiment analysis"""
        score = 0.0
        text_lower = cleaned_text.lower()
        
        strong_bullish = ['100x', '1000x', 'guaranteed gains', 'easy profit']
        strong_bearish = ['exit scam', 'rug pull', 'total loss', 'going to zero']
        
        for indicator in strong_bullish:
            if indicator in text_lower:
                score += 0.5
        
        for indicator in strong_bearish:
            if indicator in text_lower:
                score -= 0.5
        
        if '!' in original_text:
            positive_exclaim = ['incredible', 'amazing', 'awesome', 'great', 'moon', 'mooning']
            for word in positive_exclaim:
                if word in text_lower:
                    score += 0.3
                    break
            
            negative_exclaim = ['crash', 'dump', 'scam', 'dead', 'over']
            for word in negative_exclaim:
                if word in text_lower:
                    score -= 0.3
                    break
        
        if '🚀' in original_text:
            score += 0.2
        
        alpha_chars = [c for c in original_text if c.isalpha()]
        if alpha_chars:
            caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if caps_ratio > 0.5 and score != 0:
                score *= 1.2
        
        exclamation_count = original_text.count('!')
        if exclamation_count >= 2 and score != 0:
            score *= 1.1
        
        negation_words = ['not', "don't", "won't", "isn't", "aren't", "no"]
        for neg in negation_words:
            if neg in text_lower:
                pattern = rf'{neg}\s+\w*\s*(bullish|bearish|good|bad|up|down)'
                if re.search(pattern, text_lower):
                    score *= -0.5
        
        return max(-1.0, min(1.0, score))
    
    def _get_textblob_score(self, text: str) -> float:
        """Get sentiment from TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception:
            return 0.0
    
    def _calculate_ensemble_score(self, model_scores: Dict[str, float]) -> float:
        """Calculate weighted ensemble score"""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for model, score in model_scores.items():
            weight = self.ensemble_weights.get(model, 0)
            # Don't skip FinBERT neutral - it's valid
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _calculate_probabilities(self, ensemble_score: float, model_scores: Dict) -> Dict[str, float]:
        """Convert ensemble score to probabilities"""
        if ensemble_score > 0.3:
            bullish = 0.5 + (ensemble_score * 0.5)
            bearish = 0.1
            neutral = 1 - bullish - bearish
        elif ensemble_score < -0.3:
            bearish = 0.5 + (abs(ensemble_score) * 0.5)
            bullish = 0.1
            neutral = 1 - bullish - bearish
        else:
            neutral = 0.6 - abs(ensemble_score)
            bullish = 0.2 + (ensemble_score * 0.3) if ensemble_score > 0 else 0.2
            bearish = 0.2 + (abs(ensemble_score) * 0.3) if ensemble_score < 0 else 0.2
        
        bullish = max(0, bullish)
        bearish = max(0, bearish)
        neutral = max(0, neutral)
        
        total = bullish + bearish + neutral
        return {
            'bullish': bullish / total,
            'bearish': bearish / total,
            'neutral': neutral / total
        }
    
    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert numeric score to sentiment label"""
        if score >= 0.6:
            return SentimentLabel.VERY_BULLISH
        elif score >= 0.2:
            return SentimentLabel.BULLISH
        elif score <= -0.6:
            return SentimentLabel.VERY_BEARISH
        elif score <= -0.2:
            return SentimentLabel.BEARISH
        else:
            return SentimentLabel.NEUTRAL
    
    def _calculate_confidence(self, model_scores: Dict, probabilities: Dict) -> float:
        """Calculate confidence in the prediction"""
        # Include all scores, even 0 (neutral is valid)
        active_scores = []
        for model, score in model_scores.items():
            if model == 'finbert':
                # For FinBERT, check if it actually ran
                if self.models_loaded and self._last_finbert_raw and 'label' in self._last_finbert_raw:
                    active_scores.append(score)
            else:
                active_scores.append(score)
        
        if not active_scores:
            return 0.5
        
        mean_score = sum(active_scores) / len(active_scores)
        variance = sum((s - mean_score) ** 2 for s in active_scores) / len(active_scores)
        agreement_confidence = max(0, 1 - (variance * 2))
        max_prob = max(probabilities.values())
        prob_confidence = (max_prob - 0.33) / 0.67
        confidence = (agreement_confidence * 0.5) + (prob_confidence * 0.5)
        return max(0.1, min(1.0, confidence))
    
    def _get_aspect_sentiments(self, text: str) -> Dict[str, float]:
        """Get sentiment for each cryptocurrency mentioned"""
        aspect_sentiments = {}
        text_lower = text.lower()
        
        for crypto, aliases in self.crypto_mentions.items():
            mentioned = any(alias in text_lower for alias in aliases)
            if mentioned:
                sentences = re.split(r'[.!?]', text_lower)
                relevant_sentences = [s for s in sentences if any(alias in s for alias in aliases)]
                if relevant_sentences:
                    combined_text = ' '.join(relevant_sentences)
                    lexicon_score = self._get_lexicon_score(combined_text)
                    aspect_sentiments[crypto] = lexicon_score
        
        return aspect_sentiments
    
    def _analyze_emotions(self, text: str, sentiment_score: float) -> Dict[str, float]:
        """
        Analyze emotions with PROPER sensitivity for crypto content
        """
        emotions = {}
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)
        
        if word_count == 0:
            return {emotion: 0.0 for emotion in self.emotion_lexicon}
        
        for emotion, indicators in self.emotion_lexicon.items():
            # Count matches using multiple strategies
            matches = 0
            match_details = []  # For debugging
            
            # Strategy 1: Exact word matching
            for word in words:
                word_clean = word.strip('.,!?;:')
                for indicator in indicators:
                    if ' ' not in indicator:  # Single word indicator
                        if word_clean == indicator:
                            matches += 1
                            match_details.append(f"exact:{indicator}")
                            break
            
            # Strategy 2: Partial/stem matching (for words like "liquidat" matching "liquidation")
            for word in words:
                word_clean = word.strip('.,!?;:')
                if len(word_clean) >= 4:  # Only for words 4+ chars
                    for indicator in indicators:
                        if ' ' not in indicator and len(indicator) >= 4:
                            # Check if word starts with indicator (stem match)
                            if word_clean.startswith(indicator) or indicator.startswith(word_clean[:5]):
                                matches += 0.5  # Partial credit for stem matches
                                match_details.append(f"stem:{indicator}")
                                break
            
            # Strategy 3: Phrase matching in full text
            for indicator in indicators:
                if ' ' in indicator:  # Multi-word phrases
                    if indicator in text_lower:
                        matches += 1.5  # Phrases get more weight
                        match_details.append(f"phrase:{indicator}")
                        
            # Crypto content is brief but emotionally charged - need lower thresholds
            if word_count < 10:
                emotions[emotion] = min(1.0, matches * 2.0)
            elif word_count < 15:
                # Very short text: single match should register strongly
                emotions[emotion] = min(1.0, matches * 1.5)  # 1 match = 1.0
            elif word_count < 30:
                # Short text: be very sensitive (tweets, headlines)
                emotions[emotion] = min(1.0, matches * 1.2)
            elif word_count < 50:
                # Medium text: still quite sensitive
                emotions[emotion] = min(1.0, matches * 0.8)
            else:
                # Long text: normalize but keep sensitive
                threshold = max(word_count / 50, 2)  # Reduced from 80
                emotions[emotion] = min(1.0, matches / threshold)
            
            # Enhanced debug logging for high-confidence detections
            if emotions[emotion] > 0.3:
                logger.debug(
                    f"Emotion '{emotion}': {emotions[emotion]:.2f} | "
                    f"Matches: {matches:.1f} | Words: {word_count} | "
                    f"Details: {match_details[:3]}"
                )
        
        total_emotion = sum(emotions.values())
        if total_emotion < 0.1:  # No emotions detected
            # Infer from sentiment score
            if abs(sentiment_score) > 0.5:  # Strong sentiment
                if sentiment_score > 0:
                    # Bullish -> infer greed/excitement
                    emotions['greed'] = min(0.6, sentiment_score * 0.8)
                    emotions['excitement'] = min(0.5, sentiment_score * 0.6)
                else:
                    # Bearish -> infer fear
                    emotions['fear'] = min(0.6, abs(sentiment_score) * 0.8)
            
            # Check for analytical/neutral content
            analytical_terms = ['analysis', 'prediction', 'tutorial', 'guide', 
                            'review', 'research', 'technical']
            if any(term in text_lower for term in analytical_terms):
                emotions['confidence'] = 0.5  # Analytical content has confidence

        return emotions


    
    def _predict_market_impact(self, text: str, sentiment_score: float, 
                              metadata: Optional[Dict]) -> Tuple[str, float]:
        """Predict potential market impact"""
        text_lower = text.lower()
        impact_score = 0.0
        
        for term in self.high_impact_terms:
            if term in text_lower:
                impact_score += 0.3
        
        strong_bullish = ['moon', 'mooning', 'rocket', '🚀', 'ath', 'breakout', 'pump', 'surge', 'soar', 'incredible']
        strong_bearish = ['crash', 'dump', 'plunge', 'collapse', 'rekt', 'liquidat']
        
        for term in strong_bullish:
            if term in text_lower:
                impact_score += 0.25
        
        for term in strong_bearish:
            if term in text_lower:
                impact_score += 0.25
        
        source_multiplier = 1.0
        if metadata:
            source_score = metadata.get('source_credibility', {}).get('trust_score', 5)
            source_multiplier = source_score / 5
        
        sentiment_impact = abs(sentiment_score) * 0.5
        final_impact = min(1.0, (impact_score + sentiment_impact) * source_multiplier)
        
        if final_impact >= 0.5:
            return 'high', min(0.95, 0.6 + final_impact * 0.35)
        elif final_impact >= 0.3:
            return 'medium', 0.5 + final_impact * 0.3
        elif final_impact >= 0.15:
            return 'low', 0.4 + final_impact * 0.3
        else:
            return 'none', max(0.3, 0.5 - final_impact)
    
    def _identify_flags(self, text: str, score: float, emotions: Dict) -> List[str]:
        """Identify content flags"""
        flags = []
        text_lower = text.lower()
        
        if abs(score) > 0.8:
            flags.append('extreme_sentiment')
        if emotions.get('fear', 0) > 0.5:
            flags.append('high_fear')
        if emotions.get('greed', 0) > 0.5:
            flags.append('high_greed_fomo')
         
        pump_indicators = ['guaranteed', '100x', '1000x', 'free money', 'cant lose']
        if any(indicator in text_lower for indicator in pump_indicators):
            flags.append('potential_pump_spam')
        
        fud_indicators = ['scam confirmed', 'going to zero', 'its over', 'rip']
        if any(indicator in text_lower for indicator in fud_indicators):
            flags.append('potential_fud')
        
        return flags
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key with version"""
        # Include model version to invalidate cache on updates
        model_version = "v4"  # Update when model/logic changes
        key_data = f"{model_version}:{text}"
        text_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"sentiment:{text_hash}"
    
    def _create_neutral_result(self) -> SentimentResult:
        """Create neutral result for edge cases"""
        return SentimentResult(
            label=SentimentLabel.NEUTRAL,
            score=0.0,
            confidence=0.3,
            bullish_probability=0.33,
            bearish_probability=0.33,
            neutral_probability=0.34,
            predicted_market_impact='none',
            impact_confidence=0.5,
            aspect_sentiments={},
            emotions={},
            flags=['insufficient_content'],
            model_scores={}
        )
    
    def _result_to_dict(self, result: SentimentResult) -> Dict:
        """Convert result to dictionary for caching"""
        return {
            'label': result.label.value if isinstance(result.label, SentimentLabel) else result.label,
            'score': result.score,
            'confidence': result.confidence,
            'bullish_probability': result.bullish_probability,
            'bearish_probability': result.bearish_probability,
            'neutral_probability': result.neutral_probability,
            'predicted_market_impact': result.predicted_market_impact,
            'impact_confidence': result.impact_confidence,
            'aspect_sentiments': result.aspect_sentiments,
            'emotions': result.emotions,
            'flags': result.flags,
            'model_scores': result.model_scores
        }
    
    def _dict_to_result(self, data: Dict) -> SentimentResult:
        """Convert cached dictionary back to SentimentResult"""
        label = data['label']
        if isinstance(label, str):
            label = SentimentLabel(label)
        
        return SentimentResult(
            label=label,
            score=data['score'],
            confidence=data['confidence'],
            bullish_probability=data['bullish_probability'],
            bearish_probability=data['bearish_probability'],
            neutral_probability=data['neutral_probability'],
            predicted_market_impact=data['predicted_market_impact'],
            impact_confidence=data['impact_confidence'],
            aspect_sentiments=data.get('aspect_sentiments', {}),
            emotions=data.get('emotions', {}),
            flags=data.get('flags', []),
            model_scores=data.get('model_scores', {})
        )


# Singleton instance
_sentiment_analyzer_instance = None
_singleton_lock = threading.Lock()


def get_sentiment_analyzer() -> CryptoSentimentAnalyzer:
    """Get singleton sentiment analyzer instance (thread-safe)"""
    global _sentiment_analyzer_instance
    
    if _sentiment_analyzer_instance is None:
        with _singleton_lock:
            if _sentiment_analyzer_instance is None:
                _sentiment_analyzer_instance = CryptoSentimentAnalyzer()
    
    return _sentiment_analyzer_instance


def reset_sentiment_analyzer():
    """Reset the singleton instance"""
    global _sentiment_analyzer_instance
    with _singleton_lock:
        _sentiment_analyzer_instance = None