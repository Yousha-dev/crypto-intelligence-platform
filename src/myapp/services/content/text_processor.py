"""
Advanced Text Processing for Cryptocurrency Content
Implements preprocessing, NER, and language detection
"""

import re
import logging
import threading
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)
 
# Lazy imports for heavy libraries
_spacy_model = None
_langdetect_available = False
    
   
def _get_spacy_model():
    """Lazy load spaCy model"""
    global _spacy_model
    if _spacy_model is None:
        try:
            import spacy
            try:
                _spacy_model = spacy.load("en_core_web_lg")
            except OSError:
                logger.warning("en_core_web_lg not found, downloading...")
                from spacy.cli import download
                download("en_core_web_lg")
                _spacy_model = spacy.load("en_core_web_lg")
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load spaCy: {e}")
            _spacy_model = None
    return _spacy_model


@dataclass
class ProcessedText:
    """Result of text preprocessing"""
    original: str
    cleaned: str
    tokens: List[str]
    lemmas: List[str]
    language: str
    language_confidence: float
    is_english: bool
    hashtags: List[str]
    mentions: List[str]
    urls: List[str]
    word_count: int
    char_count: int


@dataclass 
class ExtractedEntities:
    """Named entities extracted from text"""
    cryptocurrencies: List[str]
    exchanges: List[str]
    persons: List[str]
    organizations: List[str]
    money_amounts: List[str]
    dates: List[str]
    locations: List[str]
    all_entities: List[Dict[str, str]]  # [{text, label, start, end}]


class CryptoTextProcessor:
    """
    Advanced text processor for cryptocurrency content
    Handles preprocessing, NER, and language detection
    """
    
    def __init__(self):
        # Stopwords (keeping negation words)
        self.stopwords = self._init_stopwords()
        
        # Crypto-specific entity patterns
        self.crypto_patterns = self._init_crypto_patterns()
        
        # Regex patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.hashtag_pattern = re.compile(r'#(\w+)')
        self.mention_pattern = re.compile(r'@(\w+)')
        self.ticker_pattern = re.compile(r'\$([A-Z]{2,5})\b')
        self.number_pattern = re.compile(
            r'(?<![$#@])\b\d+(?:\.\d{1,2})?[KkMmBb]?\b'
        )
        
        logger.info("CryptoTextProcessor initialized")
    
    def _init_stopwords(self) -> Set[str]:
        """Initialize stopwords, keeping negation words"""
        # Common English stopwords
        stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how', 'all',
            'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
            'now', 'here', 'there', 'then', 'once', 'if', 'because', 'until',
            'while', 'about', 'against', 'between', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over',
            'under', 'again', 'further', 'am', 'being', 'having', 'doing'
        }
        
        # Keep these negation words (important for sentiment)
        negation_words = {
            'not', 'no', 'nor', 'neither', 'never', 'none', 'nobody', 'nothing',
            'nowhere', 'hardly', 'scarcely', 'barely', "don't", "doesn't", "didn't",
            "won't", "wouldn't", "couldn't", "shouldn't", "isn't", "aren't", "wasn't",
            "weren't", "haven't", "hasn't", "hadn't", "can't", "cannot"
        }
        
        return stopwords - negation_words
    
    def _init_crypto_patterns(self) -> Dict[str, List[str]]:
        """Initialize crypto-specific entity patterns"""
        return {
            'CRYPTO': [
                # Major cryptocurrencies
                ('bitcoin', 'BTC'), ('btc', 'BTC'), ('satoshi', 'BTC'), ('sats', 'BTC'),
                ('ethereum', 'ETH'), ('eth', 'ETH'), ('ether', 'ETH'),
                ('solana', 'SOL'), ('sol', 'SOL'),
                ('cardano', 'ADA'), ('ada', 'ADA'),
                ('ripple', 'XRP'), ('xrp', 'XRP'),
                ('dogecoin', 'DOGE'), ('doge', 'DOGE'),
                ('polkadot', 'DOT'), ('dot', 'DOT'),
                ('avalanche', 'AVAX'), ('avax', 'AVAX'),
                ('chainlink', 'LINK'), ('link', 'LINK'),
                ('polygon', 'MATIC'), ('matic', 'MATIC'),
                ('litecoin', 'LTC'), ('ltc', 'LTC'),
                ('uniswap', 'UNI'), ('uni', 'UNI'),
                ('shiba inu', 'SHIB'), ('shib', 'SHIB'),
                ('binance coin', 'BNB'), ('bnb', 'BNB'),
                ('tether', 'USDT'), ('usdt', 'USDT'),
                ('usdc', 'USDC'), ('usd coin', 'USDC'),
                ('dai', 'DAI'),
            ],
            'EXCHANGE': [
                'binance', 'coinbase', 'kraken', 'ftx', 'kucoin', 'huobi',
                'bitfinex', 'gemini', 'bitstamp', 'okx', 'bybit', 'gate.io',
                'crypto.com', 'bittrex', 'poloniex', 'upbit', 'bithumb'
            ],
            'PERSON': [
                ('elon musk', 'Elon Musk'), ('musk', 'Elon Musk'),
                ('vitalik buterin', 'Vitalik Buterin'), ('vitalik', 'Vitalik Buterin'),
                ('satoshi nakamoto', 'Satoshi Nakamoto'),
                ('cz', 'Changpeng Zhao'), ('changpeng zhao', 'Changpeng Zhao'),
                ('brian armstrong', 'Brian Armstrong'),
                ('michael saylor', 'Michael Saylor'), ('saylor', 'Michael Saylor'),
                ('sam bankman-fried', 'Sam Bankman-Fried'), ('sbf', 'Sam Bankman-Fried'),
                ('gary gensler', 'Gary Gensler'), ('gensler', 'Gary Gensler'),
                ('jerome powell', 'Jerome Powell'), ('powell', 'Jerome Powell'),
                ('cathie wood', 'Cathie Wood'),
            ],
            'ORG': [
                'sec', 'securities and exchange commission',
                'cftc', 'commodity futures trading commission',
                'federal reserve', 'fed', 'fdic',
                'treasury', 'treasury department',
                'doj', 'department of justice',
                'fbi', 'irs',
                'blackrock', 'fidelity', 'grayscale', 'ark invest',
                'microstrategy', 'tesla', 'square', 'block',
                'jpmorgan', 'goldman sachs', 'morgan stanley',
                'visa', 'mastercard', 'paypal',
            ]
        }
    
    def preprocess(self, text: str) -> ProcessedText:
        """
        Comprehensive text preprocessing
        
        Args:
            text: Raw input text
            
        Returns:
            ProcessedText with all preprocessing results
        """
        if not text:
            return self._create_empty_processed()
        
        original = text
        
        # Extract components before cleaning
        urls = self.url_pattern.findall(text)
        hashtags = self.hashtag_pattern.findall(text)
        mentions = self.mention_pattern.findall(text)
        
        # Detect language
        language, lang_confidence = self._detect_language(text)
        is_english = language == 'en' and lang_confidence > 0.5
        
        # Clean text
        cleaned = self._clean_text(text)
        
        # Tokenize and lemmatize
        tokens, lemmas = self._tokenize_and_lemmatize(cleaned)
        
        # Remove stopwords from tokens (but keep for lemmas analysis)
        filtered_tokens = [t for t in tokens if t.lower() not in self.stopwords]
        
        return ProcessedText(
            original=original,
            cleaned=cleaned,
            tokens=filtered_tokens,
            lemmas=lemmas,
            language=language,
            language_confidence=lang_confidence,
            is_english=is_english,
            hashtags=hashtags,
            mentions=mentions,
            urls=urls,
            word_count=len(filtered_tokens),
            char_count=len(cleaned)
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean text while preserving important content"""
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove mentions (but keep for separate extraction)
        text = re.sub(r'@\w+', ' ', text)
        
        # Normalize ticker symbols ($BTC -> BTC)
        text = re.sub(r'\$([A-Z]{2,5})\b', r'\1', text)
        
        # Remove excessive punctuation but keep some
        text = re.sub(r'([!?.]){3,}', r'\1\1', text)
        
        # Remove special characters except basic punctuation
        text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language of text"""
        try:
            from langdetect import detect_langs
            results = detect_langs(text)
            if results:
                return results[0].lang, results[0].prob
        except ImportError:
            logger.warning("langdetect not installed, defaulting to English")
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
        
        return 'en', 0.5  # Default to English with medium confidence
    
    def _tokenize_and_lemmatize(self, text: str) -> Tuple[List[str], List[str]]:
        """Tokenize and lemmatize text using spaCy"""
        nlp = _get_spacy_model()
        
        if nlp is None:
            # Fallback to basic tokenization
            tokens = text.lower().split()
            return tokens, tokens
        
        try:
            doc = nlp(text.lower())
            tokens = [token.text for token in doc if not token.is_space]
            lemmas = [token.lemma_ for token in doc if not token.is_space]
            return tokens, lemmas
        except Exception as e:
            logger.warning(f"spaCy processing failed: {e}")
            tokens = text.lower().split()
            return tokens, tokens
    
    def extract_entities(self, text: str) -> ExtractedEntities:
        """Extract named entities - public interface"""
        entities = self._create_empty_entities()
        
        # Pattern-based extraction (fast)
        self._extract_crypto_entities(text.lower(), entities)
        
        # NER extraction (slower but thorough)
        self._extract_spacy_entities(text, entities)
        
        # Deduplicate
        self._deduplicate_entities(entities)
        
        return ExtractedEntities(**entities)

    def _create_empty_entities(self) -> Dict:
        """Create empty entities structure"""
        return {
            'cryptocurrencies': [],
            'exchanges': [],
            'persons': [],
            'organizations': [],
            'money_amounts': [],
            'dates': [],
            'locations': [],
            'all_entities': []
        }

    def _deduplicate_entities(self, entities: Dict):
        """Deduplicate entity lists in place"""
        for key in ['cryptocurrencies', 'exchanges', 'persons', 'organizations', 
                    'money_amounts', 'dates', 'locations']:
            entities[key] = list(set(entities[key]))

    
    def _extract_crypto_entities(self, text_lower: str, entities: Dict):
        """Extract crypto-specific entities using pattern matching"""
        
        # Cryptocurrencies
        for pattern in self.crypto_patterns['CRYPTO']:
            if isinstance(pattern, tuple):
                search_term, normalized = pattern
            else:
                search_term = normalized = pattern
            
            if search_term in text_lower:
                entities['cryptocurrencies'].append(normalized)
                entities['all_entities'].append({
                    'text': normalized,
                    'label': 'CRYPTO',
                    'start': text_lower.find(search_term),
                    'end': text_lower.find(search_term) + len(search_term)
                })
        
        # Exchanges
        for exchange in self.crypto_patterns['EXCHANGE']:
            if exchange in text_lower:
                entities['exchanges'].append(exchange.title())
                entities['all_entities'].append({
                    'text': exchange.title(),
                    'label': 'EXCHANGE',
                    'start': text_lower.find(exchange),
                    'end': text_lower.find(exchange) + len(exchange)
                })
        
        # Persons
        for pattern in self.crypto_patterns['PERSON']:
            if isinstance(pattern, tuple):
                search_term, normalized = pattern
            else:
                search_term = normalized = pattern
            
            if search_term in text_lower:
                entities['persons'].append(normalized)
                entities['all_entities'].append({
                    'text': normalized,
                    'label': 'PERSON',
                    'start': text_lower.find(search_term),
                    'end': text_lower.find(search_term) + len(search_term)
                })
        
        # Organizations
        for org in self.crypto_patterns['ORG']:
            if org in text_lower:
                entities['organizations'].append(org.upper() if len(org) <= 4 else org.title())
                entities['all_entities'].append({
                    'text': org.upper() if len(org) <= 4 else org.title(),
                    'label': 'ORG',
                    'start': text_lower.find(org),
                    'end': text_lower.find(org) + len(org)
                })
        
        # Extract ticker symbols
        tickers = self.ticker_pattern.findall(text_lower.upper())
        for ticker in tickers:
            if ticker not in [e['text'] for e in entities['all_entities'] if e['label'] == 'CRYPTO']:
                entities['cryptocurrencies'].append(ticker)
                entities['all_entities'].append({
                    'text': ticker,
                    'label': 'CRYPTO',
                    'start': -1,
                    'end': -1
                })
                
    def _is_valid_money_entity(self, text: str) -> bool:
        """Return True if text is a valid money amount (not a hashtag, mention, or ticker)"""
        text = text.strip()
        # Exclude hashtags and mentions
        if text.startswith('#') or text.startswith('@'):
            return False
        # Allow $ followed by a digit (e.g. $12.51), but not $BTC
        if text.startswith('$'):
            if len(text) > 1 and text[1].isdigit():
                return True
            # Exclude $BTC, $ETH, etc.
            if text[1:].isalpha():
                return False
        # Exclude if it's just a crypto ticker (BTC, ETH, etc.)
        tickers = {t[1] for t in self.crypto_patterns['CRYPTO'] if isinstance(t, tuple)}
        if text.upper() in tickers:
            return False
        # Exclude if contains # or @ anywhere
        if '#' in text or '@' in text:
            return False
        # Otherwise, allow
        return True
    
    def _extract_spacy_entities(self, text: str, entities: Dict):
        """Extract entities using spaCy NER"""
        nlp = _get_spacy_model()
        
        if nlp is None:
            return
        
        try:
            doc = nlp(text)
            
            for ent in doc.ents:
                entity_data = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                }
                
                # Skip if already found by pattern matching
                if any(e['text'].lower() == ent.text.lower() for e in entities['all_entities']):
                    continue
                
                entities['all_entities'].append(entity_data)
                
                # Categorize by label
                if ent.label_ == 'MONEY':
                    if self._is_valid_money_entity(ent.text):
                        entities['money_amounts'].append(ent.text)
                elif ent.label_ == 'DATE':
                    entities['dates'].append(ent.text)
                elif ent.label_ in ('GPE', 'LOC'):
                    entities['locations'].append(ent.text)
                elif ent.label_ == 'PERSON':
                    if ent.text not in entities['persons']:
                        entities['persons'].append(ent.text)
                elif ent.label_ == 'ORG':
                    if ent.text not in entities['organizations']:
                        entities['organizations'].append(ent.text)
                        
        except Exception as e:
            logger.warning(f"spaCy NER failed: {e}")
    
    def _create_empty_processed(self) -> ProcessedText:
        """Create empty ProcessedText for null input"""
        return ProcessedText(
            original='',
            cleaned='',
            tokens=[],
            lemmas=[],
            language='unknown',
            language_confidence=0.0,
            is_english=False,
            hashtags=[],
            mentions=[],
            urls=[],
            word_count=0,
            char_count=0
        )


# Singleton instance

_text_processor_instance = None
_processor_lock = threading.Lock()

def get_text_processor() -> CryptoTextProcessor:
    """Get singleton text processor instance (thread-safe)"""
    global _text_processor_instance
    
    if _text_processor_instance is None:
        with _processor_lock:
            if _text_processor_instance is None:
                _text_processor_instance = CryptoTextProcessor()
    
    return _text_processor_instance