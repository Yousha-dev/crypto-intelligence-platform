"""
Semantic Knowledge Graph for Crypto News
Links entities, events, and relationships
"""

import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from django.utils import timezone
from collections import defaultdict
import json
import os

from django.conf import settings

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Knowledge graph entity"""
    id: str
    name: str
    entity_type: str  # cryptocurrency, exchange, person, organization, event
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: timezone.now())
    updated_at: datetime = field(default_factory=lambda: timezone.now())


@dataclass
class Relationship:
    """Relationship between entities"""
    source_id: str
    target_id: str
    relation_type: str  # mentions, affects, partners_with, competes_with, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = field(default_factory=lambda: timezone.now())


@dataclass
class Event:
    """News/Market event node"""
    id: str
    event_type: str  # price_movement, regulation, partnership, hack, etc.
    title: str
    description: str
    entities_involved: List[str] = field(default_factory=list)
    sentiment: str = "neutral"
    impact_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: timezone.now())
    source_articles: List[str] = field(default_factory=list)

class CryptoKnowledgeGraph:
    """
    Semantic Knowledge Graph for cryptocurrency domain
    Links entities, events, and their relationships
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path or os.path.join(
            settings.BASE_DIR, 'data', 'knowledge_graph.json'
        )   
        
        # Graph storage
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.events: Dict[str, Event] = {}
        
        # Indexes for fast lookup
        self.entity_by_type: Dict[str, Set[str]] = defaultdict(set)
        self.entity_by_name: Dict[str, str] = {}  # lowercase name -> id
        self.relationships_by_source: Dict[str, List[Relationship]] = defaultdict(list)
        self.relationships_by_target: Dict[str, List[Relationship]] = defaultdict(list)
        self.events_by_entity: Dict[str, List[str]] = defaultdict(list)
        
        # Lazy-loaded content services
        self._text_processor = None
        self._sentiment_analyzer = None
        
        # Predefined crypto entities
        self._initialize_base_entities()
        
        # Load persisted graph
        self._load_graph()
    
    # =========================================================================
    # CONTENT SERVICE PROPERTIES (Lazy Loading)
    # =========================================================================
    
    @property
    def text_processor(self):
        """Lazy load text processor service"""
        if self._text_processor is None:
            from myapp.services.content.text_processor import get_text_processor
            self._text_processor = get_text_processor()
        return self._text_processor
    
    @property
    def sentiment_analyzer(self):
        """Lazy load sentiment analyzer service"""
        if self._sentiment_analyzer is None:
            from myapp.services.content.sentiment_analyzer import get_sentiment_analyzer
            self._sentiment_analyzer = get_sentiment_analyzer()
        return self._sentiment_analyzer
    
    def _initialize_base_entities(self):
        """Initialize base cryptocurrency entities"""
        base_cryptos = [
            ("bitcoin", "BTC", "cryptocurrency"),
            ("ethereum", "ETH", "cryptocurrency"),
            ("binance_coin", "BNB", "cryptocurrency"),
            ("solana", "SOL", "cryptocurrency"),
            ("cardano", "ADA", "cryptocurrency"),
            ("ripple", "XRP", "cryptocurrency"),
            ("dogecoin", "DOGE", "cryptocurrency"),
            ("polkadot", "DOT", "cryptocurrency"),
            ("avalanche", "AVAX", "cryptocurrency"),
            ("chainlink", "LINK", "cryptocurrency"),
        ]
        
        base_exchanges = [
            ("binance", "Binance", "exchange"),
            ("coinbase", "Coinbase", "exchange"),
            ("kraken", "Kraken", "exchange"),
            ("ftx", "FTX", "exchange"),
            ("okx", "OKX", "exchange"),
        ]
        
        base_organizations = [
            ("sec", "SEC", "regulator"),
            ("cftc", "CFTC", "regulator"),
            ("federal_reserve", "Federal Reserve", "central_bank"),
            ("blackrock", "BlackRock", "institution"),
            ("grayscale", "Grayscale", "institution"),
        ]
        
        for entity_id, name, entity_type in base_cryptos + base_exchanges + base_organizations:
            if entity_id not in self.entities:
                self.add_entity(Entity(
                    id=entity_id,
                    name=name,
                    entity_type=entity_type,
                    properties={'is_base_entity': True}
                ))
    
    def add_entity(self, entity: Entity) -> bool:
        """Add entity to knowledge graph"""
        try:
            if entity.id in self.entities:
                # Update existing entity
                existing = self.entities[entity.id]
                existing.properties.update(entity.properties)
                existing.updated_at = timezone.now()
                return True
            
            self.entities[entity.id] = entity
            self.entity_by_type[entity.entity_type].add(entity.id)
            self.entity_by_name[entity.name.lower()] = entity.id
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding entity {entity.id}: {e}")
            return False
    
    def add_relationship(self, relationship: Relationship) -> bool:
        """Add relationship between entities"""
        try:
            # Verify entities exist
            if relationship.source_id not in self.entities:
                logger.warning(f"Source entity {relationship.source_id} not found")
                return False
            if relationship.target_id not in self.entities:
                logger.warning(f"Target entity {relationship.target_id} not found")
                return False
            
            # Check for duplicate relationship
            for existing in self.relationships:
                if (existing.source_id == relationship.source_id and 
                    existing.target_id == relationship.target_id and
                    existing.relation_type == relationship.relation_type):
                    # Update weight instead of adding duplicate
                    existing.weight += relationship.weight
                    return True
            
            self.relationships.append(relationship)
            self.relationships_by_source[relationship.source_id].append(relationship)
            self.relationships_by_target[relationship.target_id].append(relationship)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding relationship: {e}")
            return False
    
    def add_event(self, event: Event) -> bool:
        """Add event to knowledge graph"""
        try:
            self.events[event.id] = event
            
            # Link event to entities
            for entity_id in event.entities_involved:
                if entity_id in self.entities:
                    self.events_by_entity[entity_id].append(event.id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding event {event.id}: {e}")
            return False
    
    def extract_and_link_from_content(self, content: Dict) -> Dict[str, Any]:
        """
        Extract entities from content and create relationships
        Uses dedicated TextProcessor for entity extraction
        
        Args:
            content: News content (raw or processed format)
            
        Returns:
            Extraction results
        """
        results = {
            'entities_found': [],
            'relationships_created': [],
            'event_created': None
        }
        
        try:
            content_id = content.get('source_id', content.get('id', ''))
            
            # Build full text for processing
            full_text = f"{content.get('title', '')} {content.get('description', '')} {content.get('content', '')}"
            
            # Check if entities already extracted (by integrator service)
            extracted = content.get('extracted_entities', {})
            
            if extracted:
                # Use pre-extracted entities from integrator
                cryptos = extracted.get('cryptocurrencies', [])
                exchanges = extracted.get('exchanges', [])
                organizations = extracted.get('organizations', [])
                persons = extracted.get('persons', [])
            else:
                # Use TextProcessor service for entity extraction
                entities = self.text_processor.extract_entities(full_text)
                cryptos = entities.cryptocurrencies
                exchanges = entities.exchanges
                organizations = entities.organizations
                persons = entities.persons
            
            # Also extract from RAW fields (CryptoPanic instruments, CryptoCompare tags)
            raw_cryptos = self._extract_cryptos_from_raw_fields(content)
            cryptos = list(set(cryptos + list(raw_cryptos)))
            
            # Process cryptocurrencies
            for crypto in cryptos:
                entity_id = self._normalize_entity_id(crypto)
                if entity_id:
                    if entity_id not in self.entities:
                        self.add_entity(Entity(
                            id=entity_id,
                            name=crypto.upper() if len(crypto) <= 5 else crypto.title(),
                            entity_type='cryptocurrency',
                            properties={'discovered_from': content_id}
                        ))
                    results['entities_found'].append(entity_id)
            
            # Process exchanges
            for exchange in exchanges:
                entity_id = self._normalize_entity_id(exchange)
                if entity_id:
                    if entity_id not in self.entities:
                        self.add_entity(Entity(
                            id=entity_id,
                            name=exchange.title(),
                            entity_type='exchange',
                            properties={'discovered_from': content_id}
                        ))
                    results['entities_found'].append(entity_id)
            
            # Process organizations
            for org in organizations:
                entity_id = self._normalize_entity_id(org)
                if entity_id:
                    if entity_id not in self.entities:
                        self.add_entity(Entity(
                            id=entity_id,
                            name=org.upper() if len(org) <= 4 else org.title(),
                            entity_type='organization',
                            properties={'discovered_from': content_id}
                        ))
                    results['entities_found'].append(entity_id)
            
            # Process persons
            for person in persons:
                entity_id = self._normalize_entity_id(person)
                if entity_id:
                    if entity_id not in self.entities:
                        self.add_entity(Entity(
                            id=entity_id,
                            name=person,
                            entity_type='person',
                            properties={'discovered_from': content_id}
                        ))
                    results['entities_found'].append(entity_id)
            
            # Create relationships between co-occurring entities
            entities = list(set(results['entities_found']))
            for i, source_id in enumerate(entities):
                for target_id in entities[i+1:]:
                    relationship = Relationship(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type='co_mentioned',
                        properties={
                            'content_id': content_id,
                            'content_title': content.get('title', '')
                        },
                        weight=1.0
                    )
                    if self.add_relationship(relationship):
                        results['relationships_created'].append(
                            f"{source_id} -> {target_id}"
                        )
            
            # Create event if significant
            trust_score = content.get('trust_score', 5.0)
            if len(entities) >= 2 and trust_score >= 6.0:
                event = Event(
                    id=f"event_{content_id}",
                    event_type=self._classify_event_type(content),
                    title=content.get('title', ''),
                    description=(content.get('description') or content.get('content', ''))[:500],
                    entities_involved=entities,
                    sentiment=self._get_sentiment_from_article(content, full_text),
                    impact_score=self._calculate_impact_score(content, full_text),
                    source_articles=[content_id]
                )
                
                if self.add_event(event):
                    results['event_created'] = event.id
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting from article: {e}")
            return results
    
    def _extract_cryptos_from_raw_fields(self, article: Dict) -> Set[str]:
        """Extract cryptocurrency symbols from RAW API fields"""
        cryptos = set()
        
        # CryptoPanic instruments
        instruments = article.get('instruments', [])
        if instruments:
            for inst in instruments:
                if isinstance(inst, dict):
                    code = inst.get('code', '').lower()
                    if code:
                        cryptos.add(code)
        
        # CryptoCompare tags
        tags = article.get('tags', '')
        if isinstance(tags, str) and tags:
            crypto_symbols = {'btc', 'eth', 'sol', 'ada', 'xrp', 'doge', 'bnb', 'dot', 'avax', 'link', 'matic', 'ltc'}
            for tag in tags.split('|'):
                tag_lower = tag.strip().lower()
                if tag_lower in crypto_symbols:
                    cryptos.add(tag_lower)
        
        return cryptos
    
    def _get_sentiment_from_article(self, article: Dict, full_text: str = '') -> str:
        """Get sentiment from article using SentimentAnalyzer service"""
        
        # Check sentiment_analysis.label (CORRECT FIELD)
        sentiment_data = article.get('sentiment_analysis', {})
        if sentiment_data and isinstance(sentiment_data, dict):
            label = sentiment_data.get('label', '').lower()
            
            # Map to knowledge graph format
            if label in ['positive', 'very_bullish', 'bullish']:
                return 'bullish'
            elif label in ['negative', 'very_bearish', 'bearish']:
                return 'bearish'
            return 'neutral'
        
        # Fallback: Use SentimentAnalyzer
        if not full_text:
            full_text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
        
        if full_text.strip():
            try:
                result = self.sentiment_analyzer.analyze(full_text)
                
                # ✅ ALREADY HAS None check (good!)
                if result:
                    label = result.label.value.lower()
                    
                    if label in ['very_bullish', 'bullish']:
                        return 'bullish'
                    elif label in ['very_bearish', 'bearish']:
                        return 'bearish'
                    else:
                        return 'neutral'
                else:
                    return 'neutral'
            except Exception as e:
                logger.warning(f"Error analyzing sentiment: {e}")
                return 'neutral'
        
        return 'neutral'

    
    def _extract_entities_from_raw(self, article: Dict) -> Dict[str, List[str]]:
        """
        Extract entities from RAW article data
        Used when credibility engine hasn't processed the article yet
        """
        extracted = {
            'cryptocurrencies': [],
            'organizations': [],
            'exchanges': [],
            'persons': []
        }
        
        # Extract from CryptoPanic instruments
        instruments = article.get('instruments', [])
        if instruments:
            for inst in instruments:
                if isinstance(inst, dict):
                    code = inst.get('code', '')
                    if code:
                        extracted['cryptocurrencies'].append(code.lower())
        
        # Extract from CryptoCompare tags
        tags = article.get('tags', '')
        if isinstance(tags, str) and tags:
            for tag in tags.split('|'):
                tag_lower = tag.lower().strip()
                if tag_lower in ['btc', 'eth', 'sol', 'ada', 'xrp', 'doge', 'bnb', 'dot', 'avax', 'link']:
                    extracted['cryptocurrencies'].append(tag_lower)
        
        # Simple text-based extraction from title/content
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        
        # Known cryptocurrencies
        crypto_patterns = {
            'bitcoin': 'bitcoin', 'btc': 'bitcoin',
            'ethereum': 'ethereum', 'eth': 'ethereum',
            'solana': 'solana', 'sol': 'solana',
            'cardano': 'cardano', 'ada': 'cardano',
            'ripple': 'ripple', 'xrp': 'ripple',
            'dogecoin': 'dogecoin', 'doge': 'dogecoin',
            'binance': 'binance_coin', 'bnb': 'binance_coin',
            'polkadot': 'polkadot', 'dot': 'polkadot',
            'avalanche': 'avalanche', 'avax': 'avalanche',
            'chainlink': 'chainlink', 'link': 'chainlink',
        }
        
        for pattern, entity_id in crypto_patterns.items():
            if pattern in text and entity_id not in extracted['cryptocurrencies']:
                extracted['cryptocurrencies'].append(entity_id)
        
        # Known exchanges
        exchange_patterns = ['binance', 'coinbase', 'kraken', 'okx', 'bybit', 'kucoin']
        for exchange in exchange_patterns:
            if exchange in text:
                extracted['exchanges'].append(exchange)
        
        # Known organizations/regulators
        org_patterns = ['sec', 'cftc', 'federal reserve', 'blackrock', 'grayscale', 'fidelity']
        for org in org_patterns:
            if org in text:
                extracted['organizations'].append(org.replace(' ', '_'))
        
        return extracted
    
    def _extract_sentiment_from_article(self, article: Dict) -> str:
        """Extract sentiment from article (raw or processed)"""
        # Check processed sentiment first
        sentiment_data = article.get('sentiment_analysis', {})
        if sentiment_data and isinstance(sentiment_data, dict):
            label = sentiment_data.get('label', '').lower()
            if label:
                return label
        
        # Fallback: simple rule-based for raw data
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        
        bullish_words = ['surge', 'rally', 'soar', 'gain', 'bullish', 'up', 'high', 'record']
        bearish_words = ['crash', 'dump', 'fall', 'drop', 'bearish', 'down', 'low', 'decline']
        
        bullish_count = sum(1 for word in bullish_words if word in text)
        bearish_count = sum(1 for word in bearish_words if word in text)
        
        if bullish_count > bearish_count:
            return 'bullish'
        elif bearish_count > bullish_count:
            return 'bearish'
        return 'neutral'
    
    
    def _normalize_entity_id(self, name: str) -> Optional[str]:
        """Normalize entity name to ID"""
        if not name:
            return None
        
        normalized = name.lower().strip().replace(' ', '_')
        
        # Check if it matches known entity
        if normalized in self.entity_by_name:
            return self.entity_by_name[normalized]
        
        # Check common aliases
        aliases = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'bnb': 'binance_coin',
            'sol': 'solana',
            'xrp': 'ripple',
            'ada': 'cardano',
            'doge': 'dogecoin',
            'dot': 'polkadot',
        }
        
        if normalized in aliases:
            return aliases[normalized]
        
        return normalized
    
    def _classify_event_type(self, article: Dict) -> str:
        """
        Classify event type using TextProcessor for preprocessing
        """
        title = article.get('title', '').lower()
        content = (article.get('description', '') or article.get('content', '')).lower()
        full_text = f"{title} {content}"
        
        # Use TextProcessor to preprocess text
        try:
            processed = self.text_processor.preprocess(full_text)
            text_lower = ' '.join(processed.tokens) if processed.tokens else full_text
        except Exception:
            text_lower = full_text
        
        # Event type classification based on keywords
        event_patterns = {
            'regulation': ['sec', 'regulation', 'regulatory', 'law', 'legal', 'compliance', 'ban', 'approve', 'etf', 'lawsuit'],
            'partnership': ['partner', 'collaboration', 'integrate', 'alliance', 'deal', 'agreement'],
            'product_launch': ['launch', 'release', 'announce', 'introduce', 'unveil', 'upgrade', 'update'],
            'market_movement': ['surge', 'crash', 'rally', 'dump', 'pump', 'ath', 'all-time', 'record', 'breakout'],
            'security_incident': ['hack', 'breach', 'exploit', 'vulnerability', 'attack', 'stolen', 'scam'],
            'adoption': ['adopt', 'accept', 'payment', 'merchant', 'institutional', 'mainstream'],
            'technology': ['blockchain', 'protocol', 'network', 'consensus', 'scaling', 'layer'],
        }
        
        for event_type, keywords in event_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return event_type
        
        return 'general_news'
    
    def _calculate_impact_score(self, article: Dict, full_text: str = '') -> float:
        """
        Calculate event impact score using SentimentAnalyzer for extremity
        """
        score = 5.0  # Base score
        
        # Trust score contribution (if available)
        trust = article.get('trust_score', 5.0)
        score += (trust - 5.0) * 0.3
        
        # Check CryptoPanic votes (RAW)
        votes = article.get('votes', {})
        if votes:
            positive = votes.get('positive', 0)
            important = votes.get('important', 0)
            score += min((positive + important * 2) * 0.1, 2.0)
        
        # Check CryptoCompare upvotes (RAW)
        upvotes = article.get('upvotes', 0)
        downvotes = article.get('downvotes', 0)
        if upvotes > 0:
            vote_ratio = upvotes / (upvotes + downvotes) if (upvotes + downvotes) > 0 else 0.5
            score += vote_ratio * 1.0
        
        # Sentiment extremity contribution with None check
        if not full_text:
            full_text = f"{article.get('title', '')} {article.get('description', '')}"
        
        if full_text.strip():
            try:
                sentiment_result = self.sentiment_analyzer.analyze(full_text)
                
                # Check if result is not None
                if sentiment_result is not None:
                    # Extreme sentiment = higher impact
                    extremity = abs(sentiment_result.score)
                    score += extremity * 2
            except Exception as e:
                logger.warning(f"Error analyzing sentiment for impact: {e}")
        
        return min(max(score, 0), 10)
    
    def get_entity_context(self, entity_id: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get context for an entity including related entities and events
        
        Args:
            entity_id: Entity to get context for
            depth: How many relationship hops to include
            
        Returns:
            Entity context with relationships and events
        """
        if entity_id not in self.entities:
            return {'error': f'Entity {entity_id} not found'}
        
        entity = self.entities[entity_id]
        
        # Get direct relationships
        outgoing = self.relationships_by_source.get(entity_id, [])
        incoming = self.relationships_by_target.get(entity_id, [])
        
        # Get related entities
        related_entities = {}
        for rel in outgoing:
            if rel.target_id in self.entities:
                related_entities[rel.target_id] = {
                    'entity': self.entities[rel.target_id],
                    'relation': rel.relation_type,
                    'direction': 'outgoing',
                    'weight': rel.weight
                }
        
        for rel in incoming:
            if rel.source_id in self.entities:
                if rel.source_id not in related_entities:
                    related_entities[rel.source_id] = {
                        'entity': self.entities[rel.source_id],
                        'relation': rel.relation_type,
                        'direction': 'incoming',
                        'weight': rel.weight
                    }
        
        # Get events
        events = [
            self.events[event_id] 
            for event_id in self.events_by_entity.get(entity_id, [])
            if event_id in self.events
        ]
        
        # Sort events by timestamp
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return {
            'entity': {
                'id': entity.id,
                'name': entity.name,
                'type': entity.entity_type,
                'properties': entity.properties
            },
            'related_entities': [
                {
                    'id': eid,
                    'name': data['entity'].name,
                    'type': data['entity'].entity_type,
                    'relation': data['relation'],
                    'direction': data['direction'],
                    'weight': data['weight']
                }
                for eid, data in sorted(
                    related_entities.items(), 
                    key=lambda x: x[1]['weight'], 
                    reverse=True
                )[:20]
            ],
            'recent_events': [
                {
                    'id': e.id,
                    'type': e.event_type,
                    'title': e.title,
                    'sentiment': e.sentiment,
                    'impact_score': e.impact_score,
                    'timestamp': e.timestamp.isoformat()
                }
                for e in events[:10]
            ],
            'statistics': {
                'total_relationships': len(outgoing) + len(incoming),
                'total_events': len(events),
                'outgoing_connections': len(outgoing),
                'incoming_connections': len(incoming)
            }
        }
        
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 4) -> List[List[str]]:
        """
        Find paths between two entities
        
        Args:
            source_id: Starting entity
            target_id: Target entity
            max_depth: Maximum path length
            
        Returns:
            List of paths (each path is list of entity IDs)
        """
        if source_id not in self.entities or target_id not in self.entities:
            return []
        
        paths = []
        visited = set()
        
        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            if current == target_id:
                paths.append(path.copy())
                return
            if current in visited:
                return
            
            visited.add(current)
            
            # Explore outgoing relationships
            for rel in self.relationships_by_source.get(current, []):
                path.append(rel.target_id)
                dfs(rel.target_id, path, depth + 1)
                path.pop()
            
            # Explore incoming relationships
            for rel in self.relationships_by_target.get(current, []):
                path.append(rel.source_id)
                dfs(rel.source_id, path, depth + 1)
                path.pop()
            
            visited.remove(current)
        
        dfs(source_id, [source_id], 0)
        
        return paths
    
    def get_trending_entities(self, hours_back: int = 24, limit: int = 10) -> List[Dict]:
        """Get entities with most recent activity"""
        cutoff = timezone.now() - timedelta(hours=hours_back)
        
        entity_activity = defaultdict(int)
        
        # Count recent events per entity
        for event in self.events.values():
            if event.timestamp >= cutoff:
                for entity_id in event.entities_involved:
                    entity_activity[entity_id] += 1
        
        # Sort by activity
        trending = sorted(
            entity_activity.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {
                'entity_id': entity_id,
                'name': self.entities[entity_id].name if entity_id in self.entities else entity_id,
                'type': self.entities[entity_id].entity_type if entity_id in self.entities else 'unknown',
                'event_count': count
            }
            for entity_id, count in trending
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        try:
            return {
                'total_entities': len(self.entities) if hasattr(self, 'entities') else 0,
                'total_relationships': len(self.relationships) if hasattr(self, 'relationships') else 0,
                'total_events': len(self.events) if hasattr(self, 'events') else 0,
                'entity_types': self._count_entity_types() if hasattr(self, 'entities') else {},
                'relationship_types': self._count_relationship_types() if hasattr(self, 'relationships') else {},
                'last_updated': getattr(self, 'last_updated', None),
            }
        except Exception as e:
            logger.warning(f"Error getting KG statistics: {e}")
            return {
                'total_entities': 0,
                'total_relationships': 0,
                'error': str(e)
            }
    
    def _count_entity_types(self) -> Dict[str, int]:
        """Count entities by type"""
        counts = {}
        for entity in getattr(self, 'entities', {}).values():
            # Entity is a dataclass, use attribute access not .get()
            entity_type = entity.entity_type if hasattr(entity, 'entity_type') else 'unknown'
            counts[entity_type] = counts.get(entity_type, 0) + 1
        return counts
    
    def _count_relationship_types(self) -> Dict[str, int]:
        """Count relationships by type"""
        counts = {}
        for rel in getattr(self, 'relationships', []):
            # Relationship is a dataclass, use attribute access not .get()
            rel_type = rel.relation_type if hasattr(rel, 'relation_type') else 'unknown'
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts

    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'total_events': len(self.events),
            'entities_by_type': {
                entity_type: len(entity_ids)
                for entity_type, entity_ids in self.entity_by_type.items()
            },
            'relationship_types': list(set(r.relation_type for r in self.relationships)),
            'event_types': list(set(e.event_type for e in self.events.values()))
        }
    
    def save_graph(self):
        """Persist knowledge graph to disk"""
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            
            data = {
                'entities': {
                    eid: {
                        'id': e.id,
                        'name': e.name,
                        'entity_type': e.entity_type,
                        'properties': e.properties,
                        'created_at': e.created_at.isoformat(),
                        'updated_at': e.updated_at.isoformat()
                    }
                    for eid, e in self.entities.items()
                },
                'relationships': [
                    {
                        'source_id': r.source_id,
                        'target_id': r.target_id,
                        'relation_type': r.relation_type,
                        'properties': r.properties,
                        'weight': r.weight,
                        'created_at': r.created_at.isoformat()
                    }
                    for r in self.relationships
                ],
                'events': {
                    eid: {
                        'id': e.id,
                        'event_type': e.event_type,
                        'title': e.title,
                        'description': e.description,
                        'entities_involved': e.entities_involved,
                        'sentiment': e.sentiment,
                        'impact_score': e.impact_score,
                        'timestamp': e.timestamp.isoformat(),
                        'source_articles': e.source_articles
                    }
                    for eid, e in self.events.items()
                }
            }
            
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved knowledge graph with {len(self.entities)} entities")
            
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")
    
    def _load_graph(self):
        """Load knowledge graph from disk"""
        try:
            if not os.path.exists(self.persist_path):
                return
            
            with open(self.persist_path, 'r') as f:
                data = json.load(f)
            
            # Load entities
            for eid, e_data in data.get('entities', {}).items():
                entity = Entity(
                    id=e_data['id'],
                    name=e_data['name'],
                    entity_type=e_data['entity_type'],
                    properties=e_data.get('properties', {}),
                    created_at=datetime.fromisoformat(e_data['created_at']),
                    updated_at=datetime.fromisoformat(e_data['updated_at'])
                )
                self.entities[eid] = entity
                self.entity_by_type[entity.entity_type].add(eid)
                self.entity_by_name[entity.name.lower()] = eid
            
            # Load relationships
            for r_data in data.get('relationships', []):
                relationship = Relationship(
                    source_id=r_data['source_id'],
                    target_id=r_data['target_id'],
                    relation_type=r_data['relation_type'],
                    properties=r_data.get('properties', {}),
                    weight=r_data.get('weight', 1.0),
                    created_at=datetime.fromisoformat(r_data['created_at'])
                )
                self.relationships.append(relationship)
                self.relationships_by_source[relationship.source_id].append(relationship)
                self.relationships_by_target[relationship.target_id].append(relationship)
            
            # Load events
            for eid, e_data in data.get('events', {}).items():
                event = Event(
                    id=e_data['id'],
                    event_type=e_data['event_type'],
                    title=e_data['title'],
                    description=e_data['description'],
                    entities_involved=e_data.get('entities_involved', []),
                    sentiment=e_data.get('sentiment', 'neutral'),
                    impact_score=e_data.get('impact_score', 0),
                    timestamp=datetime.fromisoformat(e_data['timestamp']),
                    source_articles=e_data.get('source_articles', [])
                )
                self.events[eid] = event
                for entity_id in event.entities_involved:
                    self.events_by_entity[entity_id].append(eid)
            
            logger.info(f"Loaded knowledge graph with {len(self.entities)} entities")
            
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")


# Singleton instance
_knowledge_graph_instance = None


def get_knowledge_graph() -> CryptoKnowledgeGraph:
    """Get singleton knowledge graph instance"""
    global _knowledge_graph_instance
    
    if _knowledge_graph_instance is None:
        _knowledge_graph_instance = CryptoKnowledgeGraph()
    
    return _knowledge_graph_instance