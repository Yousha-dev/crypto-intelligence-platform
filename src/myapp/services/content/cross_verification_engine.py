"""
Production-Ready Advanced Cross-Verification Engine
Uses semantic similarity, claim extraction, and source diversity analysis

Install requirements:
pip install sentence-transformers scikit-learn numpy faiss-cpu

File: myapp/services/content/cross_verification_engine.py
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from django.utils import timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import timezone as timezoneDt
from collections import defaultdict
import hashlib
import json

logger = logging.getLogger(__name__)

# Lazy imports for heavy dependencies
_sentence_transformer = None
_faiss_index = None

@dataclass
class Claim:
    """A specific claim extracted from content"""
    text: str
    entity: str  # Main entity (BTC, ETH, etc.)
    claim_type: str  # price_movement, regulation, technology, etc.
    confidence: float
    source_id: str
    timestamp: datetime


@dataclass
class CrossReference:
    """A cross-reference match between two pieces of content"""
    source_id: str
    target_id: str
    similarity_score: float
    temporal_proximity: float  # 0-1, how close in time
    source_diversity: float  # 0-1, how different the sources are
    claim_overlap: float  # 0-1, how many claims match
    contradiction_score: float  # 0-1, do they contradict
    corroboration_strength: float  # Overall strength
    matched_claims: List[Tuple[str, str]]  # [(source_claim, target_claim)]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of cross-verification analysis"""
    content_id: str
    total_references: int
    unique_sources: int
    avg_similarity: float
    corroboration_score: float  # 0-10
    confidence: float  # 0-1
    references: List[CrossReference]
    verified_claims: List[str]
    contradicted_claims: List[str]
    source_diversity_score: float
    temporal_clustering_score: float
    flags: List[str]
    reasoning: str


class AdvancedCrossVerificationEngine:
    """
    Production-grade cross-verification engine
    Uses semantic similarity, claim extraction, and multi-source analysis
    """
    
    def __init__(self):
        self._embedding_model = None
        self._initialized = False
        
        # Configuration
        self.config = {
            'similarity_threshold': 0.4,  # Cosine similarity threshold
            'temporal_window_hours': 24,
            'min_references_for_high_confidence': 3,
            'source_diversity_weight': 0.3,
            'temporal_weight': 0.2,
            'semantic_weight': 0.5,
            'max_references_to_return': 20,
            'contradiction_threshold': 0.4,
        }
        
        # Source reputation database (simplified - should be in DB)
        self.source_tiers = {
            # Tier 1: Premium sources (weight: 1.0)
            'tier1': ['reuters', 'bloomberg', 'wsj', 'ft', 'coindesk'],
            # Tier 2: Established sources (weight: 0.8)
            'tier2': ['cointelegraph', 'cryptonews', 'messari', 'theblock'],
            # Tier 3: Community sources (weight: 0.6)
            'tier3': ['cryptocompare', 'cryptopanic'],
            # Social: Lower weight but valuable for trends (weight: 0.4)
            'social': ['twitter', 'reddit', 'youtube']
        }
        
        # Claim extraction patterns (simplified - use NLP in production)
        self.claim_patterns = {
            'price_movement': [
                'price', 'surged', 'dropped', 'reached', 'hit', 'crossed',
                'trading at', 'valued at', 'market cap'
            ],
            'regulation': [
                'sec', 'regulation', 'lawsuit', 'ban', 'approval', 'legal',
                'compliance', 'enforcement', 'regulatory'
            ],
            'technology': [
                'upgrade', 'fork', 'network', 'launch', 'release', 'protocol',
                'blockchain', 'consensus', 'implementation'
            ],
            'adoption': [
                'partnership', 'integration', 'accepted', 'adoption', 'usage',
                'merchant', 'payment', 'institutional'
            ],
            'security': [
                'hack', 'vulnerability', 'breach', 'exploit', 'attack',
                'security', 'bug', 'theft'
            ]
        }
        
        logger.info("Advanced Cross-Verification Engine initialized")
    
    def _ensure_initialized(self):
        """Lazy load embedding model"""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info("Loading sentence transformer for semantic similarity...")
            self._embedding_model = SentenceTransformer('all-mpnet-base-v2')
            self._initialized = True
            logger.info("Embedding model loaded successfully")
            
        except ImportError:
            logger.error("sentence-transformers not installed")
            logger.info("Install: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts using embeddings
        Returns: float between 0-1
        """
        self._ensure_initialized()
        
        try:
            # Generate embeddings
            embeddings = self._embedding_model.encode([text1, text2])
            
            # Compute cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def extract_claims(self, content_data: Dict) -> List[Claim]:
        """
        Extract verifiable claims from content
        In production, use NLP pipeline (spaCy, AllenNLP) for better extraction
        """
        claims = []
        
        try:
            # Get text
            title = content_data.get('title', '')
            description = content_data.get('description', '') or content_data.get('content', '')
            text = f"{title}. {description}".lower()
            
            # Get entities
            entities = content_data.get('extracted_entities', {})
            cryptos = entities.get('cryptocurrencies', [])
            
            # Extract claims by pattern matching
            for claim_type, patterns in self.claim_patterns.items():
                for pattern in patterns:
                    if pattern in text:
                        # Create claim for each relevant crypto
                        for crypto in cryptos[:3]:  # Top 3 mentioned cryptos
                            # Extract sentence containing the pattern
                            sentences = text.split('.')
                            for sentence in sentences:
                                if pattern in sentence and crypto.lower() in sentence:
                                    claim = Claim(
                                        text=sentence.strip()[:200],
                                        entity=crypto,
                                        claim_type=claim_type,
                                        confidence=0.7,  # Simplified
                                        source_id=content_data.get('source_id', ''),
                                        timestamp=content_data.get('created_at') or timezone.now()
                                    )
                                    claims.append(claim)
                                    break  # One claim per pattern per crypto
            
            return claims
            
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            return []
    
    def get_source_tier(self, platform: str) -> Tuple[str, float]:
        """Get source tier and weight"""
        platform_lower = platform.lower()
        
        for tier, sources in self.source_tiers.items():
            if platform_lower in sources:
                weights = {'tier1': 1.0, 'tier2': 0.8, 'tier3': 0.6, 'social': 0.4}
                return tier, weights.get(tier, 0.5)
        
        return 'unknown', 0.5
    
    def calculate_source_diversity(self, references: List[Dict]) -> float:
        """
        Calculate how diverse the sources are
        Higher score = more independent confirmation
        """
        if not references:
            return 0.0
        
        # Count unique platforms
        platforms = set(ref.get('platform', 'unknown') for ref in references)
        
        # Count tier distribution
        tier_counts = defaultdict(int)
        for ref in references:
            tier, _ = self.get_source_tier(ref.get('platform', ''))
            tier_counts[tier] += 1
        
        # Diversity score
        platform_diversity = len(platforms) / max(len(references), 1)
        tier_diversity = len(tier_counts) / 4.0  # Max 4 tiers
        
        return (platform_diversity * 0.6 + tier_diversity * 0.4)
    
    def calculate_temporal_proximity(self, time1: datetime, time2: datetime) -> float:
        """
        Calculate temporal proximity (0-1)
        1.0 = same time, decreases with time difference
        """
        try:
            if not time1 or not time2:
                return 0.5
            
            # Ensure timezone aware
            if time1.tzinfo is None:
                time1 = time1.replace(tzinfo=timezoneDt.utc)
            if time2.tzinfo is None:
                time2 = time2.replace(tzinfo=timezoneDt.utc)
            
            # Calculate hours difference
            delta = abs((time1 - time2).total_seconds() / 3600)
            
            # Decay function: high score for <6h, low for >48h
            if delta < 1:
                return 1.0
            elif delta < 6:
                return 0.9
            elif delta < 24:
                return 0.7
            elif delta < 48:
                return 0.4
            else:
                return 0.1
                
        except Exception as e:
            logger.warning(f"Error calculating temporal proximity: {e}")
            return 0.5
    
    def detect_contradiction(self, claim1: str, claim2: str) -> float:
        """
        Detect if two claims contradict each other
        Uses sentiment and semantic analysis
        Returns: 0-1 (1 = strong contradiction)
        """
        try:
            # Simple heuristic: look for opposing sentiments
            positive_words = ['surge', 'gain', 'increase', 'bullish', 'positive', 'approval']
            negative_words = ['drop', 'fall', 'decrease', 'bearish', 'negative', 'rejection']
            
            claim1_lower = claim1.lower()
            claim2_lower = claim2.lower()
            
            claim1_positive = any(word in claim1_lower for word in positive_words)
            claim1_negative = any(word in claim1_lower for word in negative_words)
            
            claim2_positive = any(word in claim2_lower for word in positive_words)
            claim2_negative = any(word in claim2_lower for word in negative_words)
            
            # Contradiction if opposing sentiments on same topic
            if (claim1_positive and claim2_negative) or (claim1_negative and claim2_positive):
                # Compute semantic similarity to ensure they're about same thing
                similarity = self.compute_semantic_similarity(claim1, claim2)
                if similarity > 0.6:  # Only flag if semantically similar
                    return 0.8
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error detecting contradiction: {e}")
            return 0.0
    
    def find_similar_content(self, content_data: Dict, 
                            candidate_pool: List[Dict]) -> List[CrossReference]:
        """
        Find semantically similar content from candidate pool
        Returns ranked list of cross-references
        """
        self._ensure_initialized()
        
        if not candidate_pool:
            return []
        
        try:
            source_text = f"{content_data.get('title', '')} {content_data.get('description', '')}"
            source_claims = self.extract_claims(content_data)
            source_time = content_data.get('created_at') or timezone.now()
            
            references = []
            
            for candidate in candidate_pool:
                # Skip self-reference
                if candidate.get('source_id') == content_data.get('source_id'):
                    continue
                
                # Compute semantic similarity
                target_text = f"{candidate.get('title', '')} {candidate.get('description', '')}"
                similarity = self.compute_semantic_similarity(source_text, target_text)
                
                if similarity < self.config['similarity_threshold']:
                    continue
                
                # Extract claims from candidate
                target_claims = self.extract_claims(candidate)
                
                # Match claims
                matched_claims = []
                claim_similarities = []
                contradictions = []
                
                for source_claim in source_claims:
                    for target_claim in target_claims:
                        # Only compare claims about same entity
                        if source_claim.entity.lower() == target_claim.entity.lower():
                            claim_sim = self.compute_semantic_similarity(
                                source_claim.text, 
                                target_claim.text
                            )
                            
                            if claim_sim > 0.6:
                                matched_claims.append((source_claim.text, target_claim.text))
                                claim_similarities.append(claim_sim)
                                
                                # Check for contradiction
                                contradiction = self.detect_contradiction(
                                    source_claim.text,
                                    target_claim.text
                                )
                                contradictions.append(contradiction)
                
                # Calculate scores
                temporal_proximity = self.calculate_temporal_proximity(
                    source_time,
                    candidate.get('created_at') or timezone.now()
                )
                
                _, source_weight = self.get_source_tier(candidate.get('platform', ''))
                
                claim_overlap = (len(matched_claims) / max(len(source_claims), 1)) if source_claims else 0
                
                avg_contradiction = sum(contradictions) / len(contradictions) if contradictions else 0
                
                # Corroboration strength (weighted combination)
                corroboration = (
                    similarity * self.config['semantic_weight'] +
                    temporal_proximity * self.config['temporal_weight'] +
                    source_weight * self.config['source_diversity_weight']
                ) * (1 - avg_contradiction)  # Penalize contradictions
                
                reference = CrossReference(
                    source_id=content_data.get('source_id', ''),
                    target_id=candidate.get('source_id', ''),
                    similarity_score=similarity,
                    temporal_proximity=temporal_proximity,
                    source_diversity=source_weight,
                    claim_overlap=claim_overlap,
                    contradiction_score=avg_contradiction,
                    corroboration_strength=corroboration,
                    matched_claims=matched_claims[:5],  # Top 5
                    metadata={
                        'target_platform': candidate.get('platform'),
                        'target_title': candidate.get('title', '')[:100],
                        'target_trust_score': candidate.get('trust_score', 0)
                    }
                )
                
                references.append(reference)
            
            # Sort by corroboration strength
            references.sort(key=lambda x: x.corroboration_strength, reverse=True)
            
            return references[:self.config['max_references_to_return']]
            
        except Exception as e:
            logger.error(f"Error finding similar content: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def verify_content(self, content_data: Dict, 
                    candidate_pool: List[Dict]) -> VerificationResult:
        """
        Perform comprehensive cross-verification
        
        Args:
            content_data: Content to verify
            candidate_pool: List of candidate articles/posts for comparison
        
        Returns:
            VerificationResult with detailed analysis
        """
        try:
            if not candidate_pool:
                return self._create_empty_result(content_data.get('source_id', ''))
            
            # Remove duplicates
            seen_ids = set()
            unique_candidates = []
            for candidate in candidate_pool:
                cid = candidate.get('source_id')
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    unique_candidates.append(candidate)
            
            if not unique_candidates:
                return self._create_empty_result(content_data.get('source_id', ''))
            
            # Find similar content
            references = self.find_similar_content(content_data, unique_candidates)
            
            if not references:
                return self._create_empty_result(content_data.get('source_id', ''))
            
            # Calculate aggregate scores
            unique_sources = len(set(ref.metadata.get('target_platform') for ref in references))
            avg_similarity = sum(ref.similarity_score for ref in references) / len(references)
            
            # Source diversity
            source_diversity = self.calculate_source_diversity([
                {'platform': ref.metadata.get('target_platform')} for ref in references
            ])
            
            # Temporal clustering (are references clustered in time?)
            temporal_scores = [ref.temporal_proximity for ref in references]
            temporal_clustering = sum(temporal_scores) / len(temporal_scores)
            
            # Verified vs contradicted claims
            verified_claims = []
            contradicted_claims = []
            
            for ref in references:
                for source_claim, target_claim in ref.matched_claims:
                    if ref.contradiction_score < 0.3:
                        verified_claims.append(source_claim)
                    else:
                        contradicted_claims.append(source_claim)
            
            # Overall corroboration score (0-10 scale)
            base_score = avg_similarity * 10
            
            # Bonuses
            if len(references) >= 3:
                base_score += 1.0  # Multiple confirmations
            if source_diversity > 0.7:
                base_score += 1.0  # Diverse sources
            if temporal_clustering > 0.7:
                base_score += 0.5  # Clustered in time (event-based)
            
            # Penalties
            if contradicted_claims:
                base_score -= len(contradicted_claims) * 0.5
            
            corroboration_score = min(10.0, max(0.0, base_score))
            
            # Confidence calculation
            confidence = min(1.0, (
                (len(references) / self.config['min_references_for_high_confidence']) * 0.4 +
                source_diversity * 0.3 +
                avg_similarity * 0.3
            ))
            
            # Generate flags
            flags = []
            if contradicted_claims:
                flags.append('contradictions_detected')
            if source_diversity < 0.3:
                flags.append('low_source_diversity')
            if temporal_clustering < 0.3:
                flags.append('scattered_temporal_distribution')
            if len(references) < 2:
                flags.append('insufficient_references')
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                references, unique_sources, source_diversity, 
                verified_claims, contradicted_claims
            )
            
            return VerificationResult(
                content_id=content_data.get('source_id', ''),
                total_references=len(references),
                unique_sources=unique_sources,
                avg_similarity=avg_similarity,
                corroboration_score=corroboration_score,
                confidence=confidence,
                references=references[:10],  # Top 10 for storage
                verified_claims=list(set(verified_claims))[:5],
                contradicted_claims=list(set(contradicted_claims))[:5],
                source_diversity_score=source_diversity,
                temporal_clustering_score=temporal_clustering,
                flags=flags,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error in verify_content: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_result(content_data.get('source_id', ''))
    
    def _create_empty_result(self, content_id: str) -> VerificationResult:
        """Create empty verification result"""
        return VerificationResult(
            content_id=content_id,
            total_references=0,
            unique_sources=0,
            avg_similarity=0.0,
            corroboration_score=5.0,  # Neutral
            confidence=0.0,
            references=[],
            verified_claims=[],
            contradicted_claims=[],
            source_diversity_score=0.0,
            temporal_clustering_score=0.0,
            flags=['no_cross_references_found'],
            reasoning="No related content found for cross-verification"
        )
    
    def _generate_reasoning(self, references: List[CrossReference], 
                          unique_sources: int, source_diversity: float,
                          verified_claims: List[str], 
                          contradicted_claims: List[str]) -> str:
        """Generate human-readable reasoning"""
        parts = []
        
        if len(references) >= 3:
            parts.append(f"Found {len(references)} corroborating references from {unique_sources} independent sources")
        elif len(references) > 0:
            parts.append(f"Found {len(references)} reference(s) with moderate corroboration")
        
        if source_diversity > 0.7:
            parts.append("High source diversity increases confidence")
        elif source_diversity < 0.3:
            parts.append("Low source diversity - most references from similar sources")
        
        if verified_claims:
            parts.append(f"{len(verified_claims)} claims independently verified")
        
        if contradicted_claims:
            parts.append(f"️ {len(contradicted_claims)} claims contradicted by other sources")
        
        return ". ".join(parts) + "." if parts else "Insufficient data for verification"


# Singleton
_verification_engine = None

def get_verification_engine() -> AdvancedCrossVerificationEngine:
    """Get singleton verification engine"""
    global _verification_engine
    if _verification_engine is None:
        _verification_engine = AdvancedCrossVerificationEngine()
    return _verification_engine