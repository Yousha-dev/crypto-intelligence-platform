"""
RAG Post-processing and Re-ranking
Advanced result refinement and output formatting
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from django.utils import timezone

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """Re-ranked search result"""
    id: str
    content: str
    metadata: Dict[str, Any]
    original_score: float
    reranked_score: float
    final_rank: int


class RAGPostProcessor:
    """
    Post-processing pipeline for RAG results
    Includes re-ranking, filtering, and output formatting
    """
    
    def __init__(self):
        self.cross_encoder = None
        self._initialize_cross_encoder()
    
    def _initialize_cross_encoder(self):
        """Initialize cross-encoder for re-ranking"""
        try:
            from sentence_transformers import CrossEncoder
            
            self.cross_encoder = CrossEncoder(
                'cross-encoder/ms-marco-MiniLM-L-6-v2',
                max_length=512
            )
            logger.info("Cross-encoder initialized for re-ranking")
            
        except ImportError:
            logger.warning("Cross-encoder not available, using score-based ranking")
            self.cross_encoder = None
        except Exception as e:
            logger.warning(f"Error initializing cross-encoder: {e}")
            self.cross_encoder = None
    
    def rerank_results(self, query: str, results: List[Any],
                       top_k: int = 10) -> List[RankedResult]:
        """
        Re-rank results using cross-encoder
        
        Args: 
            query: Original query
            results: List of SearchResult objects
            top_k: Number of results to return
               
        Returns:
            Re-ranked results
        """
        if not results:
            return []
        
        if self.cross_encoder:
            return self._rerank_with_cross_encoder(query, results, top_k)
        else:
            return self._rerank_with_heuristics(query, results, top_k)
    
    def _rerank_with_cross_encoder(self, query: str, results: List[Any],
                                    top_k: int) -> List[RankedResult]:
        """Re-rank using cross-encoder model"""
        # Prepare pairs for cross-encoder
        pairs = [(query, r.content[:512]) for r in results]
        
        # Get cross-encoder scores
        try:
            scores = self.cross_encoder.predict(pairs)
        except Exception as e:
            logger.error(f"Cross-encoder error: {e}")
            return self._rerank_with_heuristics(query, results, top_k)
        
        # Combine with original scores
        ranked_results = []
        for i, (result, ce_score) in enumerate(zip(results, scores)):
            # Weighted combination: 60% cross-encoder, 40% original
            combined_score = 0.6 * float(ce_score) + 0.4 * result.score
            
            ranked_results.append(RankedResult(
                id=result.id,
                content=result.content,
                metadata=result.metadata,
                original_score=result.score,
                reranked_score=combined_score,
                final_rank=0  # Will be set after sorting
            ))
        
        # Sort by combined score
        ranked_results.sort(key=lambda x: x.reranked_score, reverse=True)
        
        # Assign final ranks
        for i, result in enumerate(ranked_results[:top_k]):
            result.final_rank = i + 1
        
        return ranked_results[:top_k]
    
    def _rerank_with_heuristics(self, query: str, results: List[Any],
                                 top_k: int) -> List[RankedResult]:
        """Re-rank using heuristic scoring"""
        ranked_results = []
        query_terms = set(query.lower().split())
        
        for result in results:
            # Base score
            score = result.score
            
            # Boost for trust score
            trust_score = result.metadata.get('trust_score', 5)
            score += (trust_score - 5) * 0.05  # ±0.25 adjustment
            
            # Boost for title match
            title = result.metadata.get('title', '').lower()
            title_terms = set(title.split())
            title_overlap = len(query_terms & title_terms) / max(len(query_terms), 1)
            score += title_overlap * 0.1
            
            # Boost for recency (if date available)
            # Could add date-based boosting here
            
            # Penalty for very short content
            if len(result.content) < 100:
                score *= 0.9
            
            ranked_results.append(RankedResult(
                id=result.id,
                content=result.content,
                metadata=result.metadata,
                original_score=result.score,
                reranked_score=score,
                final_rank=0
            ))
        
        ranked_results.sort(key=lambda x: x.reranked_score, reverse=True)
        
        for i, result in enumerate(ranked_results[:top_k]):
            result.final_rank = i + 1
        
        return ranked_results[:top_k]
    
    def filter_results(self, results: List[RankedResult],
                       filters: Dict[str, Any]) -> List[RankedResult]:
        """Apply additional filters to re-ranked results"""
        filtered = []
        
        min_trust = filters.get('min_trust_score', 0)
        allowed_types = filters.get('types', None)
        allowed_sources = filters.get('sources', None)
        min_score = filters.get('min_score', 0)
        
        for result in results:
            # Trust score filter
            if result.metadata.get('trust_score', 0) < min_trust:
                continue
            
            # Type filter
            if allowed_types and result.metadata.get('type') not in allowed_types:
                continue
            
            # Source filter
            if allowed_sources and result.metadata.get('source') not in allowed_sources:
                continue
            
            # Score filter
            if result.reranked_score < min_score:
                continue
            
            filtered.append(result)
        
        # Re-assign ranks after filtering
        for i, result in enumerate(filtered):
            result.final_rank = i + 1
        
        return filtered
    
    def extract_key_insights(self, answer: str) -> Dict[str, Any]:
        """Extract structured insights from generated answer"""
        insights = {
            'key_points': [],
            'entities_mentioned': [],
            'sentiment_indicators': [],
            'action_items': [],
            'confidence_level': 'medium'
        }
        
        lines = answer.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Extract bullet points as key points
            if line.startswith(('- ', '• ', '* ', '1.', '2.', '3.')):
                clean_point = line.lstrip('-•* 0123456789.').strip()
                if clean_point:
                    insights['key_points'].append(clean_point)
            
            # Detect sentiment indicators
            sentiment_words = {
                'bullish': ['bullish', 'surge', 'rally', 'growth', 'optimistic', 'positive'],
                'bearish': ['bearish', 'crash', 'decline', 'drop', 'pessimistic', 'negative'],
                'neutral': ['stable', 'unchanged', 'mixed', 'uncertain']
            }
            
            line_lower = line.lower()
            for sentiment, words in sentiment_words.items():
                if any(word in line_lower for word in words):
                    insights['sentiment_indicators'].append({
                        'text': line[:100],
                        'sentiment': sentiment
                    })
                    break
        
        # Determine overall confidence based on citation count
        citation_count = answer.count('[Source')
        if citation_count >= 3:
            insights['confidence_level'] = 'high'
        elif citation_count >= 1:
            insights['confidence_level'] = 'medium'
        else:
            insights['confidence_level'] = 'low'
        
        return insights
    
    def format_output(self, answer: str, sources: List[Dict],
                      format_type: str = 'detailed') -> Dict[str, Any]:
        """Format final output for display"""
        if format_type == 'brief':
            return self._format_brief(answer, sources)
        elif format_type == 'structured':
            return self._format_structured(answer, sources)
        else:
            return self._format_detailed(answer, sources)
    
    def _format_detailed(self, answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """Detailed format with full answer and sources"""
        insights = self.extract_key_insights(answer)
        
        return {
            'answer': answer,
            'sources': sources,
            'insights': insights,
            'metadata': {
                'format': 'detailed',
                'source_count': len(sources),
                'generated_at': timezone.now().isoformat()
            }
        }
    
    def _format_brief(self, answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """Brief format with summary only"""
        # Extract first paragraph or first 500 chars
        paragraphs = answer.split('\n\n')
        summary = paragraphs[0][:500] if paragraphs else answer[:500]
        
        return {
            'summary': summary,
            'source_count': len(sources),
            'top_sources': [s['title'] for s in sources[:3]],
            'metadata': {
                'format': 'brief',
                'generated_at': timezone.now().isoformat()
            }
        }
    
    def _format_structured(self, answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """Structured format with sections"""
        insights = self.extract_key_insights(answer)
        
        return {
            'executive_summary': answer.split('\n\n')[0] if answer else '',
            'key_points': insights['key_points'],
            'sentiment': insights['sentiment_indicators'],
            'confidence': insights['confidence_level'],
            'sources': [
                {
                    'title': s['title'],
                    'source': s['source'],
                    'trust_score': s.get('trust_score', 0),
                    'relevance': s.get('relevance_score', 0)
                }
                for s in sources
            ],
            'metadata': {
                'format': 'structured',
                'generated_at': timezone.now().isoformat()
            }
        }


# Singleton instance
_postprocessor_instance = None


def get_postprocessor() -> RAGPostProcessor:
    """Get singleton post-processor instance"""
    global _postprocessor_instance
    
    if _postprocessor_instance is None:
        _postprocessor_instance = RAGPostProcessor()
    
    return _postprocessor_instance