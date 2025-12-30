"""
RAG Context Manager
Manages query history, session context, and in-context learning
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from django.utils import timezone
from collections import deque
import hashlib
import json

from django.core.cache import cache

logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Context for a single query"""
    query: str
    answer: str
    sources: List[Dict]
    timestamp: datetime
    feedback: Optional[str] = None
    rating: Optional[int] = None


@dataclass
class SessionContext:
    """Session context for in-context learning"""
    session_id: str
    user_id: Optional[str] = None
    queries: List[QueryContext] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: timezone.now())
    last_activity: datetime = field(default_factory=lambda: timezone.now())
    
    def add_query(self, query: str, answer: str, sources: List[Dict]):
        """Add a query to session history"""
        self.queries.append(QueryContext(
            query=query,
            answer=answer,
            sources=sources,
            timestamp=timezone.now()
        ))
        self.last_activity = timezone.now()
        
        # Keep only last 10 queries per session
        if len(self.queries) > 10:
            self.queries = self.queries[-10:]
    
    def get_recent_context(self, n: int = 3) -> str:
        """Get recent query context for in-context learning"""
        if not self.queries:
            return ""
        
        context_parts = ["Previous conversation:"]
        for q in self.queries[-n:]:
            context_parts.append(f"Q: {q.query[:200]}")
            context_parts.append(f"A: {q.answer[:300]}...")
        
        return "\n".join(context_parts)
    
    def add_feedback(self, query_index: int, feedback: str, rating: int):
        """Add feedback to a specific query"""
        if 0 <= query_index < len(self.queries):
            self.queries[query_index].feedback = feedback
            self.queries[query_index].rating = rating


class RAGContextManager:
    """
    Manages context for RAG queries
    Enables in-context learning and session-based interactions
    """
    
    def __init__(self, cache_ttl: int = 3600):
        self.cache_ttl = cache_ttl  # 1 hour default
        self.sessions: Dict[str, SessionContext] = {}
        
        # Global query cache for frequently asked questions
        self.query_cache_prefix = "rag_query_"
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
    
    def get_or_create_session(self, session_id: str,
                              user_id: Optional[str] = None) -> SessionContext:
        """Get existing session or create new one"""
        # Try cache first
        cache_key = f"rag_session_{session_id}"
        cached = cache.get(cache_key)
        
        if cached:
            return self._deserialize_session(cached)
        
        # Create new session
        session = SessionContext(
            session_id=session_id,
            user_id=user_id
        )
        
        self.sessions[session_id] = session
        self._cache_session(session)
        
        return session
    
    def update_session(self, session_id: str, query: str,
                       answer: str, sources: List[Dict]):
        """Update session with new query"""
        session = self.get_or_create_session(session_id)
        session.add_query(query, answer, sources)
        self._cache_session(session)
    
    def get_context_prompt(self, session_id: str, n_context: int = 3) -> str:
        """Get context prompt for in-context learning"""
        session = self.get_or_create_session(session_id)
        return session.get_recent_context(n_context)
    
    def cache_query_result(self, query: str, result: Dict,
                           ttl: Optional[int] = None):
        """Cache a query result for future use"""
        query_hash = self._hash_query(query)
        cache_key = f"{self.query_cache_prefix}{query_hash}"
        
        cache.set(cache_key, json.dumps(result), ttl or self.cache_ttl)
    
    def get_cached_result(self, query: str) -> Optional[Dict]:
        """Get cached result for a query"""
        query_hash = self._hash_query(query)
        cache_key = f"{self.query_cache_prefix}{query_hash}"
        
        cached = cache.get(cache_key)
        if cached:
            return json.loads(cached)
        return None
    
    def record_performance(self, query: str, latency: float,
                           tokens_used: int, success: bool,
                           feedback: Optional[str] = None):
        """Record query performance for analysis"""
        self.performance_history.append({
            'query_hash': self._hash_query(query)[:8],
            'latency': latency,
            'tokens_used': tokens_used,
            'success': success,
            'feedback': feedback,
            'timestamp': timezone.now().isoformat()
        })
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.performance_history:
            return {'message': 'No performance data'}
        
        history = list(self.performance_history)
        successful = [h for h in history if h['success']]
        
        return {
            'total_queries': len(history),
            'success_rate': len(successful) / len(history) if history else 0,
            'avg_latency': sum(h['latency'] for h in successful) / len(successful) if successful else 0,
            'avg_tokens': sum(h['tokens_used'] for h in successful) / len(successful) if successful else 0,
            'recent_queries': len([h for h in history if 
                datetime.fromisoformat(h['timestamp']) > timezone.now() - timedelta(hours=1)])
        }
    
    def build_adaptive_prompt(self, query: str, session_id: str,
                              base_system_prompt: str) -> str:
        """Build adaptive system prompt based on session context"""
        session = self.get_or_create_session(session_id)
        
        # Start with base prompt
        prompt_parts = [base_system_prompt]
        
        # Add session context
        if session.queries:
            context = session.get_recent_context(3)
            prompt_parts.append(f"\n\n{context}")
        
        # Add user preferences if any
        if session.preferences:
            prefs = session.preferences
            if prefs.get('preferred_detail_level'):
                prompt_parts.append(f"\nUser prefers {prefs['preferred_detail_level']} level of detail.")
            if prefs.get('focus_areas'):
                prompt_parts.append(f"\nUser is particularly interested in: {', '.join(prefs['focus_areas'])}")
        
        # Add feedback-based adjustments
        recent_feedback = [
            q.feedback for q in session.queries[-5:]
            if q.feedback
        ]
        if recent_feedback:
            prompt_parts.append(f"\nPrevious feedback: {'; '.join(recent_feedback[-2:])}")
        
        return "\n".join(prompt_parts)
    
    def _hash_query(self, query: str) -> str:
        """Create consistent hash for query"""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _cache_session(self, session: SessionContext):
        """Cache session to Redis/cache backend"""
        cache_key = f"rag_session_{session.session_id}"
        cache.set(cache_key, self._serialize_session(session), self.cache_ttl)
    
    def _serialize_session(self, session: SessionContext) -> str:
        """Serialize session for caching"""
        data = {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'queries': [
                {
                    'query': q.query,
                    'answer': q.answer[:500],  # Truncate for cache
                    'sources': q.sources[:3],
                    'timestamp': q.timestamp.isoformat(),
                    'feedback': q.feedback,
                    'rating': q.rating
                } 
                for q in session.queries
            ],   
            'preferences': session.preferences,
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat()
        }
        return json.dumps(data)
    
    def _deserialize_session(self, data: str) -> SessionContext:
        """Deserialize session from cache"""
        parsed = json.loads(data)
        
        session = SessionContext(
            session_id=parsed['session_id'],
            user_id=parsed.get('user_id'),
            preferences=parsed.get('preferences', {}),
            created_at=datetime.fromisoformat(parsed['created_at']),
            last_activity=datetime.fromisoformat(parsed['last_activity'])
        )
        
        for q_data in parsed.get('queries', []):
            session.queries.append(QueryContext(
                query=q_data['query'],
                answer=q_data['answer'],
                sources=q_data['sources'],
                timestamp=datetime.fromisoformat(q_data['timestamp']),
                feedback=q_data.get('feedback'),
                rating=q_data.get('rating')
            ))
        
        return session


# Singleton instance
_context_manager_instance = None


def get_context_manager() -> RAGContextManager:
    """Get singleton context manager instance"""
    global _context_manager_instance
    
    if _context_manager_instance is None:
        _context_manager_instance = RAGContextManager()
    
    return _context_manager_instance