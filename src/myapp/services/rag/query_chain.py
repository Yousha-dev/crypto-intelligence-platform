"""
RAG Query Chains
Multi-step query processing for complex questions
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from django.utils import timezone
from enum import Enum
from dataclasses import field

logger = logging.getLogger(__name__)


class ChainStepType(Enum):
    """Types of chain steps"""
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    ANALYZE = "analyze"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    FILTER = "filter"


@dataclass
class ChainStep:
    """A single step in the query chain"""
    step_type: ChainStepType
    input_data: Any
    output_data: Any = None
    params: Dict[str, Any] = None
    duration_ms: float = 0
    success: bool = False


@dataclass 
class ChainResult:
    """Result of chain execution"""
    final_answer: str
    steps: List[ChainStep]
    total_duration_ms: float
    success: bool
    metadata: Dict[str, Any]


class RAGQueryChain:
    """
    Multi-step query chain for complex RAG queries
    """
    
    def __init__(self, rag_engine=None):
        self.rag_engine = rag_engine
        self.steps: List[ChainStep] = []
    
    def _get_rag_engine(self):
        """Get RAG engine lazily"""
        if self.rag_engine is None:
            from myapp.services.rag.rag_service import get_rag_engine
            self.rag_engine = get_rag_engine()
        return self.rag_engine
      
    def decompose_query(self, query: str) -> List[str]:
        """ 
        Decompose complex query into sub-queries
        
        Uses LLM to break down complex questions
        """
        rag_engine = self._get_rag_engine()
        
        if not rag_engine.llm:
            # Fallback: simple decomposition
            return [query]
        
        decompose_prompt = f"""Analyze this question and break it into simpler sub-questions if needed.
    If the question is already simple, return just the original question.
    Return each sub-question on a new line.

    Question: {query}

    Sub-questions:"""
        
        try:
            # ✅ Special handling for Ollama using raw HTTP
            if rag_engine.llm_provider == 'ollama':
                import requests
                
                messages = [
                    {
                        "role": "system",
                        "content": "You decompose complex questions into simpler sub-questions. Be concise."
                    },
                    {"role": "user", "content": decompose_prompt}
                ]
                
                response = requests.post(
                    f"{rag_engine.ollama_base_url}/api/chat",
                    json={
                        "model": rag_engine.llm_model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 500
                        }
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('message', {}).get('content', '')
                    
                    # Parse sub-questions
                    sub_queries = [
                        line.strip().lstrip('0123456789.-) ').strip()
                        for line in content.split('\n')
                        if line.strip() and len(line.strip()) > 5
                    ]
                    
                    return sub_queries if sub_queries else [query]
                else:
                    logger.error(f"Ollama error: {response.text}")
                    return [query]
            
            # For other providers, use LlamaIndex
            from llama_index.core.llms import ChatMessage, MessageRole
            
            messages = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="You decompose complex questions into simpler sub-questions. Be concise."
                ),
                ChatMessage(role=MessageRole.USER, content=decompose_prompt)
            ]
            
            response = rag_engine.llm.chat(messages)
            
            # Parse sub-questions
            sub_queries = [
                line.strip().lstrip('0123456789.-) ').strip()
                for line in response.message.content.split('\n')
                if line.strip() and len(line.strip()) > 5
            ]
            
            return sub_queries if sub_queries else [query]
            
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            return [query]

    
    def execute_chain(self, query: str,
                      chain_type: str = 'auto') -> ChainResult:
        """
        Execute a query chain
        
        Args:
            query: User query
            chain_type: 'auto', 'simple', 'deep_analysis', 'comparison'
            
        Returns:
            ChainResult with final answer and step details
        """
        start_time = datetime.now()
        self.steps = []
        
        try:
            if chain_type == 'auto':
                chain_type = self._determine_chain_type(query)
            
            if chain_type == 'simple':
                result = self._execute_simple_chain(query)
            elif chain_type == 'deep_analysis':
                result = self._execute_deep_analysis_chain(query)
            elif chain_type == 'comparison':
                result = self._execute_comparison_chain(query)
            else:
                result = self._execute_simple_chain(query)
            
            total_duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return ChainResult(
                final_answer=result,
                steps=self.steps,
                total_duration_ms=total_duration,
                success=True,
                metadata={
                    'chain_type': chain_type,
                    'step_count': len(self.steps),
                    'query': query
                }
            )
            
        except Exception as e:
            logger.error(f"Chain execution error: {e}")
            total_duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return ChainResult(
                final_answer=f"Error processing query: {str(e)}",
                steps=self.steps,
                total_duration_ms=total_duration,
                success=False,
                metadata={'error': str(e)}
            )
    
    def _determine_chain_type(self, query: str) -> str:
        """Determine appropriate chain type based on query"""
        query_lower = query.lower()
        
        # Comparison indicators
        comparison_words = ['compare', 'versus', 'vs', 'difference', 'better', 'which']
        if any(word in query_lower for word in comparison_words):
            return 'comparison'
        
        # Deep analysis indicators
        analysis_words = ['analyze', 'explain why', 'detailed', 'comprehensive', 'impact']
        if any(word in query_lower for word in analysis_words):
            return 'deep_analysis'
        
        return 'simple'
    
    def _execute_simple_chain(self, query: str) -> str:
        """Execute simple retrieve-generate chain"""
        rag_engine = self._get_rag_engine()
        
        # Step 1: Retrieve
        step_start = datetime.now()
        results = rag_engine.retrieve(query, top_k=10)
        
        self.steps.append(ChainStep(
            step_type=ChainStepType.RETRIEVE,
            input_data=query,
            output_data={'count': len(results)},
            duration_ms=(datetime.now() - step_start).total_seconds() * 1000,
            success=True
        ))
        
        # Step 2: Generate
        step_start = datetime.now()
        response = rag_engine.generate_answer(query, context_results=results)
        
        self.steps.append(ChainStep(
            step_type=ChainStepType.GENERATE,
            input_data={'query': query, 'context_count': len(results)},
            output_data={'tokens': response.tokens_used},
            duration_ms=(datetime.now() - step_start).total_seconds() * 1000,
            success=True
        ))
        
        return response.answer
    
    def _execute_deep_analysis_chain(self, query: str) -> str:
        """Execute deep analysis chain with multiple retrieval steps"""
        rag_engine = self._get_rag_engine()
        
        # Step 1: Decompose query
        step_start = datetime.now()
        sub_queries = self.decompose_query(query)
        
        self.steps.append(ChainStep(
            step_type=ChainStepType.ANALYZE,
            input_data=query,
            output_data={'sub_queries': sub_queries},
            duration_ms=(datetime.now() - step_start).total_seconds() * 1000,
            success=True
        ))
        
        # Step 2: Retrieve for each sub-query
        all_results = []
        for sub_query in sub_queries[:3]:  # Limit to 3 sub-queries
            step_start = datetime.now()
            results = rag_engine.retrieve(sub_query, top_k=5)
            all_results.extend(results)
            
            self.steps.append(ChainStep(
                step_type=ChainStepType.RETRIEVE,
                input_data=sub_query,
                output_data={'count': len(results)},
                duration_ms=(datetime.now() - step_start).total_seconds() * 1000,
                success=True
            ))
        
        # Step 3: Deduplicate results
        unique_results = self._deduplicate_results(all_results)[:10]
        
        # Step 4: Generate comprehensive answer
        step_start = datetime.now()
        
        system_prompt = """You are providing a deep analysis. Structure your answer with:
1. Executive Summary
2. Key Findings (from each aspect of the question)
3. Analysis
4. Conclusion

Always cite sources using [Source X] notation."""
        
        response = rag_engine.generate_answer(
            query,
            context_results=unique_results,
            system_prompt=system_prompt
        )
        
        self.steps.append(ChainStep(
            step_type=ChainStepType.GENERATE,
            input_data={'query': query, 'context_count': len(unique_results)},
            output_data={'tokens': response.tokens_used},
            duration_ms=(datetime.now() - step_start).total_seconds() * 1000,
            success=True
        ))
        
        return response.answer
    
    def _execute_comparison_chain(self, query: str) -> str:
        """Execute comparison chain for comparing entities/topics"""
        rag_engine = self._get_rag_engine()
        
        # Step 1: Extract comparison entities
        step_start = datetime.now()
        entities = self._extract_comparison_entities(query)
        
        self.steps.append(ChainStep(
            step_type=ChainStepType.ANALYZE,
            input_data=query,
            output_data={'entities': entities},
            duration_ms=(datetime.now() - step_start).total_seconds() * 1000,
            success=True
        )) 
        
        # Step 2: Retrieve for each entity
        entity_results = {}
        for entity in entities[:2]:  # Compare up to 2 entities
            step_start = datetime.now()
            results = rag_engine.retrieve(entity, top_k=7)
            entity_results[entity] = results
            
            self.steps.append(ChainStep(
                step_type=ChainStepType.RETRIEVE,
                input_data=entity,
                output_data={'count': len(results)},
                duration_ms=(datetime.now() - step_start).total_seconds() * 1000,
                success=True
            ))
        
        # Step 3: Generate comparison
        step_start = datetime.now()
        
        # Combine results
        all_results = []
        for results in entity_results.values():
            all_results.extend(results)
        unique_results = self._deduplicate_results(all_results)[:10]
        
        system_prompt = f"""You are comparing {' vs '.join(entities)}. Structure your comparison:
1. Overview of each
2. Key similarities
3. Key differences  
4. Pros and cons of each
5. Conclusion

Always cite sources using [Source X] notation."""
        
        response = rag_engine.generate_answer(
            query,
            context_results=unique_results,
            system_prompt=system_prompt
        )
        
        self.steps.append(ChainStep(
            step_type=ChainStepType.COMPARE,
            input_data={'entities': entities, 'context_count': len(unique_results)},
            output_data={'tokens': response.tokens_used},
            duration_ms=(datetime.now() - step_start).total_seconds() * 1000,
            success=True
        ))
        
        return response.answer
    
    def _extract_comparison_entities(self, query: str) -> List[str]:
        """Extract entities being compared"""
        # Simple extraction - look for common patterns
        query_lower = query.lower()
        
        # Pattern: "X vs Y", "X versus Y", "compare X and Y"
        import re
        
        patterns = [
            r'compare\s+(\w+)\s+(?:and|with|to)\s+(\w+)',
            r'(\w+)\s+(?:vs|versus|vs\.)\s+(\w+)',
            r'difference\s+between\s+(\w+)\s+and\s+(\w+)',
            r'(\w+)\s+or\s+(\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return list(match.groups())
        
        # Fallback: look for crypto names
        crypto_names = ['bitcoin', 'ethereum', 'solana', 'cardano', 'ripple', 'dogecoin']
        found = [name for name in crypto_names if name in query_lower]
        
        return found if found else [query.split()[0]]
    
    def _deduplicate_results(self, results: List[Any]) -> List[Any]:
        """Remove duplicate results based on ID"""
        seen_ids = set()
        unique = []
        
        for result in results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique.append(result)
        
        # Sort by score
        unique.sort(key=lambda x: x.score, reverse=True)
        return unique


def get_query_chain() -> RAGQueryChain:
    """Get a new query chain instance"""
    return RAGQueryChain()