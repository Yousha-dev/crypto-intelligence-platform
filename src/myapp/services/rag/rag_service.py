"""
RAG (Retrieval-Augmented Generation) Service using LlamaIndex
Vector embeddings, semantic search, and LLM-powered analysis
"""

import os
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from django.utils import timezone
from datetime import timezone as timezoneDt
from enum import Enum

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
    load_index_from_storage,
    ServiceContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.indices.base import BaseIndex

# LlamaIndex Vector Store (FAISS)
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# LlamaIndex Embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# LlamaIndex LLMs
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.llms.anthropic import Anthropic as LlamaAnthropic
from llama_index.llms.ollama import Ollama as LlamaOllama
from llama_index.llms.groq import Groq as LlamaGroq

# Django
from django.conf import settings as django_settings
from django.core.cache import cache

logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """Supported embedding models"""
    MINILM = "sentence-transformers/all-MiniLM-L6-v2"
    MPNET = "sentence-transformers/all-mpnet-base-v2"
    BGE_SMALL = "BAAI/bge-small-en-v1.5"
    BGE_BASE = "BAAI/bge-base-en-v1.5"


class LLMProviderType(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GROQ = "groq"


@dataclass
class RAGResponse:
    """Response from RAG system"""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    tokens_used: int
    processing_time: float
    model_used: str


@dataclass
class SearchResult:
    """Search result with relevance score"""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    rank: int


class LlamaIndexRAGEngine:
    """
    LlamaIndex-based RAG Engine for cryptocurrency news analysis
    """
    
    def __init__(self,
                 embedding_model: str = EmbeddingModel.MPNET.value,
                 llm_provider: str = "ollama",
                 llm_model: str = "llama3.1",
                 persist_dir: Optional[str] = None):
        """
        Initialize LlamaIndex RAG Engine
        
        Args:
            embedding_model: HuggingFace embedding model name
            llm_provider: LLM provider (openai, anthropic, ollama, groq)
            llm_model: Model name for the LLM
            persist_dir: Directory to persist index
        """
        self.embedding_model_name = embedding_model
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.persist_dir = persist_dir or os.path.join(
            django_settings.BASE_DIR, 'data', 'llamaindex_store'
        )
        
        # Configuration
        self.config = {
            'chunk_size': 512,
            'chunk_overlap': 50,
            'top_k_retrieval': 10,
            'similarity_cutoff': 0.5,
            'max_tokens': 1000,
            'temperature': 0.3
        }
        
        # Lazy-loaded content services
        self._text_processor = None
        self._sentiment_analyzer = None
        self._hashtag_analyzer = None
        
         # Initialize components
        self._initialize_embedding_model()
        self._initialize_llm()
        self._initialize_index()
        
        logger.info(f"LlamaIndex RAG Engine initialized with {llm_provider}/{llm_model}")
    
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
    
    @property
    def hashtag_analyzer(self):
        """Lazy load hashtag analyzer service"""
        if self._hashtag_analyzer is None:
            from myapp.services.content.hashtag_analyzer import get_hashtag_analyzer
            self._hashtag_analyzer = get_hashtag_analyzer()
        return self._hashtag_analyzer
    
    
    def _initialize_embedding_model(self):
        """Initialize HuggingFace embedding model"""
        try:
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.embedding_model_name,
                trust_remote_code=True
            )
            
            # Get embedding dimension
            test_embedding = self.embed_model.get_text_embedding("test")
            self.embedding_dim = len(test_embedding)
            
            # Set as default embedding model
            Settings.embed_model = self.embed_model
            
            logger.info(f"Initialized embedding model: {self.embedding_model_name} (dim={self.embedding_dim})")
            
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
    
    def _initialize_llm(self):
        """Initialize LLM based on provider"""
        try:
            if self.llm_provider == LLMProviderType.OPENAI.value:
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    logger.warning("OpenAI API key not found")
                    self.llm = None
                    return
                
                self.llm = LlamaOpenAI(
                    model=self.llm_model,
                    api_key=api_key,
                    temperature=self.config['temperature'],
                    max_tokens=self.config['max_tokens']
                )
                Settings.llm = self.llm
            
            elif self.llm_provider == LLMProviderType.ANTHROPIC.value:
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    logger.warning("Anthropic API key not found")
                    self.llm = None
                    return
                
                self.llm = LlamaAnthropic(
                    model=self.llm_model,
                    api_key=api_key,
                    temperature=self.config['temperature'],
                    max_tokens=self.config['max_tokens']
                )
                Settings.llm = self.llm
            
            elif self.llm_provider == LLMProviderType.OLLAMA.value:
                # ✅ Use raw HTTP client for Ollama (avoid memory check issues)
                import requests
                
                self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
                
                try:
                    # Test connection
                    response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        self.llm = True  # Just a flag
                        logger.info(f"Initialized LLM: ollama/{self.llm_model} at {self.ollama_base_url}")
                    else:
                        self.llm = None
                except Exception as e:
                    logger.warning(f"Ollama not available: {e}")
                    self.llm = None
                
                # Don't set Settings.llm for Ollama (we'll use HTTP directly)
                return
            
            elif self.llm_provider == LLMProviderType.GROQ.value:
                api_key = os.getenv('GROQ_API_KEY')
                if not api_key:
                    logger.warning("Groq API key not found")
                    self.llm = None
                    return
                
                self.llm = LlamaGroq(
                    model=self.llm_model,
                    api_key=api_key,
                    temperature=self.config['temperature'],
                    max_tokens=self.config['max_tokens']
                )
                Settings.llm = self.llm
            
            else:
                logger.warning(f"Unknown LLM provider: {self.llm_provider}")
                self.llm = None
                return
            
            if self.llm and self.llm_provider != LLMProviderType.OLLAMA.value:
                logger.info(f"Initialized LLM: {self.llm_provider}/{self.llm_model}")
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            self.llm = None
    
    def _initialize_index(self):
        """Initialize or load FAISS vector index"""
        try:
            # Check if existing index exists
            if os.path.exists(os.path.join(self.persist_dir, 'docstore.json')):
                logger.info("Loading existing index from storage...")
                
                # Load FAISS index
                faiss_index = faiss.read_index(
                    os.path.join(self.persist_dir, 'faiss.index')
                )
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=self.persist_dir
                )
                
                self.index = load_index_from_storage(storage_context)
                logger.info(f"Loaded index with {len(self.index.docstore.docs)} documents")
            
            else:
                logger.info("Creating new FAISS index...")
                
                # Create new FAISS index
                faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )
                
                # Create empty index
                self.index = VectorStoreIndex(
                    nodes=[],
                    storage_context=storage_context,
                    embed_model=self.embed_model
                )
                
                logger.info("Created new empty index")
            
            # Initialize node parser
            self.node_parser = SentenceSplitter(
                chunk_size=self.config['chunk_size'],
                chunk_overlap=self.config['chunk_overlap']
            )
            
            # Store reference to storage context
            self.storage_context = self.index.storage_context
            
        except Exception as e:
            logger.error(f"Error initializing index: {e}")
            raise
    
    def _create_document_from_article(self, article: Dict) -> Document:
        """
        Create LlamaIndex Document from news article (RAW data format)
        Uses dedicated content services for text processing and analysis
        """
        # Build content from RAW fields
        content_parts = []
        
        title = article.get('title', '')
        if title:
            content_parts.append(f"Title: {title}")
        
        # RAW content fields - fetchers use different field names
        description = (
            article.get('description') or 
            article.get('content') or 
            article.get('body') or
            article.get('summary') or
            ''
        )
        if description:
            content_parts.append(f"Content: {description[:2000]}")
        
        # RAW source info (fetchers return dict with title/domain)
        source = article.get('source', {})
        if isinstance(source, dict):
            source_name = source.get('title') or source.get('name') or 'Unknown'
        else:
            source_name = str(source) if source else 'Unknown'
        content_parts.append(f"Source: {source_name}")
        
        # Build full text for processing
        full_text = f"{title} {description}"
        
        # Use TextProcessor for entity extraction
        cryptos = []
        organizations = []
        persons = []
        
        if full_text.strip():
            # Check if entities already extracted (by integrator)
            extracted = article.get('extracted_entities', {})
            
            if extracted:
                # Use pre-extracted entities
                cryptos = extracted.get('cryptocurrencies', [])
                organizations = extracted.get('organizations', [])
                persons = extracted.get('persons', [])
            else:
                # Extract using TextProcessor
                entities = self.text_processor.extract_entities(full_text)
                cryptos = entities.cryptocurrencies
                organizations = entities.organizations
                persons = entities.persons
        
        # Also check RAW instruments field (CryptoPanic)
        instruments = article.get('instruments', [])
        if instruments:
            for inst in instruments:
                if isinstance(inst, dict):
                    code = inst.get('code', '')
                    if code and code not in cryptos:
                        cryptos.append(code)
        
        # Also check RAW tags (CryptoCompare)
        tags = article.get('tags', '')
        if isinstance(tags, str) and tags:
            for tag in tags.split('|'):
                tag_upper = tag.strip().upper()
                if tag_upper and tag_upper not in cryptos:
                    cryptos.append(tag_upper)
        
        if cryptos:
            content_parts.append(f"Cryptocurrencies: {', '.join(cryptos[:10])}")
        
        # Get sentiment - check if already analyzed, otherwise use SentimentAnalyzer
        sentiment_label = 'unknown'
        sentiment_score = 0.0
        
        # Priority 1: Use sentiment_analysis.label (CORRECT FIELD)
        sentiment_data = article.get('sentiment_analysis', {})
        if sentiment_data and isinstance(sentiment_data, dict):
            sentiment_label = sentiment_data.get('label', 'unknown')
            sentiment_score = sentiment_data.get('score', 0.0)
        
        # Priority 2: Fallback to analyzing text (WITH NONE CHECK!)
        elif full_text.strip():
            try:
                sentiment_result = self.sentiment_analyzer.analyze(full_text)
                
                # ✅ FIX: Check if result is None before accessing attributes
                if sentiment_result is not None:
                    sentiment_label = sentiment_result.label.value
                    sentiment_score = sentiment_result.score
                else:
                    # analyze() returned None (insufficient content)
                    sentiment_label = 'unknown'
                    sentiment_score = 0.0
            except Exception as e:
                logger.warning(f"Error analyzing sentiment: {e}")
                sentiment_label = 'unknown'
                sentiment_score = 0.0
        
        if sentiment_label != 'unknown':
            content_parts.append(f"Sentiment: {sentiment_label}")

        
        content = "\n".join(content_parts)
        
        # Build metadata from RAW fields
        doc_id = f"news_{article.get('id', article.get('source_id', ''))}"
        
        metadata = {
            'type': 'news',
            'doc_id': doc_id,
            'title': title,
            'source': source_name,
            'platform': article.get('platform', 'unknown'),
            'published_at': str(article.get('published_at', '')),
            'trust_score': float(article.get('trust_score', 0)),
            'url': article.get('url', ''),
            'sentiment': sentiment_label,
            'sentiment_score': sentiment_score,
            'cryptocurrencies': ','.join(cryptos[:5]) if cryptos else '',
            'organizations': ','.join(organizations[:5]) if organizations else '',
            'persons': ','.join(persons[:5]) if persons else '',
        }
        
        return Document(
            text=content,
            metadata=metadata,
            doc_id=doc_id
        )
    
    def _create_document_from_social(self, post: Dict) -> Document:
        """
        Create LlamaIndex Document from social media post (RAW data format)
        Uses dedicated content services for text processing and analysis
        """
        content_parts = []
        
        title = post.get('title', '')
        if title:
            content_parts.append(f"Title: {title}")
        
        # RAW content - different platforms use different field names
        text = (
            post.get('content') or 
            post.get('text') or 
            post.get('selftext') or  # Reddit
            post.get('description') or  # YouTube
            ''
        )
        if text:
            content_parts.append(f"Content: {text[:1500]}")
        
        platform = post.get('platform', 'Unknown')
        content_parts.append(f"Platform: {platform}")
        
        # RAW author info - extract from nested objects if needed
        author = post.get('author') or post.get('username') or post.get('channel_title', '')
        if not author:
            user_info = post.get('user_info', {}) or post.get('author_info', {})
            author = user_info.get('username') or user_info.get('name') or 'Unknown'
        content_parts.append(f"Author: {author}")
        
        # Build full text for processing
        full_text = f"{title} {text}"
        
        # Use TextProcessor for entity extraction
        cryptos = []
        hashtags = []
        
        if full_text.strip():
            # Check if entities already extracted (by integrator)
            extracted = post.get('extracted_entities', {})
            text_processing = post.get('text_processing', {})
            
            if extracted:
                cryptos = extracted.get('cryptocurrencies', [])
            else:
                entities = self.text_processor.extract_entities(full_text)
                cryptos = entities.cryptocurrencies
            
            if text_processing:
                hashtags = text_processing.get('hashtags', [])
            else:
                processed = self.text_processor.preprocess(full_text)
                hashtags = processed.hashtags
        
        # Use HashtagAnalyzer to extract and record hashtags
        if full_text.strip():
            hashtag_result = self.hashtag_analyzer.extract_and_record(
                full_text, 
                sentiment=0.0, 
                source=platform.lower()
            )
            if hashtag_result.get('hashtags'):
                hashtags.extend(hashtag_result['hashtags'])
                hashtags = list(set(hashtags))  # Dedupe
        
        # Get sentiment
        sentiment_label = 'unknown'
        sentiment_score = 0.0
        
        # ✅ CORRECT: Use 'label' and 'score' fields
        sentiment_data = post.get('sentiment_analysis', {})
        if sentiment_data and isinstance(sentiment_data, dict):
            sentiment_label = sentiment_data.get('label', 'unknown')
            sentiment_score = sentiment_data.get('score', 0.0)
        
        # ✅ FIX: Add None check for sentiment_result
        elif full_text.strip():
            try:
                sentiment_result = self.sentiment_analyzer.analyze(full_text)
                
                if sentiment_result is not None:
                    sentiment_label = sentiment_result.label.value
                    sentiment_score = sentiment_result.score
                else:
                    sentiment_label = 'unknown'
                    sentiment_score = 0.0
            except Exception as e:
                logger.warning(f"Error analyzing sentiment: {e}")
                sentiment_label = 'unknown'
                sentiment_score = 0.0
        
        if sentiment_label != 'unknown':
            content_parts.append(f"Sentiment: {sentiment_label}")

            
        content = "\n".join(content_parts)
        
        # Build doc_id
        post_id = post.get('id') or post.get('video_id') or post.get('source_id', '')
        doc_id = f"social_{post_id}"
        
        # Extract published_at from various RAW formats
        published_at = post.get('published_at') or post.get('created_at') or ''
        if not published_at and post.get('created_utc'):
            try:
                published_at = datetime.fromtimestamp(
                    post['created_utc'], tz=timezoneDt.utc
                ).isoformat()
            except:
                published_at = ''
        
        metadata = {
            'type': 'social',
            'doc_id': doc_id,
            'title': title or (text[:100] if text else ''),
            'platform': platform.lower(),
            'author': author,
            'published_at': str(published_at),
            'trust_score': float(post.get('trust_score', 0)),
            'url': post.get('url') or post.get('permalink', ''),
            'sentiment': sentiment_label,
            'sentiment_score': sentiment_score,
            'cryptocurrencies': ','.join(cryptos[:5]) if cryptos else '',
            'hashtags': ','.join(hashtags[:10]) if hashtags else '',
        }
        
        # Add platform-specific metadata from RAW fields
        if platform.lower() == 'reddit':
            metadata['subreddit'] = post.get('subreddit', '')
            metadata['score'] = post.get('score', 0)
            metadata['upvote_ratio'] = post.get('upvote_ratio', 0)
        elif platform.lower() == 'twitter':
            public_metrics = post.get('public_metrics', {})
            metadata['likes'] = public_metrics.get('like_count', 0)
            metadata['retweets'] = public_metrics.get('retweet_count', 0)
        elif platform.lower() == 'youtube':
            metadata['view_count'] = post.get('view_count', 0)
            metadata['like_count'] = post.get('like_count', 0)
            metadata['channel'] = post.get('channel_title', '')
        
        return Document(
            text=content,
            metadata=metadata,
            doc_id=doc_id
        )
    
    def index_article(self, article: Dict) -> bool:
        """Index a single news article"""
        try:
            doc = self._create_document_from_article(article)
            
            # Check for duplicates
            if doc.doc_id in self.index.docstore.docs:
                logger.debug(f"Document {doc.doc_id} already exists")
                return False
            
            # Parse into nodes
            nodes = self.node_parser.get_nodes_from_documents([doc])
            
            # Add nodes to index
            self.index.insert_nodes(nodes)
            
            logger.debug(f"Indexed article: {doc.doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing article: {e}")
            return False
    
    def index_social_post(self, post: Dict) -> bool:
        """Index a single social media post"""
        try:
            doc = self._create_document_from_social(post)
            
            if doc.doc_id in self.index.docstore.docs:
                logger.debug(f"Document {doc.doc_id} already exists")
                return False
            
            nodes = self.node_parser.get_nodes_from_documents([doc])
            self.index.insert_nodes(nodes)
            
            logger.debug(f"Indexed social post: {doc.doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing social post: {e}")
            return False
    
    def bulk_index_articles(self, articles: List[Dict]) -> Dict[str, int]:
        """Bulk index news articles"""
        stats = {'added': 0, 'duplicates': 0, 'errors': 0}
        
        documents = []
        for article in articles:
            try:
                doc = self._create_document_from_article(article)
                
                # Check for duplicates
                if doc.doc_id in self.index.docstore.docs:
                    stats['duplicates'] += 1
                    continue
                
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error creating document: {e}")
                stats['errors'] += 1
        
        if documents:
            try:
                # Parse all documents into nodes
                nodes = self.node_parser.get_nodes_from_documents(documents)
                
                # Bulk insert
                self.index.insert_nodes(nodes)
                stats['added'] = len(documents)
                
                logger.info(f"Bulk indexed {stats['added']} articles")
                
            except Exception as e:
                logger.error(f"Error in bulk indexing: {e}")
                stats['errors'] += len(documents)
                stats['added'] = 0
        
        return stats
    
    def bulk_index_social_posts(self, posts: List[Dict]) -> Dict[str, int]:
        """Bulk index social media posts"""
        stats = {'added': 0, 'duplicates': 0, 'errors': 0}
        
        documents = []
        for post in posts:
            try:
                doc = self._create_document_from_social(post)
                
                # Check for duplicates
                if doc.doc_id in self.index.docstore.docs:
                    stats['duplicates'] += 1
                    continue
                
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error creating document from social post: {e}")
                stats['errors'] += 1
        
        if documents:
            try:
                # Parse all documents into nodes
                nodes = self.node_parser.get_nodes_from_documents(documents)
                
                # Bulk insert
                self.index.insert_nodes(nodes)
                stats['added'] = len(documents)
                
                logger.info(f"Bulk indexed {stats['added']} social posts")
                
            except Exception as e:
                logger.error(f"Error in bulk indexing social posts: {e}")
                stats['errors'] += len(documents)
                stats['added'] = 0
        
        return stats
    
    def retrieve(self, query: str, top_k: int = 10,
                 filters: Optional[Dict] = None) -> List[SearchResult]:
        """
        Retrieve relevant documents using semantic search
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
             
        Returns: 
            List of SearchResult objects
        """
        try: 
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k * 2 if filters else top_k  # Get more for filtering
            )
            
            # Retrieve nodes
            nodes_with_scores = retriever.retrieve(query)
            
            # Post-process and filter
            results = []
            for node_with_score in nodes_with_scores:
                node = node_with_score.node
                score = node_with_score.score
                
                # Apply similarity cutoff
                if score < self.config['similarity_cutoff']:
                    continue
                
                # Apply metadata filters
                if filters:
                    if not self._matches_filter(node.metadata, filters):
                        continue
                
                results.append(SearchResult(
                    id=node.metadata.get('doc_id', node.node_id),
                    content=node.text,
                    metadata=node.metadata,
                    score=float(score),
                    rank=len(results) + 1
                ))
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []
    
    def _matches_filter(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches filters"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        
        return True
    
    def generate_answer(self, query: str,
                        context_results: Optional[List[SearchResult]] = None,
                        system_prompt: Optional[str] = None) -> RAGResponse:
        """
        Generate answer using RAG
        
        Args:
            query: User query
            context_results: Pre-retrieved results (optional)
            system_prompt: Custom system prompt
            
        Returns:
            RAGResponse with answer and sources
        """
        start_time = datetime.now()
        
        # Retrieve if not provided
        if context_results is None:
            context_results = self.retrieve(query, top_k=self.config['top_k_retrieval'])
        
        if not context_results:
            return RAGResponse(
                query=query,
                answer="I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different topic.",
                sources=[],
                confidence=0.0,
                tokens_used=0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                model_used=f"{self.llm_provider}/{self.llm_model}"
            )
        
        # Use LlamaIndex query engine
        if self.llm:
            answer, tokens_used = self._generate_with_llm(query, context_results, system_prompt)
        else:
            answer, tokens_used = self._generate_fallback(query, context_results)
        
        # Calculate confidence
        avg_score = sum(r.score for r in context_results) / len(context_results)
        confidence = min(avg_score, 1.0)
        
        # Prepare sources
        sources = [
            {
                'id': r.id,
                'title': r.metadata.get('title', 'Untitled'),
                'source': r.metadata.get('source', 'Unknown'),
                'url': r.metadata.get('url', ''),
                'relevance_score': round(r.score, 3),
                'trust_score': r.metadata.get('trust_score', 0),
                'type': r.metadata.get('type', 'news'),
                'sentiment': r.metadata.get('sentiment', 'neutral')
            }
            for r in context_results[:5]
        ]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return RAGResponse(
            query=query,
            answer=answer,
            sources=sources,
            confidence=round(confidence, 3),
            tokens_used=tokens_used,
            processing_time=round(processing_time, 3),
            model_used=f"{self.llm_provider}/{self.llm_model}"
        )
    
    def _generate_with_llm(self, query: str, context_results: List[SearchResult],
                        system_prompt: Optional[str] = None) -> Tuple[str, int]:
        """Generate answer using LLM (with Ollama HTTP support)"""
        if system_prompt is None:
            system_prompt = """You are a cryptocurrency news analyst assistant. Your role is to:
1. Analyze and synthesize information from multiple crypto news sources
2. Provide accurate, well-sourced answers based on the provided context
3. Always cite your sources using [Source X] notation
4. Mention the credibility/trust score of sources when relevant
5. If information conflicts between sources, acknowledge this
6. If you're unsure or the context doesn't contain enough information, say so
7. Focus on factual information and avoid speculation

Be concise but thorough. Always ground your answers in the provided sources."""
        
        # Build context
        context_parts = []
        for i, result in enumerate(context_results, 1):
            context_parts.append(f"[Source {i}]")
            context_parts.append(f"Title: {result.metadata.get('title', 'Untitled')}")
            context_parts.append(f"From: {result.metadata.get('source', 'Unknown')}")
            context_parts.append(f"Trust Score: {result.metadata.get('trust_score', 'N/A')}/10")
            context_parts.append(f"Content: {result.content[:1000]}")
            context_parts.append("---")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Based on the following sources, answer this question:

Question: {query}

Sources:
{context}

Please provide a comprehensive answer with citations to the sources."""
        
        try:
            # ✅ Special handling for Ollama using raw HTTP
            if self.llm_provider == LLMProviderType.OLLAMA.value:
                import requests
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                response = requests.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.llm_model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": self.config['temperature'],
                            "num_predict": self.config['max_tokens']
                        }
                    },
                    timeout=120.0  # ✅ Increased timeout for longer responses
                )
                
                if response.status_code == 200:
                    result = response.json()
                    message_content = result.get('message', {}).get('content', '')
                    tokens = result.get('eval_count', 0)
                    return message_content, tokens
                else:
                    raise RuntimeError(f"Ollama error: {response.text}")
            
            # For other providers, use LlamaIndex
            from llama_index.core.llms import ChatMessage, MessageRole
            
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=prompt)
            ]
            
            response = self.llm.chat(messages)
            
            # Extract token usage if available
            tokens_used = 0
            if hasattr(response, 'raw') and response.raw:
                usage = getattr(response.raw, 'usage', None)
                if usage:
                    tokens_used = getattr(usage, 'total_tokens', 0)
            
            return response.message.content, tokens_used
            
        except Exception as e:
            logger.error(f"Error generating with LLM: {e}")
            return f"Error generating answer: {str(e)}", 0

    def _generate_fallback(self, query: str, results: List[SearchResult]) -> Tuple[str, int]:
        """Fallback answer generation without LLM"""
        answer_parts = [f"Based on {len(results)} relevant sources:\n"]
        
        for i, result in enumerate(results[:3], 1):
            title = result.metadata.get('title', 'Untitled')
            source = result.metadata.get('source', 'Unknown')
            trust = result.metadata.get('trust_score', 'N/A')
            
            answer_parts.append(f"{i}. **{title}** (Source: {source}, Trust: {trust}/10)")
            
            content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            answer_parts.append(f"   {content_preview}\n")
        
        return "\n".join(answer_parts), 0
    
    def generate_answer_with_kg(self, query: str,
                                use_knowledge_graph: bool = True) -> RAGResponse:
        """Generate answer using both vector search and knowledge graph"""
        start_time = datetime.now()
        
        # Retrieve from vector store
        context_results = self.retrieve(query, top_k=self.config['top_k_retrieval'])
        
        # Get knowledge graph context if enabled
        kg_context = ""
        if use_knowledge_graph:
            try:
                from myapp.services.rag.knowledge_graph import get_knowledge_graph
                
                kg = get_knowledge_graph()
                entities_in_query = self._extract_entities_from_query(query)
                
                kg_parts = []
                for entity in entities_in_query[:3]:
                    entity_id = kg._normalize_entity_id(entity)
                    if entity_id and entity_id in kg.entities:
                        ctx = kg.get_entity_context(entity_id, depth=1)
                        
                        kg_parts.append(f"\n=== Knowledge Graph: {entity} ===")
                        kg_parts.append(f"Type: {ctx['entity']['type']}")
                        
                        if ctx['related_entities']:
                            related = [f"{r['name']} ({r['relation']})" for r in ctx['related_entities'][:5]]
                            kg_parts.append(f"Related: {', '.join(related)}")
                        
                        if ctx['recent_events']:
                            events = [f"- {e['title'][:50]}... ({e['sentiment']})" for e in ctx['recent_events'][:3]]
                            kg_parts.append(f"Recent Events:\n" + "\n".join(events))
                
                kg_context = "\n".join(kg_parts)
                
            except Exception as e:
                logger.warning(f"Error getting KG context: {e}")
        
        # Generate with combined context
        system_prompt = """You are a cryptocurrency news analyst with access to:
1. Recent news articles (with trust scores)
2. A knowledge graph of crypto entities and their relationships

Use both sources to provide comprehensive, well-sourced answers.
Always cite sources using [Source X] notation.
Include relevant entity relationships when helpful."""
        
        if self.llm and context_results:
            # Append KG context to the generation
            answer, tokens_used = self._generate_with_llm_and_kg(
                query, context_results, kg_context, system_prompt
            )
        else:
            answer, tokens_used = self._generate_fallback(query, context_results)
        
        # Calculate confidence
        avg_score = sum(r.score for r in context_results) / len(context_results) if context_results else 0
        
        sources = [
            {
                'id': r.id,
                'title': r.metadata.get('title', 'Untitled'),
                'source': r.metadata.get('source', 'Unknown'),
                'url': r.metadata.get('url', ''),
                'relevance_score': round(r.score, 3),
                'trust_score': r.metadata.get('trust_score', 0),
                'type': r.metadata.get('type', 'news')
            }
            for r in context_results[:5]
        ]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return RAGResponse(
            query=query,
            answer=answer,
            sources=sources,
            confidence=round(min(avg_score, 1.0), 3),
            tokens_used=tokens_used,
            processing_time=round(processing_time, 3),
            model_used=f"{self.llm_provider}/{self.llm_model}"
        )
    
    def _generate_with_llm_and_kg(self, query: str, context_results: List[SearchResult],
                                kg_context: str, system_prompt: str) -> Tuple[str, int]:
        """Generate with both vector results and KG context (with Ollama HTTP support)"""
        # Build context
        context_parts = []
        for i, result in enumerate(context_results, 1):
            context_parts.append(f"[Source {i}]")
            context_parts.append(f"Title: {result.metadata.get('title', 'Untitled')}")
            context_parts.append(f"From: {result.metadata.get('source', 'Unknown')}")
            context_parts.append(f"Trust Score: {result.metadata.get('trust_score', 'N/A')}/10")
            context_parts.append(f"Content: {result.content[:1000]}")
            context_parts.append("---")
        
        vector_context = "\n".join(context_parts)
        
        full_context = vector_context
        if kg_context:
            full_context = f"{vector_context}\n\n{kg_context}"
        
        prompt = f"""Based on the following sources and knowledge graph, answer this question:

Question: {query}

Sources:
{full_context}

Please provide a comprehensive answer with citations."""
        
        try:
            # ✅ Special handling for Ollama using raw HTTP
            if self.llm_provider == LLMProviderType.OLLAMA.value:
                import requests
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                response = requests.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.llm_model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": self.config['temperature'],
                            "num_predict": self.config['max_tokens']
                        }
                    },
                    timeout=120.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    message_content = result.get('message', {}).get('content', '')
                    tokens = result.get('eval_count', 0)
                    return message_content, tokens
                else:
                    raise RuntimeError(f"Ollama error: {response.text}")
            
            # For other providers, use LlamaIndex
            from llama_index.core.llms import ChatMessage, MessageRole
            
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=prompt)
            ]
            
            response = self.llm.chat(messages)
            
            tokens_used = 0
            if hasattr(response, 'raw') and response.raw:
                usage = getattr(response.raw, 'usage', None)
                if usage:
                    tokens_used = getattr(usage, 'total_tokens', 0)
            
            return response.message.content, tokens_used
            
        except Exception as e:
            logger.error(f"Error generating with LLM and KG: {e}")
            return f"Error generating answer: {str(e)}", 0

     
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extract potential entity names from query using TextProcessor
        """
        # Use TextProcessor for entity extraction
        entities = self.text_processor.extract_entities(query)
        
        # Combine all crypto-related entities
        found_entities = []
        found_entities.extend(entities.cryptocurrencies)
        found_entities.extend(entities.exchanges)
        found_entities.extend(entities.organizations)
        
        # Deduplicate and lowercase
        return list(set(e.lower() for e in found_entities if e))
    
    def generate_summary(self, topic: str, hours_back: int = 24,
                         min_trust_score: float = 6.0) -> Dict[str, Any]:
        """Generate a summary of news on a topic"""
        # Retrieve relevant documents
        results = self.retrieve(topic, top_k=20)
        
        # Filter by trust score
        filtered_results = [
            r for r in results
            if r.metadata.get('trust_score', 0) >= min_trust_score
        ]
        
        if not filtered_results:
            return {
                'topic': topic,
                'summary': 'No high-credibility news found for this topic.',
                'key_points': [],
                'sources': [],
                'sentiment_distribution': {},
                'generated_at': timezone.now().isoformat()
            }
        
        # Generate summary with LLM
        if self.llm:
            summary_text = self._generate_summary_with_llm(topic, filtered_results)
        else:
            summary_text = f"Found {len(filtered_results)} relevant articles about {topic}."
        
        # Calculate sentiment distribution
        sentiment_dist = {'bullish': 0, 'neutral': 0, 'bearish': 0}
        for r in filtered_results:
            sentiment = r.metadata.get('sentiment', 'neutral').lower()
            if sentiment in ['bullish', 'positive']:
                sentiment_dist['bullish'] += 1
            elif sentiment in ['bearish', 'negative']:
                sentiment_dist['bearish'] += 1
            else:
                sentiment_dist['neutral'] += 1
        
        return {
            'topic': topic,
            'summary': summary_text,
            'article_count': len(filtered_results),
            'sentiment_distribution': sentiment_dist,
            'average_trust_score': round(
                sum(r.metadata.get('trust_score', 0) for r in filtered_results) / len(filtered_results), 2
            ),
            'sources': [
                {
                    'title': r.metadata.get('title', ''),
                    'source': r.metadata.get('source', ''),
                    'url': r.metadata.get('url', ''),
                    'trust_score': r.metadata.get('trust_score', 0),
                    'relevance': round(r.score, 3)
                }
                for r in filtered_results[:5]
            ],
            'generated_at': timezone.now().isoformat()
        }
    
    def _generate_summary_with_llm(self, topic: str, results: List[SearchResult]) -> str:
        """Generate summary using LLM (with Ollama HTTP support)"""
        context_parts = []
        for i, r in enumerate(results[:10], 1):
            context_parts.append(f"[{i}] {r.metadata.get('title', '')}: {r.content[:500]}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Analyze these crypto news articles about "{topic}" and provide:

1. A brief executive summary (2-3 sentences)
2. Key points (bullet points)
3. Overall market sentiment
4. Any conflicting information between sources

Sources:
{context}"""
        
        try:
            # ✅ Special handling for Ollama using raw HTTP
            if self.llm_provider == LLMProviderType.OLLAMA.value:
                import requests
                
                messages = [
                    {"role": "system", "content": "You are a cryptocurrency market analyst. Provide concise, factual summaries."},
                    {"role": "user", "content": prompt}
                ]
                
                response = requests.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.llm_model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": self.config['temperature'],
                            "num_predict": self.config['max_tokens']
                        }
                    },
                    timeout=120.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('message', {}).get('content', '')
                else:
                    raise RuntimeError(f"Ollama error: {response.text}")
            
            # For other providers, use LlamaIndex
            from llama_index.core.llms import ChatMessage, MessageRole
            
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content="You are a cryptocurrency market analyst. Provide concise, factual summaries."),
                ChatMessage(role=MessageRole.USER, content=prompt)
            ]
            
            response = self.llm.chat(messages)
            return response.message.content
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Found {len(results)} relevant articles about {topic}."

    
    def analyze_entity(self, entity: str, entity_type: str = "cryptocurrency") -> Dict[str, Any]:
        """Analyze news coverage and sentiment for a specific entity"""
        results = self.retrieve(entity, top_k=30)
        
        if not results:
            return {
                'entity': entity,
                'entity_type': entity_type,
                'analysis': 'No coverage found for this entity.',
                'coverage_count': 0
            }
        
        # Analyze coverage
        sentiment_scores = []
        trust_scores = []
        sources = set()
        
        for r in results:
            sentiment = r.metadata.get('sentiment', 'neutral')
            if sentiment in ['bullish', 'positive']:
                sentiment_scores.append(1)
            elif sentiment in ['bearish', 'negative']:
                sentiment_scores.append(-1)
            else:
                sentiment_scores.append(0)
            
            trust_scores.append(r.metadata.get('trust_score', 5))
            sources.add(r.metadata.get('source', 'Unknown'))
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        avg_trust = sum(trust_scores) / len(trust_scores)
        
        if avg_sentiment > 0.2:
            sentiment_label = 'bullish'
        elif avg_sentiment < -0.2:
            sentiment_label = 'bearish'
        else:
            sentiment_label = 'neutral'
        
        # Generate analysis with LLM
        if self.llm:
            analysis_text = self._generate_entity_analysis(entity, results[:5])
        else:
            analysis_text = f"Found {len(results)} articles mentioning {entity}."
        
        return {
            'entity': entity,
            'entity_type': entity_type,
            'coverage_count': len(results),
            'unique_sources': len(sources),
            'sentiment': {
                'label': sentiment_label,
                'score': round(avg_sentiment, 3)
            },
            'average_trust_score': round(avg_trust, 2),
            'analysis': analysis_text,
            'top_sources': list(sources)[:5],
            'analyzed_at': timezone.now().isoformat()
        }
    
    def _generate_entity_analysis(self, entity: str, results: List[SearchResult]) -> str:
        """Generate entity analysis using LLM (with Ollama HTTP support)"""
        context = "\n".join([
            f"- {r.metadata.get('title', '')}: {r.content[:300]}"
            for r in results
        ])
        
        prompt = f"Analyze the recent news coverage for {entity}:\n\n{context}\n\nProvide a brief analysis of the news sentiment and key developments."
        
        try:
            # ✅ Special handling for Ollama using raw HTTP
            if self.llm_provider == LLMProviderType.OLLAMA.value:
                import requests
                
                messages = [
                    {"role": "system", "content": "You are a cryptocurrency analyst. Provide brief, factual analysis."},
                    {"role": "user", "content": prompt}
                ]
                
                response = requests.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.llm_model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": self.config['temperature'],
                            "num_predict": self.config['max_tokens']
                        }
                    },
                    timeout=120.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('message', {}).get('content', '')
                else:
                    raise RuntimeError(f"Ollama error: {response.text}")
            
            # For other providers, use LlamaIndex
            from llama_index.core.llms import ChatMessage, MessageRole
            
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content="You are a cryptocurrency analyst. Provide brief, factual analysis."),
                ChatMessage(role=MessageRole.USER, content=prompt)
            ]
            
            response = self.llm.chat(messages)
            return response.message.content
            
        except Exception as e:
            logger.error(f"Error generating entity analysis: {e}")
            return f"Found {len(results)} articles mentioning {entity}."

    
    def switch_llm_provider(self, provider: str, model: str) -> bool:
        """Switch to a different LLM provider"""
        try:
            old_provider = self.llm_provider
            old_model = self.llm_model
            
            self.llm_provider = provider
            self.llm_model = model
            
            self._initialize_llm()
            
            if self.llm is None:
                # Rollback
                self.llm_provider = old_provider
                self.llm_model = old_model
                self._initialize_llm()
                return False
            
            logger.info(f"Switched LLM to {provider}/{model}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching LLM: {e}")
            return False
    
    def save_index(self):
        """Persist index to disk"""
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            
            # Save LlamaIndex storage
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            
            # Save FAISS index separately
            faiss_index = self.storage_context.vector_store._faiss_index
            faiss.write_index(faiss_index, os.path.join(self.persist_dir, 'faiss.index'))
            
            logger.info(f"Saved index to {self.persist_dir}")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get RAG index statistics for monitoring
        
        Returns:
            Dictionary with index statistics
        """
        try:
            doc_count = len(self.index.docstore.docs) if self.index.docstore else 0
            
            # Calculate index size
            index_size_bytes = 0
            if os.path.exists(os.path.join(self.persist_dir, 'faiss.index')):
                index_size_bytes = os.path.getsize(os.path.join(self.persist_dir, 'faiss.index'))
            
            # Get document type breakdown
            type_counts = {'news': 0, 'social': 0}
            if self.index.docstore:
                for doc in self.index.docstore.docs.values():
                    doc_type = doc.metadata.get('type', 'unknown')
                    if doc_type in type_counts:
                        type_counts[doc_type] += 1
            
            # Get last update time
            last_updated = 'Never'
            if os.path.exists(os.path.join(self.persist_dir, 'docstore.json')):
                mtime = os.path.getmtime(os.path.join(self.persist_dir, 'docstore.json'))
                last_updated = datetime.fromtimestamp(mtime, tz=timezoneDt.utc).isoformat()
            
            return {
                'total_documents': doc_count,
                'document_types': type_counts,
                'index_size_mb': round(index_size_bytes / (1024 * 1024), 2),
                'embedding_model': self.embedding_model_name,
                'embedding_dimension': self.embedding_dim,
                'llm_provider': self.llm_provider,
                'llm_model': self.llm_model,
                'llm_available': self.llm is not None,
                'last_updated': last_updated,
                'persist_dir': self.persist_dir
            }
            
        except Exception as e:
            logger.error(f"Error getting RAG statistics: {e}")
            return {
                'error': str(e),
                'total_documents': 0
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG engine statistics"""
        doc_count = len(self.index.docstore.docs) if self.index.docstore else 0
        
        return {
            'vector_store': {
                'type': 'FAISS',
                'documents_count': doc_count,
                'embedding_model': self.embedding_model_name,
                'embedding_dimension': self.embedding_dim,
                'persist_dir': self.persist_dir
            },
            'llm': {
                'provider': self.llm_provider,
                'model': self.llm_model,
                'available': self.llm is not None
            },
            'config': self.config
        }
    
    def get_llm_status(self) -> Dict[str, Any]:
        """Get LLM provider status"""
        return {
            'active_provider': self.llm_provider,
            'active_model': self.llm_model,
            'is_available': self.llm is not None,
            'supported_providers': [p.value for p in LLMProviderType]
        }


# Singleton instance
_rag_engine_instance: Optional[LlamaIndexRAGEngine] = None


def get_rag_engine() -> LlamaIndexRAGEngine:
    """Get singleton RAG engine instance"""
    global _rag_engine_instance
    
    if _rag_engine_instance is None:
        embedding_model = getattr(django_settings, 'RAG_EMBEDDING_MODEL', EmbeddingModel.MPNET.value)
        llm_provider = getattr(django_settings, 'LLM_PRIMARY_PROVIDER', 'ollama')
        llm_model = getattr(django_settings, 'RAG_LLM_MODEL', 'llama3.1')
        
        _rag_engine_instance = LlamaIndexRAGEngine(
            embedding_model=embedding_model,
            llm_provider=llm_provider,
            llm_model=llm_model
        )
    
    return _rag_engine_instance


# Backward compatibility aliases
VectorStore = LlamaIndexRAGEngine
RAGEngine = LlamaIndexRAGEngine
get_vector_store = get_rag_engine