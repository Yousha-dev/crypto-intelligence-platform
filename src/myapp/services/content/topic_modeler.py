"""
Topic Modeling Service for Cryptocurrency Content
Uses BERTopic for dynamic topic discovery and tracking
"""

import logging
import threading
import warnings
import os

# Suppress warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", message=".*module compiled against ABI.*")
warnings.filterwarnings("ignore", message=".*RuntimeError.*ABI.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from datetime import datetime, timedelta
from django.utils import timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import pickle
 
from django.core.cache import cache
from django.conf import settings

logger = logging.getLogger(__name__) 
 
# Lazy imports for heavy libraries
_bertopic_model = None
_sentence_transformer = None
  
   
@dataclass
class TopicInfo:
    """Information about a discovered topic"""
    topic_id: int
    name: str
    keywords: List[str]
    document_count: int
    avg_sentiment: float
    velocity: float  # Rate of change in mentions
    first_seen: datetime
    last_seen: datetime
    is_trending: bool
    representative_docs: List[str] = field(default_factory=list)


@dataclass
class DocumentTopicAssignment:
    """Topic assignment for a document"""
    document_id: str
    primary_topic: int
    topic_name: str
    topic_probability: float
    secondary_topics: List[Tuple[int, float]]  # [(topic_id, probability)]


class CryptoTopicModeler:
    """
    Topic modeling service using BERTopic
    Discovers and tracks trending topics in crypto content
    """
    
    def __init__(self):
        self._model_initialized = False
        self._model = None
        self._embedder = None
        self._use_fallback = False
        
        # Topic tracking
        self.topic_history = defaultdict(list)  # topic_id -> [(timestamp, count)]
        self.topic_sentiments = defaultdict(list)  # topic_id -> [sentiment_scores]
        
        # Configuration
        self.config = {
            'min_topic_size': 5,  # Reduced for smaller datasets
            'n_gram_range': (1, 2),  # Reduced for smaller datasets
            'nr_topics': 'auto',
            'calculate_probabilities': True,
            'embedding_model': 'all-mpnet-base-v2',
            'velocity_window_hours': 6,
            'trending_threshold': 2.0,  # 2x normal velocity
        }
        
        # Crypto-specific seed topics for better initialization
        self.seed_topics = [
            ['bitcoin', 'btc', 'satoshi', 'halving'],
            ['ethereum', 'eth', 'vitalik', 'smart contract'],
            ['defi', 'yield', 'liquidity', 'farming'],
            ['nft', 'opensea', 'digital art', 'collectible'],
            ['regulation', 'sec', 'lawsuit', 'compliance'],
            ['exchange', 'binance', 'coinbase', 'trading'],
            ['stablecoin', 'usdt', 'usdc', 'tether'],
            ['layer2', 'scaling', 'rollup', 'polygon'],
        ]
        
        logger.info("CryptoTopicModeler initialized (model will load on first use)")
    
    def _ensure_model_loaded(self):
        """Lazy load BERTopic model"""
        if self._model_initialized:
            return
        
        try:
            # Suppress warnings during import
            import warnings
            warnings.filterwarnings("ignore")
            
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            from sklearn.feature_extraction.text import CountVectorizer
            from umap import UMAP
            from hdbscan import HDBSCAN
            
            logger.info("Loading BERTopic components...")
            
            # Custom embedding model
            self._embedder = SentenceTransformer(self.config['embedding_model'])
            
            # Store components for dynamic configuration
            self._BERTopic = BERTopic
            self._CountVectorizer = CountVectorizer
            self._UMAP = UMAP
            self._HDBSCAN = HDBSCAN
            
            self._model_initialized = True
            logger.info("BERTopic components loaded successfully")
            
        except ImportError as e:
            logger.error(f"Missing required library for topic modeling: {e}")
            logger.info("Install with: pip install bertopic sentence-transformers umap-learn hdbscan")
            self._model_initialized = True
            self._model = None
            self._use_fallback = True
        except Exception as e:
            logger.error(f"Failed to initialize BERTopic: {e}")
            self._model_initialized = True
            self._model = None
            self._use_fallback = True
    
    def _create_model_for_corpus(self, num_documents: int):
        """
        Create a BERTopic model configured safely for the given corpus size.
        Automatically avoids min_df/max_df conflicts and small-corpus errors.
        """

        if num_documents < 2:
            raise ValueError(f"Cannot create model for {num_documents} documents")

        # -------------------------------------------------------------
        # 1. Compute Safe min_df and max_df
        # -------------------------------------------------------------
        if num_documents <= 10:
            # For very small datasets, force absolute thresholds
            min_df = 1
            max_df = 1.0     # keep all terms
        elif num_documents <= 25:
            min_df = 1
            max_df = 0.95
        else:
            min_df = max(2, int(0.01 * num_documents))   # ~1% of documents
            max_df = 0.90

        # FIX CONFLICT: ensure max_df allows at least min_df+1 documents
        if isinstance(max_df, float):
            max_allowed_docs = int(max_df * num_documents)
            if max_allowed_docs < min_df + 1:
                # Expand max_df until compatible
                max_df = min(0.99, (min_df + 1) / num_documents)
                logger.warning(
                    f"[TopicModel] Auto-adjusted max_df to {max_df:.3f} for {num_documents} docs"
                )

        # -------------------------------------------------------------
        # 2. Safe topic/clustering Hyperparameters
        # -------------------------------------------------------------
        min_topic_size = max(2, min( max(3, num_documents // 4), self.config['min_topic_size'] ))
        n_neighbors = max(3, min(10, num_documents - 1))
        n_components = max(2, min(5, num_documents - 1))

        # -------------------------------------------------------------
        # 3. Vectorizer (Safe for small corpus)
        # -------------------------------------------------------------
        vectorizer = self._CountVectorizer(
            ngram_range=self.config['n_gram_range'],
            stop_words='english',
            min_df=min_df,
            max_df=max_df,
            max_features=2000
        )

        # -------------------------------------------------------------
        # 4. UMAP (Small-corpus friendly)
        # -------------------------------------------------------------
        umap_model = self._UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric='cosine',
            min_dist=0.0,
            random_state=42
        )

        # -------------------------------------------------------------
        # 5. HDBSCAN (Auto-tuned for small datasets)
        # -------------------------------------------------------------
        hdbscan_model = self._HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        # -------------------------------------------------------------
        # 6. Create BERTopic Model
        # -------------------------------------------------------------
        model = self._BERTopic(
            embedding_model=self._embedder,
            umap_model=umap_model, 
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            nr_topics=self.config['nr_topics'],
            calculate_probabilities=self.config['calculate_probabilities'],
            verbose=False
        ) 

        return model

    
    def _fallback_fit(self, documents: List[str], document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Simple TF-IDF based topic extraction as fallback"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            import numpy as np
            
            logger.info("Using fallback TF-IDF topic modeling...")
            
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Cluster into topics
            n_topics = min(5, max(2, len(documents) // 4))
            
            kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Extract top terms per cluster
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for i in range(n_topics):
                centroid = kmeans.cluster_centers_[i]
                top_indices = centroid.argsort()[-10:][::-1]
                keywords = [feature_names[idx] for idx in top_indices]
                
                topics.append({
                    'topic_id': i,
                    'name': f"Topic_{i}: {', '.join(keywords[:3])}",
                    'keywords': keywords,
                    'count': int(np.sum(clusters == i))
                })
            
            # Create assignments
            assignments = []
            if document_ids:
                for doc_id, cluster in zip(document_ids, clusters):
                    assignments.append({
                        'document_id': doc_id,
                        'topic_id': int(cluster),
                        'probability': 1.0
                    })
            
            # Record in history
            now = timezone.now()
            for topic in topics:
                self.topic_history[topic['topic_id']].append((now, topic['count']))
            
            return {
                'status': 'success',
                'mode': 'fallback_tfidf',
                'num_documents': len(documents),
                'num_topics': n_topics,
                'topics': topics,
                'topic_assignments': assignments
            }
            
        except Exception as e:
            logger.error(f"Fallback topic modeling failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def fit(self, documents: List[str], document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fit topic model on documents
        
        Args:
            documents: List of document texts
            document_ids: Optional list of document IDs
            
        Returns:
            Fitting results with topic info
        """
        self._ensure_model_loaded()
        
        # Use fallback if BERTopic not available
        if self._use_fallback or not hasattr(self, '_BERTopic'):
            return self._fallback_fit(documents, document_ids)
        
        # Check minimum documents
        min_docs = 10
        if len(documents) < min_docs:
            logger.info(f"Only {len(documents)} documents, using fallback method")
            return self._fallback_fit(documents, document_ids)
        
        try:
            logger.info(f"Fitting BERTopic model on {len(documents)} documents...")
            
            # Create model configured for this corpus size
            self._model = self._create_model_for_corpus(len(documents))
            
            # Fit the model
            topics, probs = self._model.fit_transform(documents)
            
            # Get topic info
            topic_info = self._model.get_topic_info()
            
            # Store results
            results = {
                'status': 'success',
                'mode': 'bertopic',
                'num_documents': len(documents),
                'num_topics': len(set(topics)) - (1 if -1 in topics else 0),
                'topics': [],
                'topic_assignments': []
            }
            
            now = timezone.now()
            
            # Process topic information
            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                if topic_id == -1:  # Skip outlier topic
                    continue
                
                topic_words = self._model.get_topic(topic_id)
                keywords = [word for word, _ in topic_words[:10]] if topic_words else []
                
                results['topics'].append({
                    'topic_id': topic_id,
                    'name': row.get('Name', f'Topic_{topic_id}'),
                    'keywords': keywords,
                    'count': row['Count']
                })
                
                # Record in history
                self.topic_history[topic_id].append((now, row['Count']))
            
            # Create topic assignments for documents
            if document_ids:
                for doc_id, topic, prob in zip(document_ids, topics, probs):
                    if isinstance(prob, (list, tuple)):
                        primary_prob = max(prob) if len(prob) > 0 else 0.0
                    elif hasattr(prob, '__iter__'):
                        prob_list = list(prob)
                        primary_prob = max(prob_list) if len(prob_list) > 0 else 0.0
                    else:
                        primary_prob = float(prob) if prob else 0.0
                    
                    results['topic_assignments'].append({
                        'document_id': doc_id,
                        'topic_id': int(topic),
                        'probability': float(primary_prob)
                    })
            
            logger.info(f"Topic modeling complete: {results['num_topics']} topics discovered")
            return results
            
        except Exception as e:
            logger.error(f"Error fitting BERTopic model: {e}")
            logger.info("Falling back to TF-IDF method...")
            return self._fallback_fit(documents, document_ids)
    
    def transform(self, documents: List[str], 
                  document_ids: Optional[List[str]] = None) -> List[DocumentTopicAssignment]:
        """
        Assign topics to new documents using fitted model
        
        Args:
            documents: List of document texts
            document_ids: Optional document IDs
            
        Returns:
            List of topic assignments
        """
        self._ensure_model_loaded()
        
        if self._model is None:
            return []
        
        try:
            # Transform documents
            topics, probs = self._model.transform(documents)
            
            assignments = []
            for i, (doc, topic, prob) in enumerate(zip(documents, topics, probs)):
                doc_id = document_ids[i] if document_ids else f"doc_{i}"
                
                # Get topic name
                topic_name = "Outlier"
                if topic != -1:
                    topic_info = self._model.get_topic_info()
                    topic_row = topic_info[topic_info['Topic'] == topic]
                    if not topic_row.empty:
                        topic_name = topic_row.iloc[0].get('Name', f'Topic_{topic}')
                
                # Get probability
                if isinstance(prob, (list, tuple)):
                    primary_prob = max(prob) if prob else 0.0
                    secondary = [(i, p) for i, p in enumerate(prob) if i != topic][:3]
                elif hasattr(prob, '__iter__'):
                    prob_list = list(prob)
                    primary_prob = max(prob_list) if prob_list else 0.0
                    secondary = [(i, p) for i, p in enumerate(prob_list) if i != topic][:3]
                else:
                    primary_prob = float(prob) if prob else 0.0
                    secondary = []
                
                assignments.append(DocumentTopicAssignment(
                    document_id=doc_id,
                    primary_topic=topic,
                    topic_name=topic_name,
                    topic_probability=primary_prob,
                    secondary_topics=secondary
                ))
            
            return assignments
            
        except Exception as e:
            logger.error(f"Error transforming documents: {e}")
            return []
    
    def get_trending_topics(self, hours_back: int = 24) -> List[TopicInfo]:
        """
        Get currently trending topics based on velocity
        
        Args:
            hours_back: Time window for trend analysis
            
        Returns:
            List of trending topics sorted by velocity
        """
        now = timezone.now()
        cutoff = now - timedelta(hours=hours_back)
        velocity_cutoff = now - timedelta(hours=self.config['velocity_window_hours'])
        
        trending = []
        
        for topic_id, history in self.topic_history.items():
            recent_history = [(ts, count) for ts, count in history if ts >= cutoff]
            
            if not recent_history:
                continue
            
            # Calculate velocity
            current_count = sum(count for ts, count in recent_history if ts >= velocity_cutoff)
            previous_count = sum(count for ts, count in recent_history if ts < velocity_cutoff)
            
            velocity = (current_count / max(previous_count, 1)) if previous_count > 0 else float(current_count)
            
            # Get sentiment
            sentiments = self.topic_sentiments.get(topic_id, [])
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
            
            # Get topic keywords
            keywords = []
            if self._model is not None:
                try:
                    topic_words = self._model.get_topic(topic_id)
                    keywords = [word for word, _ in topic_words[:10]] if topic_words else []
                except:
                    pass
            
            is_trending = velocity >= self.config['trending_threshold']
            
            trending.append(TopicInfo(
                topic_id=topic_id,
                name=f'Topic_{topic_id}',
                keywords=keywords,
                document_count=sum(count for _, count in recent_history),
                avg_sentiment=avg_sentiment,
                velocity=velocity,
                first_seen=min(ts for ts, _ in recent_history),
                last_seen=max(ts for ts, _ in recent_history),
                is_trending=is_trending
            ))
        
        # Sort by velocity
        trending.sort(key=lambda x: x.velocity, reverse=True)
        
        return trending
    
    def record_topic_occurrence(self, topic_id: int, sentiment: float = 0.0,
                                timestamp: Optional[datetime] = None):
        """Record a topic occurrence for velocity tracking"""
        if timestamp is None:
            timestamp = timezone.now()
        
        self.topic_history[topic_id].append((timestamp, 1))
        self.topic_sentiments[topic_id].append(sentiment)
        
        # Clean old data (keep last 7 days)
        cutoff = timezone.now() - timedelta(days=7)
        self.topic_history[topic_id] = [
            (ts, count) for ts, count in self.topic_history[topic_id]
            if ts >= cutoff
        ]
        self.topic_sentiments[topic_id] = self.topic_sentiments[topic_id][-1000:]
    
    def detect_topic_spikes(self, threshold_multiplier: float = 3.0) -> List[Dict[str, Any]]:
        """
        Detect sudden spikes in topic mentions
        
        Args:
            threshold_multiplier: How many times above average to consider a spike
            
        Returns:
            List of detected spikes
        """
        spikes = []
        now = timezone.now()
        
        for topic_id, history in self.topic_history.items():
            if len(history) < 5:  # Need enough data
                continue
            
            # Calculate hourly counts
            last_hour = now - timedelta(hours=1)
            last_24h = now - timedelta(hours=24)
            
            current_hour_count = sum(count for ts, count in history if ts >= last_hour)
            daily_counts = [count for ts, count in history if ts >= last_24h]
            
            if not daily_counts:
                continue
            
            avg_hourly = sum(daily_counts) / max(24, 1)
            
            if avg_hourly > 0 and current_hour_count >= avg_hourly * threshold_multiplier:
                keywords = []
                if self._model is not None:
                    try:
                        topic_words = self._model.get_topic(topic_id)
                        keywords = [word for word, _ in topic_words[:5]] if topic_words else []
                    except:
                        pass
                
                spikes.append({
                    'topic_id': topic_id,
                    'keywords': keywords,
                    'current_count': current_hour_count,
                    'average_count': avg_hourly,
                    'spike_multiplier': current_hour_count / max(avg_hourly, 1),
                    'detected_at': now.isoformat()
                })
        
        return sorted(spikes, key=lambda x: x['spike_multiplier'], reverse=True)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self._model is None:
            logger.warning("No model to save")
            return
        
        try:
            self._model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            self._ensure_model_loaded()
            if hasattr(self, '_BERTopic'):
                self._model = self._BERTopic.load(filepath)
                logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def get_topic_summary(self) -> Dict[str, Any]:
        """Get summary of current topic state"""
        trending = self.get_trending_topics()
        
        return {
            'total_topics': len(self.topic_history),
            'trending_topics': len([t for t in trending if t.is_trending]),
            'topics_with_history': len(self.topic_history),
            'model_loaded': self._model is not None,
            'using_fallback': self._use_fallback,
            'generated_at': timezone.now().isoformat()
        }


# Singleton
_topic_modeler_instance = None
_modeler_lock = threading.Lock()

def get_topic_modeler() -> CryptoTopicModeler:
    """Get singleton topic modeler instance (thread-safe)"""
    global _topic_modeler_instance
    
    if _topic_modeler_instance is None:
        with _modeler_lock:
            if _topic_modeler_instance is None:
                _topic_modeler_instance = CryptoTopicModeler()
    
    return _topic_modeler_instance