# mongo_manager.py
import os
from datetime import datetime, timedelta
from django.utils import timezone
from typing import List, Dict, Optional, Any
import logging
from pymongo import MongoClient, DESCENDING, ASCENDING, IndexModel
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from bson import ObjectId
from django.conf import settings

logger = logging.getLogger(__name__)

class CryptoNewsMongoManager:
    """
    MongoDB Manager for cryptocurrency news and social media content
    Optimized for rapid querying and flexible schema design
    """
      
    def __init__(self):
        self.client = None
        self.db = None
        self.collections = {}
        self._initialize_connection()
        self._setup_collections()
        self._create_indexes()
    
    def _initialize_connection(self):
        """Initialize MongoDB connection with error handling"""
        try:
            # Get MongoDB settings from Django settings
            mongo_config = getattr(settings, 'MONGODB_CONFIG', {
                'host': os.getenv('MONGODB_HOST', 'localhost'),
                'port': int(os.getenv('MONGODB_PORT', 27017)),
                'database': os.getenv('MONGO_DB_NAME', 'crypto_news'),
                'username': os.getenv('MONGODB_USER'),
                'password': os.getenv('MONGODB_PASSWORD'),
                'auth_source': os.getenv('MONGO_AUTH_SOURCE', 'admin'),
                'max_pool_size': 50,
                'server_selection_timeout_ms': 5000,
                'connect_timeout_ms': 5000
            })
            
            # Build connection string
            if mongo_config['username'] and mongo_config['password']:
                connection_string = (
                    f"mongodb://{mongo_config['username']}:{mongo_config['password']}@"
                    f"{mongo_config['host']}:{mongo_config['port']}/{mongo_config['database']}"
                    f"?authSource={mongo_config['auth_source']}"
                )
            else:
                # No authentication - for local development
                connection_string = f"mongodb://{mongo_config['host']}:{mongo_config['port']}"
            
            self.client = MongoClient(
                connection_string,
                maxPoolSize=mongo_config['max_pool_size'],
                serverSelectionTimeoutMS=mongo_config['server_selection_timeout_ms'],
                connectTimeoutMS=mongo_config['connect_timeout_ms']
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[mongo_config['database']]
            logger.info(f"MongoDB connection established to {mongo_config['database']}")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing MongoDB connection: {e}")
            raise

    def _setup_collections(self):
        """Setup MongoDB collections with proper schema"""
        collection_configs = {
            'news_articles': {
                'validator': {
                    '$jsonSchema': {
                        'bsonType': 'object',
                        'required': ['source_id', 'platform', 'title', 'trust_score', 'created_at'],
                        'properties': {
                            'source_id': {'bsonType': 'string'},
                            'platform': {'enum': ['cryptopanic', 'cryptocompare', 'newsapi', 'messari', 'coindesk']},
                            'title': {'bsonType': 'string', 'minLength': 1},
                            'trust_score': {'bsonType': 'number', 'minimum': 0, 'maximum': 10},
                            'status': {'enum': ['pending', 'approved', 'flagged', 'rejected']},
                            'created_at': {'bsonType': 'date'},
                            'updated_at': {'bsonType': 'date'}
                        }
                    }
                }
            },
            'social_posts': {
                'validator': {
                    '$jsonSchema': {
                        'bsonType': 'object',
                        'required': ['source_id', 'platform', 'content', 'trust_score', 'created_at'],
                        'properties': {
                            'source_id': {'bsonType': 'string'},
                            'platform': {'enum': ['reddit', 'twitter', 'youtube']},
                            'content': {'bsonType': 'string'},
                            'trust_score': {'bsonType': 'number', 'minimum': 0, 'maximum': 10},
                            'status': {'enum': ['pending', 'approved', 'flagged', 'rejected']},
                            'created_at': {'bsonType': 'date'},
                            'updated_at': {'bsonType': 'date'}
                        }
                    }
                }
            }
        }
        
        for collection_name, config in collection_configs.items():
            try:
                # Check if we can list collections (authentication test)
                existing_collections = self.db.list_collection_names()
                
                if collection_name not in existing_collections:
                    self.db.create_collection(collection_name, validator=config['validator'])
                    logger.info(f"Created collection: {collection_name}")
                
                self.collections[collection_name] = self.db[collection_name]
                
            except Exception as e:
                logger.error(f"Error setting up collection {collection_name}: {e}")
                # Fallback: create collection without validator
                self.collections[collection_name] = self.db[collection_name]
    
    def _create_indexes(self):
        """Create indexes for optimal query performance"""
        index_configs = {
            'news_articles': [
                IndexModel([('source_id', ASCENDING)], unique=True),
                IndexModel([('platform', ASCENDING), ('created_at', DESCENDING)]),
                IndexModel([('trust_score', DESCENDING)]),
                IndexModel([('status', ASCENDING)]),
                IndexModel([('created_at', DESCENDING)]),
                IndexModel([('crypto_relevance.relevance_score', DESCENDING)]),
                IndexModel([('sentiment_analysis.sentiment_label', ASCENDING)]),
                IndexModel([('content_hash', ASCENDING)]),  # For duplicate detection
            ],
            'social_posts': [
                IndexModel([('source_id', ASCENDING)], unique=True),
                IndexModel([('platform', ASCENDING), ('created_at', DESCENDING)]),
                IndexModel([('trust_score', DESCENDING)]),
                IndexModel([('status', ASCENDING)]),
                IndexModel([('created_at', DESCENDING)]),
                IndexModel([('author_username', ASCENDING)]),
                IndexModel([('engagement_metrics.total_engagement', DESCENDING)]),
            ]
        }
        
        for collection_name, indexes in index_configs.items():
            try:
                collection = self.collections[collection_name]
                collection.create_indexes(indexes)
                logger.info(f"Created indexes for {collection_name}")
            except Exception as e:
                logger.warning(f"Error creating indexes for {collection_name}: {e}")
    
    def get_recent_articles(self, 
                           hours_back: int = 24,
                           trust_score_threshold: float = 6.0,
                           limit: int = 500,
                           status_filter: Optional[List[str]] = None) -> List[Dict]:
        """
        Get recent articles for cross-verification candidate pool
        
        Args:
            hours_back: How many hours back to search
            trust_score_threshold: Minimum trust score
            limit: Maximum articles to return
            status_filter: List of acceptable statuses (default: ['approved', 'pending'])
            
        Returns:
            List of recent article documents
        """
        try:
            if status_filter is None:
                status_filter = ['approved', 'pending']
            
            query = {
                'created_at': {
                    '$gte': timezone.now() - timedelta(hours=hours_back)
                },
                'trust_score': {'$gte': trust_score_threshold},
                'status': {'$in': status_filter}
            }
            
            cursor = self.collections['news_articles'].find(query).sort([
                ('created_at', DESCENDING)
            ]).limit(limit)
            
            articles = list(cursor)
            logger.info(f"Retrieved {len(articles)} recent articles for verification pool")
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving recent articles: {e}")
            return []
    
    def get_recent_social_posts(self,
                                hours_back: int = 24,
                                trust_score_threshold: float = 6.0,
                                limit: int = 200,
                                status_filter: Optional[List[str]] = None,
                                platform: Optional[str] = None) -> List[Dict]:
        """
        Get recent social posts for cross-verification candidate pool
        
        Args:
            hours_back: How many hours back to search
            trust_score_threshold: Minimum trust score
            limit: Maximum posts to return
            status_filter: List of acceptable statuses (default: ['approved', 'pending'])
            platform: Optional platform filter (reddit, twitter, youtube)
            
        Returns:
            List of recent social post documents
        """
        try:
            if status_filter is None:
                status_filter = ['approved', 'pending']
            
            query = {
                'created_at': {
                    '$gte': timezone.now() - timedelta(hours=hours_back)
                },
                'trust_score': {'$gte': trust_score_threshold},
                'status': {'$in': status_filter}
            }
            
            if platform:
                query['platform'] = platform.lower()
            
            cursor = self.collections['social_posts'].find(query).sort([
                ('created_at', DESCENDING)
            ]).limit(limit)
            
            posts = list(cursor)
            logger.info(f"Retrieved {len(posts)} recent social posts for verification pool")
            return posts
            
        except Exception as e:
            logger.error(f"Error retrieving recent social posts: {e}")
            return []

        
        
    def insert_news_article(self, article_document: Dict) -> Optional[ObjectId]:
        """
        Insert a PRE-PREPARED news article document
        
        Args:
            article_document: FULLY PREPARED document from integrator
            
        Returns:
            ObjectId of inserted document or None if failed
        """
        try:
            # Validate required fields
            required = ['source_id', 'platform', 'title', 'trust_score', 'created_at']
            missing = [f for f in required if f not in article_document]
            if missing:
                logger.error(f"Missing required fields: {missing}")
                return None
            
            # Check for duplicates
            existing = self.collections['news_articles'].find_one({
                'source_id': article_document['source_id']
            })
            
            if existing:
                logger.info(f"Article already exists: {article_document['source_id']}")
                return existing['_id']
            
            # Direct insert - no transformation
            result = self.collections['news_articles'].insert_one(article_document)
            logger.info(f"Inserted news article: {article_document['source_id']}")
            return result.inserted_id
            
        except DuplicateKeyError:
            logger.warning(f"Duplicate article: {article_document.get('source_id', 'unknown')}")
            return None
        except Exception as e:
            logger.error(f"Error inserting news article: {e}")
            return None

    
    def insert_social_post(self, post_data: Dict) -> Optional[ObjectId]:
        """
        Insert a social media post with credibility validation
        
        Args:
            post_data: Social post data from fetchers
            
        Returns:
            ObjectId of inserted document or None if failed
        """
        try:
            # Prepare document
            document = self._prepare_social_document(post_data)
            
            # Check for duplicates
            existing = self.collections['social_posts'].find_one({
                'source_id': document['source_id']
            })
            
            if existing:
                logger.info(f"Post already exists: {document['source_id']}")
                return existing['_id']
            
            # Insert document
            result = self.collections['social_posts'].insert_one(document)
            logger.info(f"Inserted social post: {document['source_id']}")
            return result.inserted_id
            
        except DuplicateKeyError:
            logger.warning(f"Duplicate post: {post_data.get('id', 'unknown')}")
            return None
        except Exception as e:
            logger.error(f"Error inserting social post: {e}")
            return None
    
    def bulk_insert_articles(self, articles: List[Dict]) -> Dict:
        """
        Bulk insert PRE-PREPARED news articles
        Accepts documents that are already prepared by integrator

        Args:
            articles: List of PREPARED article documents

        Returns:
            Dict with insertion statistics
        """
        if not articles:
            return {'inserted': 0, 'duplicates': 0, 'errors': 0, 'duplicate_ids': []}

        stats = {'inserted': 0, 'duplicates': 0, 'errors': 0, 'duplicate_ids': []}

        # ADD THIS ENTIRE DEBUG SECTION HERE =====================================
        # import logging
        # logger = logging.getLogger(__name__)
        
        # logger.info(f"[BULK INSERT DEBUG] Starting bulk insert of {len(articles)} articles")
        
        # # Inspect first document in detail
        # if articles:
        #     first_doc = articles[0]
        #     logger.info(f"[BULK INSERT DEBUG] First document inspection:")
        #     logger.info(f"   source_id: {first_doc.get('source_id')} (type: {type(first_doc.get('source_id'))})")
        #     logger.info(f"   platform: {first_doc.get('platform')} (type: {type(first_doc.get('platform'))})")
        #     logger.info(f"   title: {first_doc.get('title', '')[:50]}...")
        #     logger.info(f"   trust_score: {first_doc.get('trust_score')} (type: {type(first_doc.get('trust_score'))})")
        #     logger.info(f"   status: {first_doc.get('status')} (type: {type(first_doc.get('status'))})")
        #     logger.info(f"   created_at: {first_doc.get('created_at')} (type: {type(first_doc.get('created_at'))})")
            
        #     # Check sentiment_analysis structure
        #     sentiment = first_doc.get('sentiment_analysis')
        #     logger.info(f"   sentiment_analysis type: {type(sentiment)}")
        #     if sentiment:
        #         logger.info(f"   sentiment_analysis keys: {list(sentiment.keys())}")
        #         logger.info(f"   sentiment label: {sentiment.get('label')}")
        #         logger.info(f"   sentiment score: {sentiment.get('score')}")
        #     else:
        #         logger.error(f"   sentiment_analysis is empty or None!")
            
            # Check all required fields
            # required = ['source_id', 'platform', 'title', 'trust_score', 'created_at']
            # missing = [f for f in required if not first_doc.get(f)]
            # if missing:
            #     logger.error(f"   Missing required fields: {missing}")
            # else:
            #     logger.info(f"   All required fields present")
        
        # END DEBUG SECTION ========================================================

        # Documents should already be prepared by integrator
        # Just validate they have required fields
        valid_documents = []
        for article in articles:
            if 'source_id' in article and 'platform' in article and 'title' in article:
                valid_documents.append(article)
            else:
                logger.error(f"Article missing required fields: {article.keys()}")
                stats['errors'] += 1

        if not valid_documents:
            logger.error(f"[BULK INSERT DEBUG] No valid documents to insert!")
            return stats

        # Debug: Log all attempted source_ids
        attempted_ids = [a['source_id'] for a in valid_documents]
        logger.debug(f"[bulk_insert_articles] Attempting to insert source_ids: {attempted_ids}")

        # Debug: Find existing source_ids in DB before insert
        existing_ids = set()
        try:
            cursor = self.collections['news_articles'].find(
                {'source_id': {'$in': attempted_ids}},
                {'source_id': 1}
            )
            existing_ids = set(doc['source_id'] for doc in cursor)
            logger.debug(f"[bulk_insert_articles] Existing source_ids in DB: {list(existing_ids)}")
        except Exception as e:
            logger.error(f"[bulk_insert_articles] Error checking existing source_ids: {e}")

        # ADD THIS: Try single insert first to catch validation errors
        logger.info(f"[BULK INSERT DEBUG] Testing single insert first...")
        try:
            test_result = self.collections['news_articles'].insert_one(valid_documents[0])
            logger.info(f"   Single insert successful! ID: {test_result.inserted_id}")
            # Remove the test document
            self.collections['news_articles'].delete_one({'_id': test_result.inserted_id})
            logger.info(f"   Test document cleaned up")
        except Exception as e:
            logger.error(f"   Single insert FAILED!")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error message: {str(e)}")
            if hasattr(e, 'details'):
                logger.error(f"   Error details: {e.details}")
            logger.error(f"   This is why bulk insert is failing!")
            stats['errors'] = len(valid_documents)
            return stats

        try:
            # Use ordered=False to continue on duplicates
            result = self.collections['news_articles'].insert_many(
                valid_documents, 
                ordered=False
            )
            stats['inserted'] = len(result.inserted_ids)
            logger.info(f"[BULK INSERT DEBUG] Successfully inserted {stats['inserted']} documents")
            # Duplicates are those that were already in DB before insert
            stats['duplicates'] = len(existing_ids)
            stats['duplicate_ids'] = list(existing_ids)
        except Exception as e:
            logger.error(f"[BULK INSERT DEBUG] Bulk insert exception!")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error message: {str(e)}")
            
            # Handle partial insertions and duplicates
            if hasattr(e, 'details'):
                stats['inserted'] = e.details.get('nInserted', 0)
                logger.info(f"   Partial insert: {stats['inserted']} documents inserted before error")
                
                # Get write errors details
                if 'writeErrors' in e.details:
                    logger.error(f"   Write errors: {len(e.details['writeErrors'])} errors")
                    for i, error in enumerate(e.details['writeErrors'][:3]):  # Show first 3
                        logger.error(f"   Error {i+1}: {error}")
                
                # After error, check again for which IDs are now in DB
                try:
                    cursor = self.collections['news_articles'].find(
                        {'source_id': {'$in': attempted_ids}},
                        {'source_id': 1}
                    )
                    existing_ids = set(doc['source_id'] for doc in cursor)
                    logger.debug(f"[bulk_insert_articles] Existing source_ids after insert: {list(existing_ids)}")
                except Exception as e2:
                    logger.error(f"[bulk_insert_articles] Error checking existing source_ids after insert: {e2}")
                stats['duplicates'] = len(existing_ids)
                stats['duplicate_ids'] = list(existing_ids)
            else:
                logger.error(f"Bulk insert error: {e}")
                stats['errors'] += len(valid_documents)

        logger.info(f"Bulk insert stats: {stats}")
        return stats


    def bulk_insert_social_posts(self, posts: List[Dict]) -> Dict:
        """
        Bulk insert PRE-PREPARED social media posts
        Accepts documents that are already prepared by integrator

        Args:
            posts: List of PREPARED post documents

        Returns:
            Dict with insertion statistics
        """
        if not posts:
            return {'inserted': 0, 'duplicates': 0, 'errors': 0, 'duplicate_ids': []}

        stats = {'inserted': 0, 'duplicates': 0, 'errors': 0, 'duplicate_ids': []}

        # Documents should already be prepared by integrator
        # Just validate they have required fields
        valid_documents = []
        for post in posts:
            if 'source_id' in post and 'platform' in post:
                valid_documents.append(post)
            else:
                logger.error(f"Post missing required fields: {post.keys()}")
                stats['errors'] += 1

        if not valid_documents:
            return stats

        # Debug: Log all attempted source_ids
        attempted_ids = [p['source_id'] for p in valid_documents]
        logger.debug(f"[bulk_insert_social_posts] Attempting to insert source_ids: {attempted_ids}")

        # Debug: Find existing source_ids in DB before insert
        existing_ids = set()
        try:
            cursor = self.collections['social_posts'].find(
                {'source_id': {'$in': attempted_ids}},
                {'source_id': 1}
            )
            existing_ids = set(doc['source_id'] for doc in cursor)
            logger.debug(f"[bulk_insert_social_posts] Existing source_ids in DB: {list(existing_ids)}")
        except Exception as e:
            logger.error(f"[bulk_insert_social_posts] Error checking existing source_ids: {e}")

        try:
            result = self.collections['social_posts'].insert_many(
                valid_documents, 
                ordered=False
            )
            stats['inserted'] = len(result.inserted_ids)
            stats['duplicates'] = len(existing_ids)
            stats['duplicate_ids'] = list(existing_ids)
        except Exception as e:
            if hasattr(e, 'details'):
                stats['inserted'] = e.details.get('nInserted', 0)
                # After error, check again for which IDs are now in DB
                try:
                    cursor = self.collections['social_posts'].find(
                        {'source_id': {'$in': attempted_ids}},
                        {'source_id': 1}
                    )
                    existing_ids = set(doc['source_id'] for doc in cursor)
                    logger.debug(f"[bulk_insert_social_posts] Existing source_ids after insert: {list(existing_ids)}")
                except Exception as e2:
                    logger.error(f"[bulk_insert_social_posts] Error checking existing source_ids after insert: {e2}")
                stats['duplicates'] = len(existing_ids)
                stats['duplicate_ids'] = list(existing_ids)
            else:
                logger.error(f"Bulk insert error: {e}")
                stats['errors'] += len(valid_documents)

        logger.info(f"Social posts bulk insert stats: {stats}")
        return stats

    
    def get_article_by_id(self, article_id: str) -> Optional[Dict]:
        """
        Get a single article by source_id
        
        Args:
            article_id: The source_id of the article
            
        Returns:
            Article document or None if not found
        """
        try:
            article = self.collections['news_articles'].find_one({
                'source_id': article_id
            })
            return article
            
        except Exception as e:
            logger.error(f"Error getting article by ID {article_id}: {e}")
            return None
    
    def get_social_post_by_id(self, post_id: str) -> Optional[Dict]:
        """
        Get a single social post by source_id
        
        Args:
            post_id: The source_id of the post
            
        Returns:
            Post document or None if not found
        """
        try:
            post = self.collections['social_posts'].find_one({
                'source_id': post_id
            }) 
            return post
             
        except Exception as e:
            logger.error(f"Error getting post by ID {post_id}: {e}")
            return None
    
    def get_high_credibility_articles(self, 
                                    trust_score_threshold: float = 7.0,  # Renamed param
                                    limit: int = 50,
                                    hours_back: int = 24,
                                    platform: Optional[str] = None) -> List[Dict]:
        """
        Retrieve high credibility articles
        Uses trust_score consistently
        """
        try:
            query = {
                'trust_score': {'$gte': trust_score_threshold},  # Only trust_score
                'created_at': {
                    '$gte': timezone.now() - timedelta(hours=hours_back)
                },
                'status': {'$in': ['approved', 'pending']}
            }
            
            if platform:
                query['platform'] = platform
            
            cursor = self.collections['news_articles'].find(query).sort([
                ('trust_score', DESCENDING),
                ('created_at', DESCENDING)
            ]).limit(limit)
            
            articles = list(cursor)
            logger.info(f"Retrieved {len(articles)} high credibility articles")
            return articles 
            
        except Exception as e:
            logger.error(f"Error retrieving high credibility articles: {e}")
            return []
    
    def get_high_credibility_social_posts(self, 
                                          trust_score_threshold: float = 7.0,  # Renamed
                                          limit: int = 50,
                                          hours_back: int = 24,
                                          platform: Optional[str] = None) -> List[Dict]:
        """
        Retrieve high credibility social posts
        Uses trust_score consistently
        """
        try:
            query = {
                'trust_score': {'$gte': trust_score_threshold},  # Only trust_score
                'created_at': {
                    '$gte': timezone.now() - timedelta(hours=hours_back)
                },
                'status': {'$in': ['approved', 'pending']}
            }
            
            if platform:
                query['platform'] = platform.lower()
            
            cursor = self.collections['social_posts'].find(query).sort([
                ('trust_score', DESCENDING),
                ('created_at', DESCENDING)
            ]).limit(limit)
            
            posts = list(cursor)
            logger.info(f"Retrieved {len(posts)} high credibility social posts")
            return posts
            
        except Exception as e:
            logger.error(f"Error retrieving high credibility social posts: {e}")
            return []
        
    def get_social_posts(self,
                        platform: Optional[str] = None,
                        status: Optional[str] = None,
                        hours_back: int = 24,
                        limit: int = 100,
                        min_engagement: int = 0) -> List[Dict]:
        """
        Retrieve social posts with filters
        
        Args:
            platform: Filter by platform (reddit, twitter, youtube)
            status: Filter by status (pending, approved, flagged, rejected)
            hours_back: How many hours back to search
            limit: Maximum number of posts to return
            min_engagement: Minimum total engagement
            
        Returns:
            List of social posts
        """
        try:
            query = {
                'created_at': {
                    '$gte': timezone.now() - timedelta(hours=hours_back)
                }
            }
            
            if platform:
                query['platform'] = platform.lower()
            
            if status:
                query['status'] = status
            
            if min_engagement > 0:
                query['$or'] = [
                    {'engagement_metrics.total_engagement': {'$gte': min_engagement}},
                    {'engagement_metrics.score': {'$gte': min_engagement}},
                    {'engagement_metrics.views': {'$gte': min_engagement}}
                ]
            
            cursor = self.collections['social_posts'].find(query).sort([
                ('created_at', DESCENDING)
            ]).limit(limit)
            
            posts = list(cursor)
            logger.info(f"Retrieved {len(posts)} social posts")
            return posts
            
        except Exception as e:
            logger.error(f"Error retrieving social posts: {e}")
            return []

    def get_social_posts_by_author(self, 
                                   author_username: str,
                                   platform: Optional[str] = None,
                                   limit: int = 50) -> List[Dict]:
        """
        Get social posts by author
        
        Args:
            author_username: Username to search for
            platform: Optional platform filter
            limit: Maximum posts to return
            
        Returns:
            List of posts by the author
        """
        try:
            query = {'author_username': author_username}
            
            if platform:
                query['platform'] = platform.lower()
            
            cursor = self.collections['social_posts'].find(query).sort([
                ('created_at', DESCENDING)
            ]).limit(limit)
            
            return list(cursor)
            
        except Exception as e:
            logger.error(f"Error getting posts by author {author_username}: {e}")
            return []

    def get_trending_social_content(self,
                                    hours_back: int = 24,
                                    limit: int = 20,
                                    platform: Optional[str] = None) -> List[Dict]:
        """
        Get trending social content based on engagement
        
        Args:
            hours_back: Time window
            limit: Maximum items to return
            platform: Optional platform filter
            
        Returns:
            List of trending posts sorted by engagement
        """
        try:
            match_stage = {
                'created_at': {
                    '$gte': timezone.now() - timedelta(hours=hours_back)
                },
                'status': {'$in': ['approved', 'pending']}
            }
            
            if platform:
                match_stage['platform'] = platform.lower()
            
            pipeline = [
                {'$match': match_stage},
                {'$addFields': {
                    'engagement_score': {
                        '$add': [
                            {'$ifNull': ['$engagement_metrics.total_engagement', 0]},
                            {'$ifNull': ['$engagement_metrics.score', 0]},
                            {'$multiply': [{'$ifNull': ['$engagement_metrics.views', 0]}, 0.01]}
                        ]
                    }
                }},
                {'$sort': {'engagement_score': -1, 'trust_score': -1}},
                {'$limit': limit}
            ]
            
            result = list(self.collections['social_posts'].aggregate(pipeline))
            logger.info(f"Retrieved {len(result)} trending social posts")
            return result
            
        except Exception as e:
            logger.error(f"Error getting trending social content: {e}")
            return []

    def get_social_statistics(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get social media statistics
        
        Args:
            hours_back: Time window for statistics
            
        Returns:
            Dictionary with social media statistics
        """
        try:
            time_filter = {
                'created_at': {
                    '$gte': timezone.now() - timedelta(hours=hours_back)
                }
            }
            
            # Platform breakdown
            platform_pipeline = [
                {'$match': time_filter},
                {'$group': {
                    '_id': '$platform',
                    'count': {'$sum': 1},
                    'avg_trust_score': {'$avg': '$trust_score'},
                    'avg_engagement': {'$avg': {
                        '$add': [
                            {'$ifNull': ['$engagement_metrics.total_engagement', 0]},
                            {'$ifNull': ['$engagement_metrics.score', 0]}
                        ]
                    }}
                }}
            ]
            
            platform_stats = list(self.collections['social_posts'].aggregate(platform_pipeline))
            
            # Status breakdown
            status_pipeline = [
                {'$match': time_filter},
                {'$group': {
                    '_id': '$status',
                    'count': {'$sum': 1}
                }}
            ]
            
            status_stats = list(self.collections['social_posts'].aggregate(status_pipeline))
            
            # Trust score distribution
            trust_pipeline = [
                {'$match': time_filter},
                {'$bucket': {
                    'groupBy': '$trust_score',
                    'boundaries': [0, 4, 6, 8, 10.1],
                    'default': 'unknown',
                    'output': {'count': {'$sum': 1}}
                }}
            ]
            
            trust_distribution = list(self.collections['social_posts'].aggregate(trust_pipeline))
            
            return {
                'time_window_hours': hours_back,
                'platform_breakdown': {
                    stat['_id']: {
                        'count': stat['count'],
                        'avg_trust_score': round(stat['avg_trust_score'] or 0, 2),
                        'avg_engagement': round(stat['avg_engagement'] or 0, 2)
                    }
                    for stat in platform_stats
                },
                'status_breakdown': {
                    stat['_id']: stat['count']
                    for stat in status_stats
                },
                'trust_distribution': trust_distribution,
                'total_posts': sum(stat['count'] for stat in platform_stats),
                'generated_at': timezone.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting social statistics: {e}")
            return {}
    
    def get_pending_content_for_review(self, limit: int = 100) -> Dict:
        """
        Get content pending manual review
        
        Args:
            limit: Maximum items to return
             
        Returns:
            Dict with pending articles and posts
        """
        try:
            articles = list(
                self.collections['news_articles']
                .find({'status': 'pending'})
                .sort('created_at', DESCENDING)
                .limit(limit // 2)
            )
            
            posts = list(
                self.collections['social_posts']
                .find({'status': 'pending'})
                .sort('created_at', DESCENDING)
                .limit(limit // 2)
            )
            
            return {
                'articles': articles,
                'posts': posts,
                'total': len(articles) + len(posts)
            }
            
        except Exception as e:
            logger.error(f"Error retrieving pending content: {e}")
            return {'articles': [], 'posts': [], 'total': 0}
    
    def update_content_status(self, content_id: str, status: str, 
                             collection_type: str = 'news_articles') -> bool:
        """
        Update content status (approved, flagged, rejected)
        
        Args:
            content_id: Content ID to update
            status: New status
            collection_type: 'news_articles' or 'social_posts'
            
        Returns:
            True if updated successfully
        """
        try:
            if status not in ['pending', 'approved', 'flagged', 'rejected']:
                raise ValueError(f"Invalid status: {status}")
            
            result = self.collections[collection_type].update_one(
                {'_id': ObjectId(content_id)},
                {
                    '$set': {
                        'status': status,
                        'updated_at': timezone.now()
                    }
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating content status: {e}")
            return False
    
    def get_content_by_topic(self, 
                           topic_keywords: List[str], 
                           hours_back: int = 48,
                           limit: int = 50) -> Dict:
        """
        Get content related to specific topics for cross-verification
        
        Args:
            topic_keywords: Keywords to search for
            hours_back: Time window for search
            limit: Maximum items to return
            
        Returns:
            Dict with articles and posts related to topic
        """
        try:
            # Create regex pattern for keywords
            keyword_pattern = '|'.join(topic_keywords)
            regex_query = {'$regex': keyword_pattern, '$options': 'i'}
            
            time_filter = {
                'created_at': {
                    '$gte': timezone.now() - timedelta(hours=hours_back)
                }
            } 
            
            # Search articles
            article_query = {
                **time_filter,
                '$or': [
                    {'title': regex_query},
                    {'description': regex_query},
                    {'content': regex_query}
                ]
            }
            
            articles = list(
                self.collections['news_articles']
                .find(article_query)
                .sort('trust_score', DESCENDING)
                .limit(limit // 2)
            )
            
            # Search social posts
            post_query = {
                **time_filter,
                '$or': [
                    {'content': regex_query},
                    {'title': regex_query}
                ]
            }
            
            posts = list(
                self.collections['social_posts']
                .find(post_query)
                .sort('trust_score', DESCENDING)
                .limit(limit // 2)
            )
            
            return {
                'articles': articles,
                'posts': posts,
                'total': len(articles) + len(posts),
                'topic_keywords': topic_keywords
            }
            
        except Exception as e:
            logger.error(f"Error searching content by topic: {e}")
            return {'articles': [], 'posts': [], 'total': 0}
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            stats = {}
            
            for collection_name in self.collections:
                collection = self.collections[collection_name]
                count = collection.count_documents({})
                
                # Fix: Convert cursor to list to get length
                indexes = list(collection.list_indexes())
                
                stats[collection_name] = {
                    'total_documents': count,
                    'indexes': len(indexes)
                }
            
            # Additional statistics
            now = timezone.now()
            last_24h = now - timedelta(hours=24)
            
            stats['recent_activity'] = {
                'articles_24h': self.collections['news_articles'].count_documents({
                    'created_at': {'$gte': last_24h}
                }),
                'posts_24h': self.collections['social_posts'].count_documents({
                    'created_at': {'$gte': last_24h}
                })
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

# Singleton instance
_mongo_manager_instance = None

def get_mongo_manager() -> CryptoNewsMongoManager:
    """Get singleton MongoDB manager instance"""
    global _mongo_manager_instance
    
    if _mongo_manager_instance is None:
        _mongo_manager_instance = CryptoNewsMongoManager()
    
    return _mongo_manager_instance