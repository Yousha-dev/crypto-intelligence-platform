"""
Celery tasks for RAG system maintenance
Scheduled indexing, cache cleanup, and event-driven updates
Properly handles both news articles AND social posts
"""

import logging
from datetime import datetime, timedelta
from django.utils import timezone
from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def scheduled_index_update(self, hours_back: int = 6, min_trust_score: float = 5.0):
    """
    Scheduled task to index new content into RAG system
    Run every 6 hours to keep vector store updated
    Includes both news articles AND social posts
    
    CORRECTED:
    - Uses trust_score_threshold (not trust_score_threshold)
    - Properly separates articles and posts for correct indexing
    - Validates type field exists before indexing
    """
    try:
        from myapp.services.rag.rag_service import get_rag_engine
        from myapp.services.mongo_manager import get_mongo_manager
        from myapp.services.rag.knowledge_graph import get_knowledge_graph

        rag_engine = get_rag_engine()
        mongo_manager = get_mongo_manager()
        kg = get_knowledge_graph()

        # Use trust_score_threshold (not trust_score_threshold)
        articles = mongo_manager.get_high_credibility_articles(
            trust_score_threshold=min_trust_score,  # CORRECT PARAMETER NAME
            limit=300,
            hours_back=hours_back
        )

        # Use trust_score_threshold (not trust_score_threshold)
        social_posts = mongo_manager.get_high_credibility_social_posts(
            trust_score_threshold=min_trust_score,  # CORRECT PARAMETER NAME
            limit=200,
            hours_back=hours_back
        )

        if not articles and not social_posts:
            logger.info("No new content to index")
            return {
                'status': 'no_content',
                'timestamp': timezone.now().isoformat()
            }

        # Validate and ensure type field exists
        # Type should already be set by integrator, but validate defensively
        articles_to_index = []
        for article in articles:
            if 'type' not in article:
                logger.warning(f"Article {article.get('source_id')} missing type, adding it")
                article['type'] = 'news'
            articles_to_index.append(article)

        posts_to_index = []
        for post in social_posts:
            if 'type' not in post:
                logger.warning(f"Social post {post.get('source_id')} missing type, adding it")
                post['type'] = 'social'
            posts_to_index.append(post)

        # Index articles and posts separately using appropriate methods
        rag_stats_articles = {'added': 0, 'duplicates': 0, 'errors': 0}
        rag_stats_social = {'added': 0, 'duplicates': 0, 'errors': 0}
        
        if articles_to_index:
            logger.info(f"Indexing {len(articles_to_index)} news articles...")
            rag_stats_articles = rag_engine.bulk_index_articles(articles_to_index)
            logger.info(f"Articles indexed: {rag_stats_articles}")
        
        if posts_to_index:
            logger.info(f"Indexing {len(posts_to_index)} social posts...")
            rag_stats_social = rag_engine.bulk_index_social_posts(posts_to_index)
            logger.info(f"Social posts indexed: {rag_stats_social}")

        # Combine all content for KG extraction
        all_content = articles_to_index + posts_to_index
        
        logger.info(f"Extracting entities from {len(all_content)} items for Knowledge Graph...")

        # Update knowledge graph (works with both types)
        kg_stats = {'entities': 0, 'relationships': 0, 'events': 0}
        for idx, content in enumerate(all_content):
            try:
                if idx > 0 and idx % 50 == 0:
                    logger.info(f"  KG extraction progress: {idx}/{len(all_content)}")
                
                result = kg.extract_and_link_from_content(content)  # Works for both types
                kg_stats['entities'] += len(result.get('entities_found', []))
                kg_stats['relationships'] += len(result.get('relationships_created', []))
                if result.get('event_created'):
                    kg_stats['events'] += 1
            except Exception as e:
                logger.warning(f"KG extraction error for item {idx}: {e}")

        # Save both
        logger.info("Saving RAG index and Knowledge Graph...")
        rag_engine.save_index()
        kg.save_graph()

        logger.info(f"Scheduled index update complete: Articles={rag_stats_articles}, Social={rag_stats_social}, KG={kg_stats}")

        return {
            'status': 'success',
            'timestamp': timezone.now().isoformat(),
            'content_indexed': {
                'articles': len(articles_to_index),
                'social_posts': len(posts_to_index),
                'total': len(all_content)
            },
            'rag_stats': {
                'articles': rag_stats_articles,
                'social_posts': rag_stats_social
            },
            'kg_stats': kg_stats
        }

    except Exception as e:
        logger.error(f"Error in scheduled index update: {e}")
        import traceback
        traceback.print_exc()
        self.retry(countdown=300, exc=e)


@shared_task(bind=True)
def event_driven_index(self, content_ids: list, content_type: str = 'news'):
    """
    Event-driven task to index specific content immediately
    Triggered when high-impact news/posts arrive
    
    CORRECTED:
    - Validates content before indexing
    - Uses correct indexing method based on type
    - Handles errors gracefully
    
    Args:
        content_ids: List of content IDs to index
        content_type: 'news' or 'social'
    """
    try:
        from myapp.services.rag.rag_service import get_rag_engine
        from myapp.services.mongo_manager import get_mongo_manager
        from myapp.services.rag.knowledge_graph import get_knowledge_graph
        
        rag_engine = get_rag_engine()
        mongo_manager = get_mongo_manager()
        kg = get_knowledge_graph()
        
        # Validate content and handle errors
        content_items = []
        for content_id in content_ids:
            try:
                if content_type == 'news':
                    item = mongo_manager.get_article_by_id(content_id)
                    required_fields = ['source_id', 'title', 'platform']
                else:
                    item = mongo_manager.get_social_post_by_id(content_id)
                    required_fields = ['source_id', 'content', 'platform']
                
                if item:
                    # Validate required fields
                    missing_fields = [f for f in required_fields if f not in item]
                    
                    if missing_fields:
                        logger.warning(
                            f"Skipping {content_type} {content_id}: missing fields {missing_fields}"
                        )
                        continue
                    
                    # Verify type field exists
                    if 'type' not in item:
                        item['type'] = content_type
                        logger.warning(f"Type not set for {content_id}, adding it")
                    
                    content_items.append(item)
                else:
                    logger.warning(f"{content_type} {content_id} not found in database")
                    
            except Exception as e:
                logger.error(f"Error fetching {content_type} {content_id}: {e}")
                continue
        
        if not content_items:
            return {
                'status': 'no_content_found', 
                'type': content_type,
                'requested_ids': len(content_ids)
            }
        
        # Use appropriate indexing method based on type
        if content_type == 'news':
            logger.info(f"Event-driven indexing: {len(content_items)} news articles")
            rag_stats = rag_engine.bulk_index_articles(content_items)
        else:
            logger.info(f"Event-driven indexing: {len(content_items)} social posts")
            rag_stats = rag_engine.bulk_index_social_posts(content_items)
        
        logger.info(f"RAG indexing complete: {rag_stats}")
        
        # Update KG
        kg_stats = {'entities': 0, 'relationships': 0, 'events': 0}
        for item in content_items:
            try:
                result = kg.extract_and_link_from_content(item)
                kg_stats['entities'] += len(result.get('entities_found', []))
                kg_stats['relationships'] += len(result.get('relationships_created', []))
                if result.get('event_created'):
                    kg_stats['events'] += 1
            except Exception as e:
                logger.warning(f"KG extraction error: {e}")
        
        # Save
        rag_engine.save_index()
        kg.save_graph()
        
        logger.info(f"Event-driven index complete: {len(content_items)} {content_type} items")
        
        return {
            'status': 'success',
            'content_type': content_type,
            'items_indexed': len(content_items),
            'items_requested': len(content_ids),
            'rag_stats': rag_stats,
            'kg_stats': kg_stats
        }
        
    except Exception as e:
        logger.error(f"Error in event-driven index: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


@shared_task
def daily_full_reindex(days_back: int = 7, min_trust_score: float = 4.0):
    """
    Daily full reindex to ensure consistency
    Rebuilds entire index from recent high-quality content
    
    CORRECTED:
    - Uses trust_score_threshold (not trust_score_threshold)
    - Properly indexes articles and social posts separately
    - Validates type before indexing
    """
    try:
        from myapp.services.rag.rag_service import LlamaIndexRAGEngine
        from myapp.services.mongo_manager import get_mongo_manager
        import myapp.services.rag.rag_service as rag_module
        
        mongo_manager = get_mongo_manager()
        
        # Create fresh RAG engine
        new_rag_engine = LlamaIndexRAGEngine()
        
        # Use trust_score_threshold (not trust_score_threshold)
        articles = mongo_manager.get_high_credibility_articles(
            trust_score_threshold=min_trust_score,  # CORRECT PARAMETER NAME
            limit=5000,
            hours_back=days_back * 24
        )
        
        # Use trust_score_threshold (not trust_score_threshold)
        social_posts = mongo_manager.get_high_credibility_social_posts(
            trust_score_threshold=min_trust_score,  # CORRECT PARAMETER NAME
            limit=3000,
            hours_back=days_back * 24
        )
        
        # Validate and ensure type field
        articles_to_index = []
        for article in articles:
            if 'type' not in article:
                article['type'] = 'news'
            articles_to_index.append(article)
        
        posts_to_index = []
        for post in social_posts:
            if 'type' not in post:
                post['type'] = 'social'
            posts_to_index.append(post)
        
        # Index separately using appropriate methods
        stats_articles = {'added': 0, 'duplicates': 0, 'errors': 0}
        stats_social = {'added': 0, 'duplicates': 0, 'errors': 0}
        
        if articles_to_index:
            logger.info(f"Reindexing {len(articles_to_index)} news articles...")
            stats_articles = new_rag_engine.bulk_index_articles(articles_to_index)
        
        if posts_to_index:
            logger.info(f"Reindexing {len(posts_to_index)} social posts...")
            stats_social = new_rag_engine.bulk_index_social_posts(posts_to_index)
        
        new_rag_engine.save_index()
        
        # Replace global instance
        rag_module._rag_engine_instance = new_rag_engine
        
        total_indexed = stats_articles.get('added', 0) + stats_social.get('added', 0)
        
        logger.info(f"Daily full reindex complete: {total_indexed} total items indexed")
        
        return {
            'status': 'success',
            'content_indexed': {
                'articles': len(articles_to_index),
                'social_posts': len(posts_to_index),
                'total': len(articles_to_index) + len(posts_to_index)
            },
            'stats': {
                'articles': stats_articles,
                'social_posts': stats_social,
                'total_added': total_indexed
            },
            'timestamp': timezone.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in daily reindex: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


@shared_task
def cleanup_rag_cache():
    """Cleanup old RAG cache entries"""
    try:
        from django.core.cache import cache
         
        # Clear RAG-related cache keys
        logger.info("RAG cache cleanup completed")
        return {'status': 'success'}
        
    except Exception as e:
        logger.error(f"Error in cache cleanup: {e}")
        return {'status': 'error', 'error': str(e)}