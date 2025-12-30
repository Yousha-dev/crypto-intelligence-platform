from celery import shared_task
from django.conf import settings
from django.utils import timezone
from django.core.cache import cache
from datetime import datetime, timedelta
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import asyncio
import logging
from typing import Dict
from myapp.services.content.integrator_service import ContentIntegrationService, process_social_posts_async, process_news_batch_async 
from myapp.services.content.topic_modeler import get_topic_modeler
from myapp.services.content.hashtag_analyzer import get_hashtag_analyzer


logger = logging.getLogger(__name__)

# WebSocket notification helper functions
def send_websocket_notification(group_name: str, message_type: str, data: Dict):
    """Send WebSocket notification to a group"""
    try:
        channel_layer = get_channel_layer()
        if channel_layer:
            async_to_sync(channel_layer.group_send)(
                group_name,
                {
                    'type': message_type,
                    **data,
                    'timestamp': timezone.now().isoformat()
                }
            )
    except Exception as e:
        logger.error(f"Error sending WebSocket notification: {e}")


def notify_fetch_started(source: str, task_id: str, config: Dict = None):
    """Notify that a fetch operation has started"""
    send_websocket_notification(
        "fetch_workflow",
        "fetch_started",
        {
            'source': source,
            'task_id': task_id,
            'config': config or {}
        }
    )


def notify_fetch_completed(source: str, task_id: str, result: Dict):
    """Notify that a fetch operation has completed"""
    send_websocket_notification(
        "fetch_workflow",
        "fetch_completed",
        {
            'source': source,
            'task_id': task_id,
            'result': result
        }
    )
    
    # Also send news feed update if there were successful articles/posts
    if result.get('approved', 0) > 0 or result.get('pending', 0) > 0:
        send_websocket_notification(
            "news_feed",
            "news_update",
            {
                'source': source,
                'stats': result,
                'has_new_content': True
            }
        )


def notify_fetch_failed(source: str, task_id: str, error: str):
    """Notify that a fetch operation has failed"""
    send_websocket_notification(
        "fetch_workflow",
        "fetch_failed",
        {
            'source': source,
            'task_id': task_id,
            'error': error
        }
    )


def notify_processing_stats(stats: Dict):
    """Notify about processing statistics"""
    send_websocket_notification(
        "fetch_workflow",
        "processing_stats",
        {
            'stats': stats
        }
    )


@shared_task
def initialize_series_configs():
    """Initialize series configurations and assign to workers"""
    from myapp.models import Exchange, TradingPair, SeriesConfig, WorkerShard
    from django.conf import settings
    import hashlib
    
    try:
        # Get all active exchanges and pairs
        exchanges = Exchange.objects.filter(isactive=1, isdeleted=0)
        trading_pairs = TradingPair.objects.filter(isactive=1, isdeleted=0).select_related('exchange')
        
        timeframes = getattr(settings, 'TRADING_CONFIG', {}).get('SUPPORTED_TIMEFRAMES', ['5m', '1h', '1d'])
        created_count = 0
        updated_count = 0
         
        # Create/update series configurations
        for pair in trading_pairs:
            for timeframe in timeframes:
                series_config, created = SeriesConfig.objects.get_or_create(
                    exchange=pair.exchange,
                    market_type=pair.exchange.market_type or 'spot',
                    symbol=pair.symbol,
                    timeframe=timeframe,
                    defaults={
                        'earliest_start_ms': None,
                        'last_backfill_ms': None,
                        'last_ts_ms': None,
                        'probing_completed': False,
                        'backfill_completed': False,
                        'isactive': 1,
                        'isdeleted': 0
                    }
                )
                
                if created:
                    created_count += 1
                else:
                    # Update active status if needed
                    if not series_config.isactive:
                        series_config.isactive = 1
                        series_config.save()
                        updated_count += 1
        
        # Assign series to worker shards using stable hashing
        series_configs = SeriesConfig.objects.filter(isactive=1, isdeleted=0)
        
        # Create default worker shards if none exist
        default_shards = 3
        for i in range(default_shards):
            shard_name = f"shard_{i}"
            WorkerShard.objects.get_or_create(
                shard_name=shard_name,
                defaults={
                    'worker_id': f"worker_{shard_name}",
                    'status': 'idle',
                    'assigned_series': [],
                    'series_count': 0,
                    'isactive': 1,
                    'isdeleted': 0
                }
            )
        
        # Distribute series across shards
        shards = list(WorkerShard.objects.filter(isactive=1, isdeleted=0))
        shard_assignments = {shard.shardid: [] for shard in shards}
        
        for config in series_configs:
            # Use stable hash to assign series to shard
            series_key = f"{config.exchange.exchangeid}_{config.market_type}_{config.symbol}_{config.timeframe}"
            hash_value = int(hashlib.md5(series_key.encode()).hexdigest(), 16)
            shard_index = hash_value % len(shards)
            shard_id = shards[shard_index].shardid
            shard_assignments[shard_id].append(series_key)
        
        # Update shard assignments
        for shard in shards:
            assigned_series = shard_assignments[shard.shardid]
            shard.assigned_series = assigned_series
            shard.series_count = len(assigned_series)
            shard.save()
        
        logger.info(f"Series initialization completed: {created_count} created, {updated_count} updated")
        return f"Initialized {created_count} new series, updated {updated_count}, distributed across {len(shards)} shards"
        
    except Exception as e:
        logger.error(f"Error initializing series configs: {e}")
        return f"Error: {e}"

@shared_task
def monitor_series_health():
    """Monitor series health and performance"""
    from myapp.models import SeriesConfig, WorkerShard
    from myapp.services.influx_manager import InfluxManager
    import time
    
    try:
        influx_manager = InfluxManager()
        if not influx_manager.client:
            return "InfluxDB not available for monitoring"
        
        current_time_ms = int(time.time() * 1000)
        alerts = []
        
        # Check for stale series (no data in last 3x timeframe)
        active_series = SeriesConfig.objects.filter(
            isactive=1,
            isdeleted=0,
            backfill_completed=True
        )
        
        stale_threshold = {
            '1m': 3 * 60 * 1000,      # 3 minutes
            '5m': 15 * 60 * 1000,     # 15 minutes
            '15m': 45 * 60 * 1000,    # 45 minutes
            '1h': 3 * 60 * 60 * 1000, # 3 hours
            '1d': 3 * 24 * 60 * 60 * 1000, # 3 days
        }
        
        stale_series = []
        for config in active_series:
            if config.last_ts_ms:
                age_ms = current_time_ms - config.last_ts_ms
                threshold = stale_threshold.get(config.timeframe, 60 * 60 * 1000)  # Default 1 hour
                
                if age_ms > threshold:
                    stale_series.append({
                        'series_key': config.series_key,
                        'age_minutes': age_ms // (60 * 1000),
                        'exchange': config.exchange.name,
                        'symbol': config.symbol,
                        'timeframe': config.timeframe
                    })
        
        if stale_series:
            alerts.append(f"Found {len(stale_series)} stale series")
        
        # Check worker shard health
        stale_workers = []
        workers = WorkerShard.objects.filter(isactive=1, isdeleted=0)
        
        for worker in workers:
            if worker.last_heartbeat:
                age = timezone.now() - worker.last_heartbeat
                if age.total_seconds() > 300:  # 5 minutes
                    stale_workers.append(worker.worker_id)
        
        if stale_workers:
            alerts.append(f"Found {len(stale_workers)} stale workers: {stale_workers}")
        
        # Check incomplete backfills
        incomplete_backfills = SeriesConfig.objects.filter(
            isactive=1,
            isdeleted=0,
            backfill_completed=False
        ).count()
        
        if incomplete_backfills > 0:
            alerts.append(f"Found {incomplete_backfills} incomplete backfills")
        
        result = {
            'timestamp': timezone.now().isoformat(),
            'alerts': alerts,
            'stale_series_count': len(stale_series),
            'stale_workers_count': len(stale_workers),
            'incomplete_backfills': incomplete_backfills,
            'status': 'healthy' if not alerts else 'degraded'
        }
        
        logger.info(f"Series health check: {result['status']} - {len(alerts)} alerts")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in series health monitoring: {e}")
        return {'status': 'error', 'message': str(e)}

# Celery tasks for fetcher integration
@shared_task(bind=True, max_retries=3)
def fetch_and_process_news_articles(self, source_config: Dict = None, async_processing: bool = True):
    """
    Fetch news from all sources and process through credibility pipeline
    Enhanced with WebSocket notifications for real-time monitoring
    
    UPDATED: Fetchers now return RAW data, integrator service handles analysis
    
    Args:
        source_config: Configuration for which sources to fetch from
        async_processing: Whether to process each source asynchronously
    """
    task_id = str(self.request.id)
    
    try:
        # Import fetchers - they now return RAW data only
        from myapp.fetchers.news_data import (
            fetch_cryptopanic_news,
            fetch_cryptocompare_news,
            fetch_newsapi_articles,
            fetch_messari_news,
            scrape_coindesk
        )
        
        # Default configuration
        default_config = {
            'cryptopanic': {'enabled': True, 'max_items': 50, 'filter_type': 'important'},
            'cryptocompare': {'enabled': True, 'max_items': 50, 'categories': 'BTC,ETH'},
            'newsapi': {'enabled': True, 'max_items': 30, 'query': 'cryptocurrency bitcoin ethereum'},
            'messari': {'enabled': True, 'max_items': 30},
            'coindesk': {'enabled': True, 'max_items': 30}
        }
        
        config = source_config or default_config
        results = {}
        
        # Notify start of fetch operation
        notify_fetch_started('news_batch', task_id, config)
        
        logger.info(f"Starting news fetching from {len([k for k, v in config.items() if v.get('enabled')])} sources")
        
        # Fetch from each enabled source
        for source_name, source_cfg in config.items():
            if not source_cfg.get('enabled', False):
                continue
                
            try:
                # Notify individual source fetch start
                notify_fetch_started(source_name, f"{task_id}_{source_name}", source_cfg)
                
                logger.info(f"Fetching from {source_name}...")
                articles = []
                
                # UPDATED: Call fetchers without analyze_credibility parameter
                # Fetchers now return RAW data only
                if source_name == 'cryptopanic':
                    articles = fetch_cryptopanic_news(
                        filter_type=source_cfg.get('filter_type', 'important'),
                        max_items=source_cfg.get('max_items', 50)
                    )
                elif source_name == 'cryptocompare':
                    articles = fetch_cryptocompare_news(
                        categories=source_cfg.get('categories', 'BTC,ETH'),
                        max_items=source_cfg.get('max_items', 50)
                    )
                elif source_name == 'newsapi':
                    articles = fetch_newsapi_articles(
                        query=source_cfg.get('query', 'cryptocurrency'),
                        max_items=source_cfg.get('max_items', 30)
                    )
                elif source_name == 'messari':
                    articles = fetch_messari_news(
                        max_items=source_cfg.get('max_items', 30)
                    )
                elif source_name == 'coindesk':
                    articles = scrape_coindesk(
                        max_items=source_cfg.get('max_items', 30)
                    )
                
                if articles:
                    # UPDATED: Articles are now RAW - integrator will process them
                    # Just ensure platform is set
                    for article in articles:
                        if 'platform' not in article:
                            article['platform'] = source_name
                    
                    if async_processing:
                        # Queue for async processing - integrator handles analysis
                        task = process_news_batch_async.delay(articles, source_name)
                        results[source_name] = {
                            'status': 'queued',
                            'task_id': task.id,
                            'articles_count': len(articles)
                        }
                    else:
                        # Process synchronously
                        service = ContentIntegrationService()
                        result = service.process_news_batch(articles, source_name)
                        results[source_name] = {
                            'status': 'completed',
                            'result': result.__dict__
                        }
                        
                        # Notify completion for sync processing
                        notify_fetch_completed(source_name, f"{task_id}_{source_name}", result.__dict__)
                    
                    logger.info(f"{source_name}: Fetched {len(articles)} articles (RAW)")
                else:
                    results[source_name] = {'status': 'no_articles', 'articles_count': 0}
                    notify_fetch_completed(source_name, f"{task_id}_{source_name}", {'articles_count': 0, 'status': 'no_articles'})
                    
            except Exception as e:
                logger.error(f"Error fetching from {source_name}: {e}")
                results[source_name] = {'status': 'error', 'error': str(e)}
                notify_fetch_failed(source_name, f"{task_id}_{source_name}", str(e))
                continue
        
        # Notify overall completion
        notify_fetch_completed('news_batch', task_id, results)
        
        logger.info(f"News fetching completed. Results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Error in fetch_and_process_news_articles: {e}")
        notify_fetch_failed('news_batch', task_id, str(e))
        self.retry(countdown=300, exc=e)  # Retry after 5 minutes


@shared_task(bind=True, max_retries=3)
def fetch_and_process_social_posts(self, platform_config: Dict = None, async_processing: bool = True):
    """
    Fetch social media posts and process through credibility pipeline
    Enhanced with WebSocket notifications for real-time monitoring
    
    UPDATED: Fetchers now return RAW data, integrator service handles analysis
    
    Args:
        platform_config: Configuration for which platforms to fetch from
        async_processing: Whether to process each platform asynchronously
    """
    task_id = str(self.request.id)
    
    try:
        # Import social fetchers - they now return RAW data only
        from myapp.fetchers.twitter import fetch_crypto_tweets
        from myapp.fetchers.reddit import fetch_crypto_reddit_posts
        from myapp.fetchers.youtube import fetch_crypto_youtube_videos
        
        # Default configuration
        default_config = {
            'twitter': {'enabled': True, 'max_items': 50, 'hours_back': 12},
            'reddit': {'enabled': True, 'max_items': 50, 'subreddit_limit': 3},
            'youtube': {'enabled': True, 'max_items': 20, 'days_back': 3}
        }
        
        config = platform_config or default_config
        results = {}
        
        # Notify start of social fetch operation
        notify_fetch_started('social_batch', task_id, config)
        
        logger.info(f"Starting social media fetching from {len([k for k, v in config.items() if v.get('enabled')])} platforms")
        
        # Fetch from each enabled platform
        for platform_name, platform_cfg in config.items():
            if not platform_cfg.get('enabled', False):
                continue
                
            try:
                # Notify individual platform fetch start
                notify_fetch_started(platform_name, f"{task_id}_{platform_name}", platform_cfg)
                
                logger.info(f"Fetching from {platform_name}...")
                posts = []
                
                # UPDATED: Call fetchers without trust_score_threshold
                # Fetchers return RAW data, credibility calculated by integrator
                if platform_name == 'twitter':
                    posts = fetch_crypto_tweets(
                        max_results=platform_cfg.get('max_items', 50),
                        hours_back=platform_cfg.get('hours_back', 12)
                    )
                elif platform_name == 'reddit':
                    posts = fetch_crypto_reddit_posts(
                        max_posts=platform_cfg.get('max_items', 50),
                        subreddit_limit=platform_cfg.get('subreddit_limit', 3)
                    )
                elif platform_name == 'youtube':
                    posts = fetch_crypto_youtube_videos(
                        max_results=platform_cfg.get('max_items', 20),
                        days_back=platform_cfg.get('days_back', 3)
                    )
                
                if posts:
                    # UPDATED: Posts are now RAW - integrator will process them
                    # Just ensure platform is set
                    for post in posts:
                        if 'platform' not in post:
                            post['platform'] = platform_name
                    
                    if async_processing:
                        # Queue for async processing - integrator handles analysis
                        task = process_social_posts_async.delay(posts, platform_name)
                        results[platform_name] = {
                            'status': 'queued',
                            'task_id': task.id,
                            'posts_count': len(posts)
                        }
                    else:
                        # Process synchronously
                        service = ContentIntegrationService()
                        result = service.process_social_posts_batch(posts, platform_name)
                        results[platform_name] = {
                            'status': 'completed',
                            'result': result.__dict__
                        }
                        
                        # Notify completion for sync processing
                        notify_fetch_completed(platform_name, f"{task_id}_{platform_name}", result.__dict__)
                    
                    logger.info(f"{platform_name}: Fetched {len(posts)} posts (RAW)")
                else:
                    results[platform_name] = {'status': 'no_posts', 'posts_count': 0}
                    notify_fetch_completed(platform_name, f"{task_id}_{platform_name}", {'posts_count': 0, 'status': 'no_posts'})
                    
            except Exception as e:
                logger.error(f"Error fetching from {platform_name}: {e}")
                results[platform_name] = {'status': 'error', 'error': str(e)}
                notify_fetch_failed(platform_name, f"{task_id}_{platform_name}", str(e))
                continue
        
        # Notify overall completion
        notify_fetch_completed('social_batch', task_id, results)
        
        logger.info(f"Social media fetching completed. Results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Error in fetch_and_process_social_posts: {e}")
        notify_fetch_failed('social_batch', task_id, str(e))
        self.retry(countdown=300, exc=e)


@shared_task
def comprehensive_content_update():
    """
    Comprehensive content update task - fetches from all sources
    This is the main scheduled task for regular news updates
    
    UPDATED: Removed trust_score_threshold from config since filtering
    now happens in integrator after analysis
    """
    try:
        logger.info("Starting comprehensive news update")
        
        # Notify comprehensive update start
        notify_fetch_started('comprehensive_update', 'comprehensive', {})
        
        # Configuration for comprehensive update
        # UPDATED: Removed trust_score_threshold - integrator handles filtering
        news_config = {
            'cryptopanic': {'enabled': True, 'max_items': 30, 'filter_type': 'important'},
            'cryptocompare': {'enabled': True, 'max_items': 30, 'categories': 'BTC,ETH'},
            'newsapi': {'enabled': True, 'max_items': 20, 'query': 'cryptocurrency bitcoin ethereum'},
            'messari': {'enabled': True, 'max_items': 20},
            'coindesk': {'enabled': True, 'max_items': 20}
        }
        
        social_config = {
            'twitter': {'enabled': True, 'max_items': 30, 'hours_back': 12},
            'reddit': {'enabled': True, 'max_items': 30, 'subreddit_limit': 3},
            'youtube': {'enabled': True, 'max_items': 10, 'days_back': 3}
        }
        
        # Start fetching tasks
        news_task = fetch_and_process_news_articles.delay(news_config, async_processing=True)
        social_task = fetch_and_process_social_posts.delay(social_config, async_processing=True)
        
        # Wait for completion and collect results
        news_result = news_task.get(timeout=1800)  # 30 minutes timeout
        social_result = social_task.get(timeout=1800)
        
        # Summary statistics
        summary = {
            'timestamp': timezone.now().isoformat(),
            'news_sources': news_result,
            'social_platforms': social_result,
            'total_sources_processed': len([k for k, v in {**news_result, **social_result}.items() if v.get('status') != 'error'])
        }
        
        # Notify comprehensive update completion
        notify_fetch_completed('comprehensive_update', 'comprehensive', summary)
        
        logger.info(f"Comprehensive news update completed: {summary}")
        return summary
        
    except Exception as e:
        logger.error(f"Error in comprehensive_content_update: {e}")
        notify_fetch_failed('comprehensive_update', 'comprehensive', str(e))
        return {'status': 'error', 'error': str(e)}


@shared_task
def scheduled_content_fetch():
    """
    Lighter scheduled task for frequent updates (every 2-4 hours)
    
    UPDATED: Removed trust_score_threshold from config
    """
    try:
        logger.info("Starting scheduled content fetch")
        
        # Notify scheduled fetch start
        notify_fetch_started('scheduled_fetch', 'scheduled', {})
         
        # Lighter configuration for frequent updates
        # UPDATED: Removed trust_score_threshold
        news_config = {
            'cryptopanic': {'enabled': True, 'max_items': 20, 'filter_type': 'important'},
            'coindesk': {'enabled': True, 'max_items': 15},
            'messari': {'enabled': True, 'max_items': 15}
        } 
        
        social_config = {
            'twitter': {'enabled': True, 'max_items': 20, 'hours_back': 6},
            'reddit': {'enabled': True, 'max_items': 20, 'subreddit_limit': 2}
        }
        
        # Start tasks
        news_task = fetch_and_process_news_articles.delay(news_config, async_processing=True)
        social_task = fetch_and_process_social_posts.delay(social_config, async_processing=True)
        
        result = {
            'status': 'scheduled',
            'news_task_id': news_task.id,
            'social_task_id': social_task.id,
            'timestamp': timezone.now().isoformat()
        }
        
        # Notify scheduled fetch queued
        notify_fetch_completed('scheduled_fetch', 'scheduled', result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in scheduled_content_fetch: {e}")
        notify_fetch_failed('scheduled_fetch', 'scheduled', str(e))
        return {'status': 'error', 'error': str(e)}


@shared_task
def update_topic_model():
    """Daily task to update topic model with recent documents"""
    try:
        from myapp.services.mongo_manager import get_mongo_manager
        
        mongo = get_mongo_manager()
        topic_modeler = get_topic_modeler()
        
        # Get documents from last 7 days
        documents = mongo.get_recent_documents_for_topics(days_back=7, limit=5000)
        
        if len(documents) < 100:
            return {'status': 'insufficient_data', 'count': len(documents)}
        
        # Extract texts and IDs
        texts = [doc.get('title', '') + ' ' + doc.get('content', '') for doc in documents]
        doc_ids = [str(doc.get('_id')) for doc in documents]
        
        # Fit/update model
        result = topic_modeler.fit(texts, doc_ids)
        
        # Save model
        topic_modeler.save_model('models/topic_model_latest')
        
        return result
        
    except Exception as e:
        logger.error(f"Error updating topic model: {e}")
        return {'status': 'error', 'error': str(e)}


@shared_task
def detect_trending_topics():
    """Hourly task to detect trending topics and spikes"""
    try:
        topic_modeler = get_topic_modeler()
        hashtag_analyzer = get_hashtag_analyzer()
        
        # Get trending topics
        trending_topics = topic_modeler.get_trending_topics(hours_back=24)
        
        # Detect spikes
        spikes = topic_modeler.detect_topic_spikes(threshold_multiplier=3.0)
        
        # Get trending hashtags
        trending_hashtags = hashtag_analyzer.get_trending_hashtags(limit=20)
        
        result = {
            'trending_topics': [t.name for t in trending_topics[:10]],
            'topic_spikes': spikes[:5],
            'trending_hashtags': [h.item for h in trending_hashtags[:10]],
            'detected_at': timezone.now().isoformat()
        }
        
        # Cache trending data
        cache.set('trending_data', result, timeout=3600)
        
        return result
        
    except Exception as e:
        logger.error(f"Error detecting trends: {e}")
        return {'status': 'error', 'error': str(e)}


@shared_task
def assign_topics_to_new_content():
    """Task to assign topics to newly processed content"""
    try:
        from myapp.services.mongo_manager import get_mongo_manager
        
        mongo = get_mongo_manager()
        topic_modeler = get_topic_modeler()
        
        # Get content without topic assignments
        unassigned = mongo.get_content_without_topics(limit=100)
        
        if not unassigned:
            return {'status': 'no_new_content'}
        
        texts = [doc.get('title', '') + ' ' + doc.get('content', '') for doc in unassigned]
        doc_ids = [str(doc.get('_id')) for doc in unassigned]
        
        # Assign topics
        assignments = topic_modeler.transform(texts, doc_ids)
        
        # Update MongoDB with assignments
        updated = 0
        for assignment in assignments:
            success = mongo.update_document_topics(
                assignment.document_id,
                {
                    'primary_topic': assignment.primary_topic,
                    'topic_name': assignment.topic_name,
                    'topic_probability': assignment.topic_probability
                }
            )
            if success:
                updated += 1
        
        return {'status': 'success', 'assigned': updated}
        
    except Exception as e:
        logger.error(f"Error assigning topics: {e}")
        return {'status': 'error', 'error': str(e)}



@shared_task
def persist_trending_data():
    """
    Hourly task to persist trending data to PostgreSQL
    Uses proper Django model field names
    """
    try:
        from myapp.models import Trendinghashtag, Trendingkeyword, Trendingtopic
        from myapp.services.content.hashtag_analyzer import get_hashtag_analyzer
        from myapp.services.content.topic_modeler import get_topic_modeler
        
        hashtag_analyzer = get_hashtag_analyzer()
        topic_modeler = get_topic_modeler()
        now = timezone.now()
        
        hashtags_saved = 0
        keywords_saved = 0
        topics_saved = 0
        
        # ====================================================================
        # FIXED: Persist trending hashtags with correct field names
        # ====================================================================
        try:
            trending_hashtags = hashtag_analyzer.get_trending_hashtags(limit=50)
            
            for item in trending_hashtags:
                stats = hashtag_analyzer.get_hashtag_stats(item.item.lstrip('#'))
                if stats:
                    Trendinghashtag.objects.create(
                        hashtag=item.item,  # Correct field name
                        timestamp=now,      # Correct field name (auto_now_add handles this)
                        count_1h=stats.count_1h,      # Correct field name
                        count_6h=stats.count_6h,      # Correct field name
                        count_24h=stats.count_24h,    # Correct field name
                        velocity=stats.velocity_1h,   # Correct field name
                        avg_sentiment=stats.avg_sentiment,  # Correct field name
                        trend_score=item.trend_score  # Correct field name
                    )
                    hashtags_saved += 1
            
            logger.info(f"Persisted {hashtags_saved} trending hashtags")
            
        except Exception as e:
            logger.error(f"Error persisting hashtags: {e}")
            import traceback
            traceback.print_exc()
        
        # ====================================================================
        # FIXED: Persist trending keywords with correct field names
        # ====================================================================
        try:
            trending_keywords = hashtag_analyzer.get_trending_keywords(limit=50)
            
            for item in trending_keywords:
                stats = hashtag_analyzer.get_keyword_stats(item.item)
                if stats:
                    Trendingkeyword.objects.create(
                        keyword=item.item,  # Correct field name
                        timestamp=now,      # Correct field name
                        count_1h=stats.count_1h,      # Correct field name
                        count_6h=stats.count_6h,      # Correct field name
                        count_24h=stats.count_24h,    # Correct field name
                        velocity=item.velocity,       # Correct field name
                        avg_sentiment=stats.avg_sentiment,  # Correct field name
                        sources=stats.sources if hasattr(stats, 'sources') else {}  # Correct field name (JSONField)
                    )
                    keywords_saved += 1
            
            logger.info(f"Persisted {keywords_saved} trending keywords")
            
        except Exception as e:
            logger.error(f"Error persisting keywords: {e}")
            import traceback
            traceback.print_exc()
        
        # ====================================================================
        # FIXED: Persist trending topics with correct field names
        # ====================================================================
        try:
            trending_topics = topic_modeler.get_trending_topics(hours_back=24)
            spikes = topic_modeler.detect_topic_spikes()
            spike_ids = {s['topic_id'] for s in spikes}
            
            for topic in trending_topics[:20]:
                Trendingtopic.objects.create(
                    topic_id=topic.topic_id,          # Correct field name
                    topic_name=topic.name,            # Correct field name
                    keywords=topic.keywords,          # Correct field name (JSONField)
                    timestamp=now,                    # Correct field name
                    document_count=topic.document_count,  # Correct field name
                    velocity=topic.velocity,          # Correct field name
                    avg_sentiment=topic.avg_sentiment,  # Correct field name
                    is_spike=topic.topic_id in spike_ids  # Correct field name
                )
                topics_saved += 1
            
            logger.info(f"Persisted {topics_saved} trending topics")
            
        except Exception as e:
            logger.error(f"Error persisting topics: {e}")
            import traceback
            traceback.print_exc()
        
        # ====================================================================
        # Cleanup old data (keep 30 days)
        # ====================================================================
        try:
            cutoff = now - timedelta(days=30)
            
            hashtags_deleted = Trendinghashtag.objects.filter(
                timestamp__lt=cutoff
            ).delete()[0]
            
            keywords_deleted = Trendingkeyword.objects.filter(
                timestamp__lt=cutoff
            ).delete()[0]
            
            topics_deleted = Trendingtopic.objects.filter(
                timestamp__lt=cutoff
            ).delete()[0]
            
            logger.info(f"Cleanup: Deleted {hashtags_deleted} hashtags, {keywords_deleted} keywords, {topics_deleted} topics")
            
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
        
        # ====================================================================
        # Return summary
        # ====================================================================
        result = {
            'status': 'success',
            'timestamp': now.isoformat(),
            'saved': {
                'hashtags': hashtags_saved,
                'keywords': keywords_saved,
                'topics': topics_saved
            },
            'cleanup': {
                'hashtags_deleted': hashtags_deleted if 'hashtags_deleted' in locals() else 0,
                'keywords_deleted': keywords_deleted if 'keywords_deleted' in locals() else 0,
                'topics_deleted': topics_deleted if 'topics_deleted' in locals() else 0
            }
        }
        
        logger.info(f"Trending data persistence completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Critical error persisting trending data: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': timezone.now().isoformat()
        }


@shared_task
def cleanup_old_api_usage():
    """Clean up old API usage records"""
    try:
        from myapp.models import APIUsage
        
        # Keep 30 days of data
        cutoff_date = timezone.now() - timedelta(days=30)
        deleted_count = APIUsage.objects.filter(
            timestamp__lt=cutoff_date
        ).delete()[0]
        
        logger.info(f"Cleaned up {deleted_count} old API usage records")
        return f"Cleaned up {deleted_count} old API usage records"
        
    except Exception as e:
        logger.error(f"Error in cleanup_old_api_usage: {e}")
        return f"Error: {e}"
    
@shared_task(queue='maintenance')
def detect_and_report_gaps():
    """Periodic gap detection and reporting - MAINTENANCE ONLY"""
    from myapp.models import SeriesConfig
    from myapp.services.influx_manager import InfluxManager
    
    try:
        influx_manager = InfluxManager()
        active_series = SeriesConfig.objects.filter(
            isactive=1,
            isdeleted=0,
            backfill_completed=True
        )
        
        gaps_detected = 0
        for config in active_series[:10]:  # Limit to avoid overload
            # Check for gaps in last 24 hours
            gaps = influx_manager.detect_gaps(
                config.exchange.name,
                config.market_type,
                config.symbol,
                config.timeframe,
                hours_back=24
            )
            
            if gaps:
                gaps_detected += len(gaps)
                logger.warning(f"Gaps detected in {config.series_key}: {len(gaps)} gaps")
        
        return f"Gap detection completed: {gaps_detected} gaps found"
        
    except Exception as e:
        logger.error(f"Error in gap detection: {e}")
        return f"Error: {e}"
    
