"""
Django Management Command: Run Full Content Pipeline
Fetches → Processes → Analyzes → Stores → Indexes to RAG
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import datetime
import time
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Run complete content pipeline: Fetch → Process → Analyze → Store → RAG Index'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--content-type',
            type=str,
            choices=['all', 'news', 'social'],
            default='all',
            help='Which content type to process (default: all)'
        )
        
        parser.add_argument(
            '--max-items',
            type=int,
            default=20,
            help='Maximum items per source (default: 20)'
        )
        
        parser.add_argument(
            '--min-trust-score',
            type=float,
            default=5.0,
            help='Minimum trust score for RAG indexing (default: 5.0)'
        )
        
        parser.add_argument(
            '--skip-rag',
            action='store_true',
            help='Skip RAG indexing step'
        )
        
        parser.add_argument(
            '--skip-hashtags',
            action='store_true',
            help='Skip hashtag analysis step'
        )
        
        parser.add_argument(
            '--skip-topics',
            action='store_true',
            help='Skip topic modeling step'
        )
        
        parser.add_argument(
            '--async',
            action='store_true',
            dest='use_async',
            help='Use async Celery tasks instead of synchronous processing'
        )
        
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed progress information'
        )
    
    def handle(self, *args, **options):
        content_type = options['content_type']
        max_items = options['max_items']
        min_trust_score = options['min_trust_score']
        skip_rag = options['skip_rag']
        skip_hashtags = options['skip_hashtags']
        skip_topics = options['skip_topics']
        use_async = options['use_async']
        verbose = options['verbose']
        
        start_time = time.time()
        
        self.stdout.write(self.style.SUCCESS("\n" + "="*70))
        self.stdout.write(self.style.SUCCESS("   FULL CONTENT PIPELINE EXECUTION"))
        self.stdout.write(self.style.SUCCESS("="*70 + "\n"))
        
        self.stdout.write(f"Configuration:")
        self.stdout.write(f"   Content Type: {content_type.upper()}")
        self.stdout.write(f"   Max Items per Source: {max_items}")
        self.stdout.write(f"   Min Trust Score for RAG: {min_trust_score}")
        self.stdout.write(f"   Processing Mode: {'ASYNC (Celery)' if use_async else 'SYNC (Immediate)'}")
        self.stdout.write(f"   RAG Indexing: {'DISABLED' if skip_rag else 'ENABLED'}")
        self.stdout.write(f"   Hashtag Analysis: {'DISABLED' if skip_hashtags else 'ENABLED'}")
        self.stdout.write(f"   Topic Modeling: {'DISABLED' if skip_topics else 'ENABLED'}")
        self.stdout.write("")
        
        pipeline_results = {
            'news': None,
            'social': None,
            'rag': None,
            'hashtags': None,
            'topics': None,
            'total_processed': 0,
            'total_approved': 0,
            'total_flagged': 0
        }
        
        try:
            # ===================================================================
            # STEP 1: FETCH AND PROCESS NEWS
            # ===================================================================
            if content_type in ['all', 'news']:
                self.stdout.write(self.style.WARNING("\nSTEP 1: Fetching and Processing NEWS"))
                self.stdout.write("-" * 70)
                
                news_result = self._process_news(max_items, use_async, verbose)
                pipeline_results['news'] = news_result
                
                if news_result:
                    pipeline_results['total_processed'] += news_result.get('total_processed', 0)
                    pipeline_results['total_approved'] += news_result.get('total_approved', 0)
                    pipeline_results['total_flagged'] += news_result.get('total_flagged', 0)
            
            # ===================================================================
            # STEP 2: FETCH AND PROCESS SOCIAL
            # ===================================================================
            if content_type in ['all', 'social']:
                self.stdout.write(self.style.WARNING("\nSTEP 2: Fetching and Processing SOCIAL"))
                self.stdout.write("-" * 70)
                
                social_result = self._process_social(max_items, use_async, verbose)
                pipeline_results['social'] = social_result
                
                if social_result:
                    pipeline_results['total_processed'] += social_result.get('total_processed', 0)
                    pipeline_results['total_approved'] += social_result.get('total_approved', 0)
                    pipeline_results['total_flagged'] += social_result.get('total_flagged', 0)
            
            # ===================================================================
            # STEP 3: RAG INDEXING
            # ===================================================================
            if not skip_rag:
                self.stdout.write(self.style.WARNING("\nSTEP 3: RAG Indexing"))
                self.stdout.write("-" * 70)
                
                rag_result = self._index_to_rag(min_trust_score, use_async, verbose)
                pipeline_results['rag'] = rag_result
            else:
                self.stdout.write(self.style.WARNING("\n⏭️  STEP 3: RAG Indexing SKIPPED"))
            
            # ===================================================================
            # STEP 4: HASHTAG ANALYSIS & PERSISTENCE
            # ===================================================================
            if not skip_hashtags:
                self.stdout.write(self.style.WARNING("\nSTEP 4: Hashtag Analysis & PostgreSQL Persistence"))
                self.stdout.write("-" * 70)
                
                hashtag_result = self._analyze_hashtags(
                    min_trust_score=min_trust_score,
                    hours_back=24,
                    verbose=verbose
                )
                pipeline_results['hashtags'] = hashtag_result
            else:
                self.stdout.write(self.style.WARNING("\n⏭️  STEP 4: Hashtag Analysis SKIPPED"))
            
            # ===================================================================
            # STEP 5: TOPIC MODELING & PERSISTENCE
            # ===================================================================
            if not skip_topics:
                self.stdout.write(self.style.WARNING("\nSTEP 5: Topic Modeling & PostgreSQL Persistence"))
                self.stdout.write("-" * 70)
                
                topic_result = self._analyze_topics(
                    min_trust_score=min_trust_score,
                    hours_back=24,
                    verbose=verbose
                )
                pipeline_results['topics'] = topic_result
            else:
                self.stdout.write(self.style.WARNING("\n⏭️  STEP 5: Topic Modeling SKIPPED"))
            
            # ===================================================================
            # FINAL SUMMARY
            # ===================================================================
            elapsed_time = time.time() - start_time
            
            self.stdout.write(self.style.SUCCESS("\n" + "="*70))
            self.stdout.write(self.style.SUCCESS("   PIPELINE EXECUTION COMPLETE"))
            self.stdout.write(self.style.SUCCESS("="*70))
            
            self.stdout.write(f"\n⏱️  Total Execution Time: {elapsed_time:.2f} seconds")
            
            self.stdout.write(f"\nSUMMARY:")
            self.stdout.write(f"   Total Items Processed: {pipeline_results['total_processed']}")
            self.stdout.write(f"   Total Approved: {pipeline_results['total_approved']}")
            self.stdout.write(f"   Total Flagged: {pipeline_results['total_flagged']}")
            
            if pipeline_results['news']:
                self.stdout.write(f"\n   📰 News Results:")
                for source, data in pipeline_results['news'].get('sources', {}).items():
                    if isinstance(data, dict):
                        self.stdout.write(f"      {source}: {data.get('processed', 0)} items")
            
            if pipeline_results['social']:
                self.stdout.write(f"\n   💬 Social Results:")
                for platform, data in pipeline_results['social'].get('platforms', {}).items():
                    if isinstance(data, dict):
                        self.stdout.write(f"      {platform}: {data.get('processed', 0)} items")
            
            if pipeline_results['rag'] and not skip_rag:
                self.stdout.write(f"\n   🔍 RAG Indexing:")
                self.stdout.write(f"      Items Indexed: {pipeline_results['rag'].get('items_indexed', 0)}")
                self.stdout.write(f"      Status: {pipeline_results['rag'].get('status', 'unknown')}")
            
            if pipeline_results['hashtags'] and not skip_hashtags:
                self.stdout.write(f"\n   #️⃣  Hashtag Analysis:")
                self.stdout.write(f"      Hashtags Tracked: {pipeline_results['hashtags'].get('hashtags_tracked', 0)}")
                self.stdout.write(f"      Keywords Tracked: {pipeline_results['hashtags'].get('keywords_tracked', 0)}")
                self.stdout.write(f"      ✅ Persisted to PostgreSQL:")
                self.stdout.write(f"         - Hashtags: {pipeline_results['hashtags'].get('persisted_hashtags', 0)}")
                self.stdout.write(f"         - Keywords: {pipeline_results['hashtags'].get('persisted_keywords', 0)}")
                if pipeline_results['hashtags'].get('top_hashtag'):
                    self.stdout.write(f"      Top Hashtag: {pipeline_results['hashtags']['top_hashtag']}")
            
            if pipeline_results['topics'] and not skip_topics:
                self.stdout.write(f"\n   📊 Topic Modeling:")
                self.stdout.write(f"      Topics Discovered: {pipeline_results['topics'].get('num_topics', 0)}")
                self.stdout.write(f"      Mode: {pipeline_results['topics'].get('mode', 'N/A')}")
                self.stdout.write(f"      ✅ Persisted to PostgreSQL: {pipeline_results['topics'].get('persisted_topics', 0)} topics")
                if pipeline_results['topics'].get('top_topic'):
                    self.stdout.write(f"      Top Topic: {pipeline_results['topics']['top_topic']}")
            
            self.stdout.write(self.style.SUCCESS("\n✅ Pipeline execution completed successfully!"))
            self.stdout.write("=" * 70 + "\n")
            
            return f"Pipeline completed: {pipeline_results['total_processed']} items processed"
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"\n❌ Pipeline execution failed: {e}"))
            import traceback
            traceback.print_exc()
            raise
    
    def _process_news(self, max_items, use_async, verbose):
        """
        Process news articles through the pipeline
        NO CHANGES NEEDED - Already correct
        """
        from myapp.services.content.integrator_service import ContentIntegrationService
        from myapp.fetchers import (
            fetch_cryptopanic_news,
            fetch_cryptocompare_news,
            fetch_newsapi_articles,
            fetch_messari_news,
            fetch_coindesk_news
        )
        
        service = ContentIntegrationService()
        
        sources = [
            ('cryptopanic', lambda: fetch_cryptopanic_news(max_items=max_items)),
            ('cryptocompare', lambda: fetch_cryptocompare_news(max_items=max_items)),
            ('newsapi', lambda: fetch_newsapi_articles(max_items=max_items)),
            ('messari', lambda: fetch_messari_news(max_items=max_items)),
            ('coindesk', lambda: fetch_coindesk_news(max_items=max_items)),
        ]
         
        results = {
            'sources': {},
            'total_processed': 0,
            'total_approved': 0,
            'total_flagged': 0
        }
        
        for source_name, fetch_func in sources:
            try:
                if verbose:
                    self.stdout.write(f"   Fetching from {source_name}...")
                
                articles = fetch_func()
                
                if articles:
                    if verbose:
                        self.stdout.write(f"      Fetched {len(articles)} raw articles")
                    
                    # Process through integrator
                    result = service.process_news_batch(articles, source_name)
                    
                    results['total_processed'] += result.total_processed
                    results['total_approved'] += result.approved
                    results['total_flagged'] += result.flagged
                    
                    results['sources'][source_name] = {
                        'processed': result.total_processed,
                        'approved': result.approved,
                        'avg_trust': result.average_trust_score
                    }
                    
                    if verbose:
                        self.stdout.write(f"      Processed: {result.total_processed}")
                        self.stdout.write(f"      Avg Trust: {result.average_trust_score:.2f}")
                else:
                    results['sources'][source_name] = {'status': 'no_articles'}
                    if verbose:
                        self.stdout.write(f"      No articles found")
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"      Error: {str(e)[:50]}"))
                results['sources'][source_name] = {'status': 'error', 'error': str(e)}
        
        return results
    
    def _process_social(self, max_items, use_async, verbose):
        """
        Process social posts through the pipeline
        NO CHANGES NEEDED - Already correct
        """
        from myapp.services.content.integrator_service import ContentIntegrationService
        from myapp.fetchers import (
            fetch_reddit_posts,
            fetch_twitter_posts,
            fetch_youtube_videos
        )
        
        service = ContentIntegrationService()
        
        platforms = [
            ('reddit', lambda: fetch_reddit_posts(limit=max_items)),
            ('twitter', lambda: fetch_twitter_posts(max_results=max_items)),
            ('youtube', lambda: fetch_youtube_videos(max_results=max_items)),
        ]
        
        results = {
            'platforms': {},
            'total_processed': 0,
            'total_approved': 0,
            'total_flagged': 0
        }
        
        for platform_name, fetch_func in platforms:
            try:
                if verbose:
                    self.stdout.write(f"   Fetching from {platform_name}...")
                
                posts = fetch_func()
                
                if posts:
                    if verbose:
                        self.stdout.write(f"      Fetched {len(posts)} raw posts")
                    
                    # Process through integrator
                    result = service.process_social_posts_batch(posts, platform_name)
                    
                    results['total_processed'] += result.total_processed
                    results['total_approved'] += result.approved
                    results['total_flagged'] += result.flagged
                    
                    results['platforms'][platform_name] = {
                        'processed': result.total_processed,
                        'approved': result.approved,
                        'avg_trust': result.average_trust_score
                    }
                    
                    if verbose:
                        self.stdout.write(f"      Processed: {result.total_processed}")
                        self.stdout.write(f"      Avg Trust: {result.average_trust_score:.2f}")
                else:
                    results['platforms'][platform_name] = {'status': 'no_posts'}
                    if verbose:
                        self.stdout.write(f"      No posts found")
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"      Error: {str(e)[:50]}"))
                results['platforms'][platform_name] = {'status': 'error', 'error': str(e)}
        
        return results
    
    def _index_to_rag(self, min_trust_score, use_async, verbose):
        """
        Index high-credibility content to RAG
        Use timezone-aware datetime
        """
        try:
            if use_async:
                # Use Celery task
                from myapp.tasks.rag_tasks import scheduled_index_update
                
                if verbose:
                    self.stdout.write(f"   Triggering async RAG indexing (min_trust={min_trust_score})...")
                
                task = scheduled_index_update.delay(
                    hours_back=24,
                    min_trust_score=min_trust_score
                )
                
                if verbose:
                    self.stdout.write(f"      Task ID: {task.id}")
                    self.stdout.write(f"      Waiting for completion...")
                
                result = task.get(timeout=600)  # 10 minute timeout
                
                if verbose:
                    self.stdout.write(f"      RAG indexing completed")
                
                return result
            else:
                # Synchronous indexing
                from myapp.services.rag.rag_service import get_rag_engine
                from myapp.services.mongo_manager import get_mongo_manager
                from django.utils import timezone  # FIX: Import timezone
                
                if verbose:
                    self.stdout.write(f"   Indexing to RAG (min_trust={min_trust_score})...")
                
                rag_engine = get_rag_engine()
                mongo_manager = get_mongo_manager()
                
                # FIX: Use timezone-aware datetime
                cutoff_time = timezone.now() - timedelta(hours=24)
                
                # Get high-credibility content
                articles = mongo_manager.collections['news_articles'].find(
                    {
                        'trust_score': {'$gte': min_trust_score},
                        'created_at': {'$gte': cutoff_time}  # Now timezone-aware
                    }
                ).limit(500)
                
                posts = mongo_manager.collections['social_posts'].find(
                    {
                        'trust_score': {'$gte': min_trust_score},
                        'created_at': {'$gte': cutoff_time}  # Now timezone-aware
                    }
                ).limit(300)
                
                articles = list(articles)
                posts = list(posts)
                
                all_content = []
                for article in articles:
                    article['type'] = 'news'
                    all_content.append(article)
                
                for post in posts:
                    post['type'] = 'social'
                    all_content.append(post)
                
                if verbose:
                    self.stdout.write(f"      Found {len(all_content)} items to index")
                    self.stdout.write(f"      ({len(articles)} articles + {len(posts)} posts)")
                
                if all_content:
                    stats = rag_engine.bulk_index_articles(all_content)
                    rag_engine.save_index()
                    
                    if verbose:
                        self.stdout.write(f"      Indexed {len(all_content)} items")
                    
                    return {
                        'status': 'success',
                        'items_indexed': len(all_content),
                        'articles': len(articles),
                        'posts': len(posts),
                        'stats': stats
                    }
                else:
                    if verbose:
                        self.stdout.write(f"      No items met credibility threshold")
                    
                    return {
                        'status': 'no_content',
                        'items_indexed': 0
                    }
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"      RAG indexing failed: {e}"))
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _analyze_hashtags(self, min_trust_score, hours_back, verbose):
        """
        Analyze hashtags and keywords from stored content
        UPDATED: Now persists trending data to PostgreSQL
        """
        try:
            from myapp.services.content.hashtag_analyzer import get_hashtag_analyzer
            from myapp.services.mongo_manager import get_mongo_manager
            from myapp.models import Trendinghashtag, Trendingkeyword
            from django.utils import timezone
            
            analyzer = get_hashtag_analyzer()
            mongo = get_mongo_manager()
            
            # Use timezone-aware datetime
            cutoff_time = timezone.now() - timedelta(hours=hours_back)
            
            articles = mongo.collections['news_articles'].find({
                'trust_score': {'$gte': min_trust_score},
                'created_at': {'$gte': cutoff_time}
            })
            
            posts = mongo.collections['social_posts'].find({
                'trust_score': {'$gte': min_trust_score},
                'created_at': {'$gte': cutoff_time}
            })
            
            all_content = list(articles) + list(posts)
            
            if verbose:
                self.stdout.write(f"   Analyzing {len(all_content)} high-quality items...")
            
            # Extract and record hashtags
            for item in all_content:
                # Get text
                title = item.get('title', '')
                description = item.get('description', '')
                content = item.get('content') or item.get('text') or item.get('selftext') or ''
                text = f"{title} {description} {content}"
                
                # Get sentiment
                sentiment_analysis = item.get('sentiment_analysis', {})
                sentiment = sentiment_analysis.get('score', 0)
                
                # Get source
                platform = item.get('platform', 'unknown')
                
                # Extract and record
                analyzer.extract_and_record(text, sentiment=sentiment, source=platform)
            
            # Get trending results
            trending_hashtags = analyzer.get_trending_hashtags(limit=20, min_count=1)
            trending_keywords = analyzer.get_trending_keywords(limit=20, min_count=1)
            
            if verbose:
                self.stdout.write(f"      Hashtags tracked: {len(analyzer.hashtag_occurrences)}")
                self.stdout.write(f"      Keywords tracked: {len(analyzer.keyword_occurrences)}")
                
                if trending_hashtags:
                    self.stdout.write(f"\n      Top 5 Hashtags:")
                    for item in trending_hashtags[:5]:
                        sentiment_emoji = "📈" if item.sentiment > 0.2 else "📉" if item.sentiment < -0.2 else "➡️"
                        self.stdout.write(
                            f"         #{item.item}: count={item.count}, "
                            f"velocity={item.velocity:.1f}x, {sentiment_emoji} {item.sentiment:.2f}"
                        )
            
            # ===================================================================
            # PERSIST TO POSTGRESQL
            # ===================================================================
            hashtag_count = 0
            keyword_count = 0
            
            if verbose:
                self.stdout.write(f"\n      💾 Persisting to PostgreSQL...")
            
            # Persist hashtags
            for item in trending_hashtags:
                try:
                    hashtag_name = item.item.lstrip('#')
                    stats = analyzer.get_hashtag_stats(hashtag_name)
                    if stats:
                        Trendinghashtag.objects.create(
                            hashtag=item.item,
                            count_1h=stats.count_1h,
                            count_6h=stats.count_6h,
                            count_24h=stats.count_24h,
                            velocity=stats.velocity_1h,
                            avg_sentiment=stats.avg_sentiment,
                            trend_score=item.trend_score
                        )
                        hashtag_count += 1
                except Exception as e:
                    if verbose:
                        self.stdout.write(self.style.WARNING(f"         ⚠️  Error persisting hashtag {item.item}: {e}"))
            
            # Persist keywords
            for item in trending_keywords:
                try:
                    stats = analyzer.get_keyword_stats(item.item)
                    if stats:
                        Trendingkeyword.objects.create(
                            keyword=item.item,
                            count_1h=stats.count_1h,
                            count_6h=stats.count_6h,
                            count_24h=stats.count_24h,
                            velocity=item.velocity,
                            avg_sentiment=stats.avg_sentiment,
                            sources=list(stats.sources) if stats.sources else []
                        )
                        keyword_count += 1
                except Exception as e:
                    if verbose:
                        self.stdout.write(self.style.WARNING(f"         ⚠️  Error persisting keyword {item.item}: {e}"))
            
            if verbose:
                self.stdout.write(f"      ✅ Persisted {hashtag_count} hashtags and {keyword_count} keywords to PostgreSQL")
            
            return {
                'status': 'success',
                'hashtags_tracked': len(analyzer.hashtag_occurrences),
                'keywords_tracked': len(analyzer.keyword_occurrences),
                'trending_hashtags_count': len(trending_hashtags),
                'trending_keywords_count': len(trending_keywords),
                'persisted_hashtags': hashtag_count,
                'persisted_keywords': keyword_count,
                'top_hashtag': trending_hashtags[0].item if trending_hashtags else None,
                'top_keyword': trending_keywords[0].item if trending_keywords else None
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"      Hashtag analysis failed: {e}"))
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_topics(self, min_trust_score, hours_back, verbose):
        """
        Perform topic modeling on stored content
        UPDATED: Now persists trending topics to PostgreSQL
        """
        try:
            from myapp.services.content.topic_modeler import get_topic_modeler
            from myapp.services.mongo_manager import get_mongo_manager
            from myapp.models import Trendingtopic
            from django.utils import timezone 
            
            modeler = get_topic_modeler()
            mongo = get_mongo_manager()
            
            # Get recent high-quality content
            cutoff_time = timezone.now() - timedelta(hours=hours_back)
            
            articles = mongo.collections['news_articles'].find({
                'trust_score': {'$gte': min_trust_score},
                'created_at': {'$gte': cutoff_time}
            })
            
            posts = mongo.collections['social_posts'].find({
                'trust_score': {'$gte': min_trust_score},
                'created_at': {'$gte': cutoff_time}
            })
            
            all_content = list(articles) + list(posts)
            
            if verbose:
                self.stdout.write(f"   Analyzing {len(all_content)} items for topics...")
            
            if len(all_content) < 5:
                if verbose:
                    self.stdout.write(f"      ⚠️  Need at least 5 documents (have {len(all_content)})")
                return {'status': 'insufficient_data', 'documents_provided': len(all_content)}
            
            # Prepare documents
            documents = []
            doc_ids = []
            
            for item in all_content:
                title = item.get('title', '')
                description = item.get('description', '')
                content = item.get('content') or item.get('text') or item.get('selftext') or ''
                text = f"{title} {description} {content}"
                
                if text.strip():
                    documents.append(text)
                    doc_ids.append(str(item.get('_id')))
            
            if len(documents) < 5:
                if verbose:
                    self.stdout.write(f"      ⚠️  Need at least 5 documents with text (have {len(documents)})")
                return {'status': 'insufficient_text_data', 'documents_with_text': len(documents)}
            
            # Fit model
            result = modeler.fit(documents, doc_ids)
            
            if result['status'] == 'success':
                # Record topic occurrences
                for i, doc_id in enumerate(doc_ids):
                    item = all_content[i]
                    sentiment_analysis = item.get('sentiment_analysis', {})
                    sentiment = sentiment_analysis.get('score', 0)
                    
                    # Get topic assignment
                    for assignment in result.get('topic_assignments', []):
                        if assignment['document_id'] == doc_id:
                            modeler.record_topic_occurrence(
                                assignment['topic_id'],
                                sentiment=sentiment,
                                timestamp=timezone.now()
                            )
                            break
                
                # Get trending topics
                trending = modeler.get_trending_topics(hours_back=hours_back)
                spikes = modeler.detect_topic_spikes()
                spike_ids = {s['topic_id'] for s in spikes}
                
                if verbose:
                    self.stdout.write(f"      Topics discovered: {result['num_topics']}")
                    if trending:
                        self.stdout.write(f"\n      Top 3 Topics:")
                        for topic in trending[:3]:
                            self.stdout.write(f"         Topic {topic.topic_id}: {', '.join(topic.keywords[:5])}")
                
                # ===================================================================
                # PERSIST TO POSTGRESQL
                # ===================================================================
                topic_count = 0
                
                if verbose:
                    self.stdout.write(f"\n      💾 Persisting to PostgreSQL...")
                
                for topic in trending[:10]:
                    try:
                        Trendingtopic.objects.create(
                            topic_id=topic.topic_id,
                            topic_name=topic.name,
                            keywords=topic.keywords if topic.keywords else [],
                            document_count=topic.document_count,
                            velocity=topic.velocity,
                            avg_sentiment=topic.avg_sentiment,
                            is_spike=topic.topic_id in spike_ids
                        )
                        topic_count += 1
                    except Exception as e:
                        if verbose:
                            self.stdout.write(self.style.WARNING(f"         ⚠️  Error persisting topic {topic.topic_id}: {e}"))
                
                if verbose:
                    self.stdout.write(f"      ✅ Persisted {topic_count} topics to PostgreSQL")
                
                return {
                    'status': 'success',
                    'num_topics': result['num_topics'],
                    'mode': result.get('mode', 'unknown'),
                    'documents_analyzed': len(documents),
                    'trending_topics_count': len(trending),
                    'persisted_topics': topic_count,
                    'top_topic': trending[0].name if trending else None
                }
            else:
                if verbose:
                    self.stdout.write(f"      ⚠️  {result.get('message', 'Failed')}")
                return result
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"      Topic modeling failed: {e}"))
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'error': str(e)}