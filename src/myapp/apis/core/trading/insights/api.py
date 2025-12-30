from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from datetime import datetime, timedelta
from dateutil.parser import parse as dt_parse
from django.utils import timezone
from datetime import timezone as timezoneDt
import logging

logger = logging.getLogger(__name__)

 
# CORRECTED Market Insights APIs
# Now queries PostgreSQL trending tables instead of in-memory services

class MarketSummaryAPI(APIView):
    """
    Comprehensive market summary combining all data sources
    Uses PostgreSQL for trending data and correct sentiment fields
    """
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get comprehensive market summary with sentiment, trends, and key events.",
        manual_parameters=[
            openapi.Parameter('hours', openapi.IN_QUERY, 
                description="Hours of data to analyze", 
                type=openapi.TYPE_INTEGER, default=24),
            openapi.Parameter('coins', openapi.IN_QUERY, 
                description="Comma-separated coin symbols (BTC,ETH,SOL)", 
                type=openapi.TYPE_STRING),
        ],
        responses={200: "Market summary"},
        tags=['Market Insights']
    )
    def get(self, request):
        try:
            from myapp.services.mongo_manager import get_mongo_manager
            from myapp.models import Trendinghashtag, Trendingtopic  # ✅ FIX: Use PostgreSQL
            
            hours = int(request.GET.get('hours', 24))
            coins_param = request.GET.get('coins', 'BTC,ETH,SOL,XRP,ADA')
            target_coins = [c.strip().upper() for c in coins_param.split(',')]
            
            mongo_manager = get_mongo_manager()
            
            # 1. Get content with correct parameter names
            articles = mongo_manager.get_high_credibility_articles(
                trust_score_threshold=5.0,
                limit=200, 
                hours_back=hours
            )
            
            social_posts = mongo_manager.get_high_credibility_social_posts(
                trust_score_threshold=5.0,
                limit=200, 
                hours_back=hours
            )
            
            # 2. Calculate aggregated sentiment
            all_sentiments = []
            coin_sentiments = {coin: [] for coin in target_coins}
            
            for article in articles:
                sentiment = article.get('sentiment_analysis', {})
                
                # Skip empty sentiments
                if not sentiment or 'insufficient_content' in sentiment.get('flags', []):
                    continue
                
                # ✅ Use 'score' field
                sentiment_score = sentiment.get('score', 0)
                all_sentiments.append(sentiment_score)
                
                # Check which coins are mentioned
                entities = article.get('extracted_entities', {})
                cryptos = entities.get('cryptocurrencies', [])
                
                # Fallback for backward compatibility
                if not cryptos:
                    cryptos = article.get('crypto_relevance', {}).get('mentioned_cryptocurrencies', [])
                if not cryptos:
                    cryptos = article.get('crypto_analysis', {}).get('mentioned_cryptocurrencies', [])
                
                for coin in target_coins:
                    if coin in cryptos or coin.lower() in article.get('title', '').lower():
                        coin_sentiments[coin].append(sentiment_score)
            
            for post in social_posts:
                sentiment = post.get('sentiment_analysis', {})
                
                # Skip empty sentiments
                if not sentiment or 'insufficient_content' in sentiment.get('flags', []):
                    continue
                
                sentiment_score = sentiment.get('score', 0)
                all_sentiments.append(sentiment_score)
                
                entities = post.get('extracted_entities', {})
                cryptos = entities.get('cryptocurrencies', [])
                
                # Fallback
                if not cryptos:
                    cryptos = post.get('crypto_relevance', {}).get('mentioned_cryptocurrencies', [])
                
                for coin in target_coins:
                    if coin in cryptos or coin.lower() in post.get('content', '').lower():
                        coin_sentiments[coin].append(sentiment_score)
            
            # 3. Calculate market sentiment index
            avg_sentiment = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0
            market_sentiment_label = self._get_sentiment_label(avg_sentiment)
            
            # 4. ✅ FIX: Get trending topics from PostgreSQL
            time_threshold = timezone.now() - timedelta(hours=hours)
            
            trending_hashtags = Trendinghashtag.objects.filter(
                timestamp__gte=time_threshold
            ).order_by('-trend_score')[:10]
            
            trending_topics_db = Trendingtopic.objects.filter(
                timestamp__gte=time_threshold
            ).order_by('-velocity')[:5]
            
            # Format trending data
            trending_data = {
                'hashtags': [
                    {
                        'tag': h.hashtag,
                        'trend_score': round(h.trend_score, 2),
                        'count': h.count_24h,
                        'sentiment': round(h.avg_sentiment, 3)
                    }
                    for h in trending_hashtags
                ],
                'topics': [
                    {
                        'id': t.topic_id,
                        'name': t.topic_name,
                        'keywords': t.keywords[:5],
                        'velocity': round(t.velocity, 2),
                        'is_spike': t.is_spike
                    }
                    for t in trending_topics_db
                ]
            }
            
            # 5. Get key events (high-trust, high-impact content)
            key_events = self._extract_key_events(articles, social_posts)
            
            # 6. Calculate per-coin sentiment
            coin_analysis = {}
            for coin, scores in coin_sentiments.items():
                if scores:
                    avg = sum(scores) / len(scores)
                    coin_analysis[coin] = {
                        'sentiment_score': round(avg, 3),
                        'sentiment_label': self._get_sentiment_label(avg),
                        'mention_count': len(scores),
                        'trend': self._calculate_trend(scores)
                    }
            
            # 7. Source breakdown
            source_breakdown = {
                'news_articles': len(articles),
                'social_posts': len(social_posts),
                'reddit': sum(1 for p in social_posts if p.get('platform') == 'reddit'),
                'twitter': sum(1 for p in social_posts if p.get('platform') == 'twitter'),
                'youtube': sum(1 for p in social_posts if p.get('platform') == 'youtube'),
            }
            
            return Response({
                'summary': {
                    'market_sentiment': {
                        'score': round(avg_sentiment, 3),
                        'label': market_sentiment_label,
                        'confidence': self._calculate_confidence(all_sentiments),
                        'sample_size': len(all_sentiments)
                    },
                    'time_period': f'Last {hours} hours',
                    'generated_at': timezone.now().isoformat()
                },
                'coin_analysis': coin_analysis,
                'trending_topics': trending_data['topics'],  # ✅ From PostgreSQL
                'trending_hashtags': trending_data['hashtags'],  # ✅ From PostgreSQL
                'key_events': key_events[:5],
                'source_breakdown': source_breakdown,
                'market_signals': self._generate_market_signals(avg_sentiment, coin_analysis, trending_data)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error generating market summary: {e}")
            import traceback
            traceback.print_exc()
            return Response({
                'error': 'Failed to generate market summary',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _get_sentiment_label(self, score):
        if score >= 0.3:
            return 'very_bullish'
        elif score >= 0.1:
            return 'bullish'
        elif score <= -0.3:
            return 'very_bearish'
        elif score <= -0.1:
            return 'bearish'
        return 'neutral'
    
    def _calculate_confidence(self, scores):
        if len(scores) < 10:
            return 'low'
        elif len(scores) < 50:
            return 'medium'
        return 'high'
    
    def _calculate_trend(self, scores):
        if len(scores) < 3:
            return 'insufficient_data'
        mid = len(scores) // 2
        first_half = sum(scores[:mid]) / mid if mid > 0 else 0
        second_half = sum(scores[mid:]) / (len(scores) - mid) if (len(scores) - mid) > 0 else 0
        diff = second_half - first_half
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'declining'
        return 'stable'
    
    def _extract_key_events(self, articles, social_posts):
        """Extract high-impact events"""
        events = []
        
        # High-trust news with strong sentiment
        for article in sorted(articles, key=lambda x: x.get('trust_score', 0), reverse=True)[:10]:
            sentiment = article.get('sentiment_analysis', {})
            
            # Skip empty sentiments
            if not sentiment or 'insufficient_content' in sentiment.get('flags', []):
                continue
            
            sentiment_score = sentiment.get('score', 0)
            
            if abs(sentiment_score) > 0.3 or article.get('trust_score', 0) > 8:
                events.append({
                    'type': 'news',
                    'title': article.get('title', '')[:100],
                    'source': article.get('source', {}).get('title', 'Unknown'),
                    'trust_score': round(article.get('trust_score', 0), 2),
                    'sentiment_score': round(sentiment_score, 3),
                    'sentiment_label': sentiment.get('label', 'neutral'),
                    'published_at': article.get('published_at'),
                    'url': article.get('url')
                })
        
        # Viral social posts
        for post in social_posts:
            engagement = post.get('engagement_metrics', {})
            
            # Platform-specific viral thresholds
            platform = post.get('platform', '')
            is_viral = False
            if platform == 'reddit' and engagement.get('score', 0) > 1000:
                is_viral = True
            elif platform == 'twitter' and engagement.get('like_count', 0) > 5000:
                is_viral = True
            elif platform == 'youtube' and engagement.get('view_count', 0) > 50000:
                is_viral = True
            
            if is_viral:
                events.append({
                    'type': 'social',
                    'platform': platform,
                    'title': post.get('title', post.get('content', ''))[:100],
                    'author': post.get('author_username', 'Unknown'),
                    'trust_score': round(post.get('trust_score', 0), 2),
                    'engagement': engagement.get('total_engagement', 0),
                    'published_at': post.get('published_at'),
                    'url': post.get('url')
                })
        
        # Sort by trust score
        events.sort(key=lambda x: x.get('trust_score', 0), reverse=True)
        return events
    
    def _generate_market_signals(self, avg_sentiment, coin_analysis, trending_data):
        """Generate actionable market signals"""
        signals = []
        
        # Overall market signal
        if avg_sentiment > 0.2:
            signals.append({
                'type': 'market_sentiment',
                'signal': 'bullish',
                'strength': 'strong' if avg_sentiment > 0.4 else 'moderate',
                'description': 'Overall market sentiment is positive'
            })
        elif avg_sentiment < -0.2:
            signals.append({
                'type': 'market_sentiment',
                'signal': 'bearish',
                'strength': 'strong' if avg_sentiment < -0.4 else 'moderate',
                'description': 'Overall market sentiment is negative'
            })
        
        # Per-coin signals
        for coin, data in coin_analysis.items():
            if data.get('mention_count', 0) > 10:
                if data['sentiment_score'] > 0.3:
                    signals.append({
                        'type': 'coin_sentiment',
                        'coin': coin,
                        'signal': 'bullish',
                        'mentions': data['mention_count'],
                        'description': f'{coin} has strong positive sentiment with {data["mention_count"]} mentions'
                    })
                elif data['sentiment_score'] < -0.3:
                    signals.append({
                        'type': 'coin_sentiment',
                        'coin': coin,
                        'signal': 'bearish',
                        'mentions': data['mention_count'],
                        'description': f'{coin} has strong negative sentiment with {data["mention_count"]} mentions'
                    })
        
        # Trending topic signals
        for topic in trending_data.get('topics', []):
            if topic.get('is_spike'):
                signals.append({
                    'type': 'trending_topic',
                    'signal': 'attention',
                    'topic': topic['name'],
                    'keywords': topic['keywords'],
                    'description': f'Spiking topic: {topic["name"]}'
                })
        
        return signals


class NarrativeAnalysisAPI(APIView):
    """
    Analyze dominant narratives in the market
    Uses PostgreSQL for topics and correct sentiment fields
    """
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Analyze dominant narratives and themes in crypto market.",
        manual_parameters=[
            openapi.Parameter('hours', openapi.IN_QUERY, 
                description="Hours of data to analyze", 
                type=openapi.TYPE_INTEGER, default=24),
        ],
        responses={200: "Narrative analysis"},
        tags=['Market Insights']
    )
    def get(self, request):
        try:
            from myapp.services.mongo_manager import get_mongo_manager
            from myapp.models import Trendingtopic  # ✅ FIX: Use PostgreSQL
            from myapp.services.rag.rag_service import get_rag_engine
            
            hours = int(request.GET.get('hours', 24))
            
            mongo_manager = get_mongo_manager()
            rag_engine = get_rag_engine()
            
            # Get content
            articles = mongo_manager.get_high_credibility_articles(
                trust_score_threshold=5.0,
                limit=300, 
                hours_back=hours
            )
            posts = mongo_manager.get_high_credibility_social_posts(
                trust_score_threshold=5.0,
                limit=300, 
                hours_back=hours
            )
            
            # Extract all text content with sentiment
            documents = []
            for article in articles:
                sentiment = article.get('sentiment_analysis', {})
                
                # Skip empty sentiments
                if not sentiment or 'insufficient_content' in sentiment.get('flags', []):
                    continue
                
                documents.append({
                    'text': f"{article.get('title', '')} {article.get('description', '')}",
                    'type': 'news',
                    'sentiment_score': sentiment.get('score', 0)
                })
            
            for post in posts:
                sentiment = post.get('sentiment_analysis', {})
                
                # Skip empty sentiments
                if not sentiment or 'insufficient_content' in sentiment.get('flags', []):
                    continue
                
                documents.append({
                    'text': f"{post.get('title', '')} {post.get('content', '')}",
                    'type': 'social',
                    'sentiment_score': sentiment.get('score', 0)
                })
            
            # ✅ FIX: Get topics from PostgreSQL
            time_threshold = timezone.now() - timedelta(hours=hours)
            
            trending_topics_db = Trendingtopic.objects.filter(
                timestamp__gte=time_threshold
            ).order_by('-velocity')[:10]
            
            # Use RAG to generate narrative summaries
            narratives = []
            for topic in trending_topics_db:
                topic_keywords = topic.keywords if topic.keywords else []
                
                if topic_keywords:
                    # Build query for RAG
                    query = f"Summarize the key discussions and news about {', '.join(topic_keywords[:3])} in cryptocurrency markets"
                    
                    try:
                        result = rag_engine.generate_answer(query)
                        
                        # Calculate sentiment for this topic
                        topic_sentiment = self._topic_sentiment(documents, topic_keywords)
                        
                        narratives.append({
                            'topic_id': topic.topic_id,
                            'topic_name': topic.topic_name,
                            'keywords': topic_keywords[:10],
                            'summary': result.answer if hasattr(result, 'answer') else result.get('answer', ''),
                            'document_count': topic.document_count,
                            'velocity': round(topic.velocity, 2),
                            'sentiment_score': round(topic_sentiment, 3),
                            'sentiment_label': self._get_sentiment_label(topic_sentiment),
                            'is_spike': topic.is_spike,
                            'avg_sentiment': round(topic.avg_sentiment, 3)
                        })
                    except Exception as e:
                        logger.warning(f"Failed to generate narrative for topic {topic.topic_name}: {e}")
                        # Add topic without RAG summary
                        narratives.append({
                            'topic_id': topic.topic_id,
                            'topic_name': topic.topic_name,
                            'keywords': topic_keywords[:10],
                            'summary': f'Trending topic with keywords: {", ".join(topic_keywords[:5])}',
                            'document_count': topic.document_count,
                            'velocity': round(topic.velocity, 2),
                            'sentiment_score': round(topic.avg_sentiment, 3),
                            'sentiment_label': self._get_sentiment_label(topic.avg_sentiment),
                            'is_spike': topic.is_spike
                        })
            
            # Identify emerging vs declining narratives
            trending_up = [n for n in narratives if n.get('sentiment_score', 0) > 0.1]
            trending_down = [n for n in narratives if n.get('sentiment_score', 0) < -0.1]
            
            # Identify spiking narratives
            spiking = [n for n in narratives if n.get('is_spike', False)]
            
            return Response({
                'narratives': narratives,
                'trending_up': trending_up[:3],
                'trending_down': trending_down[:3],
                'spiking': spiking[:3],
                'document_count': len(documents),
                'time_period': f'Last {hours} hours',
                'generated_at': timezone.now().isoformat()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error analyzing narratives: {e}")
            import traceback
            traceback.print_exc()
            return Response({
                'error': 'Failed to analyze narratives',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _topic_sentiment(self, documents, keywords):
        """Calculate average sentiment for documents containing topic keywords"""
        relevant_docs = []
        for doc in documents:
            text_lower = doc['text'].lower()
            if any(kw.lower() in text_lower for kw in keywords):
                relevant_docs.append(doc['sentiment_score'])
        
        return sum(relevant_docs) / len(relevant_docs) if relevant_docs else 0
    
    def _get_sentiment_label(self, score):
        if score >= 0.3:
            return 'very_bullish'
        elif score >= 0.1:
            return 'bullish'
        elif score <= -0.3:
            return 'very_bearish'
        elif score <= -0.1:
            return 'bearish'
        return 'neutral'
    
    

def _normalize_pubdate(val):
    if isinstance(val, int):
        try:
            return datetime.utcfromtimestamp(val).replace(tzinfo=timezoneDt.utc)
        except Exception:
            return datetime.min.replace(tzinfo=timezoneDt.utc)
    elif isinstance(val, str):
        try:
            return dt_parse(val)
        except Exception:
            return datetime.min.replace(tzinfo=timezoneDt.utc)
    elif isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezoneDt.utc)
    return datetime.min.replace(tzinfo=timezoneDt.utc)

class CoinInsightsAPI(APIView):
    """
    Deep insights for a specific cryptocurrency
    Uses trust_score_threshold and correct field access patterns
    """
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get comprehensive insights for a specific cryptocurrency.",
        manual_parameters=[
            openapi.Parameter('symbol', openapi.IN_PATH, 
                description="Coin symbol (BTC, ETH, SOL)", 
                type=openapi.TYPE_STRING, required=True),
            openapi.Parameter('hours', openapi.IN_QUERY, 
                description="Hours of data to analyze", 
                type=openapi.TYPE_INTEGER, default=48),
        ],
        responses={200: "Coin insights"},
        tags=['Market Insights']
    )
    def get(self, request, symbol):
        try:
            from myapp.services.mongo_manager import get_mongo_manager
            from myapp.services.rag.rag_service import get_rag_engine
            from myapp.services.rag.knowledge_graph import get_knowledge_graph
            
            symbol = symbol.upper()
            hours = int(request.GET.get('hours', 48))
            
            mongo_manager = get_mongo_manager()
            rag_engine = get_rag_engine()
            kg = get_knowledge_graph()
            
            # Use trust_score_threshold instead of trust_score_threshold
            articles = mongo_manager.get_high_credibility_articles(
                trust_score_threshold=4.0,  # CORRECT PARAMETER NAME
                limit=500, 
                hours_back=hours
            )
            
            # Use trust_score_threshold instead of trust_score_threshold
            social_posts = mongo_manager.get_high_credibility_social_posts(
                trust_score_threshold=4.0,  # CORRECT PARAMETER NAME
                limit=500, 
                hours_back=hours
            )
            
            # Filter for this coin - USES extracted_entities with fallback
            coin_articles = []
            coin_posts = []
            
            for article in articles:
                # Check extracted_entities first (NEW), then fallbacks for old data
                entities = article.get('extracted_entities', {})
                cryptos = entities.get('cryptocurrencies', [])
                if not cryptos:
                    cryptos = article.get('crypto_relevance', {}).get('mentioned_cryptocurrencies', [])
                    if not cryptos:
                        cryptos = article.get('crypto_analysis', {}).get('mentioned_cryptocurrencies', [])
                
                content = f"{article.get('title', '')} {article.get('description', '')}".lower()
                if symbol in cryptos or symbol.lower() in content:
                    coin_articles.append(article)
            
            for post in social_posts:
                # Check extracted_entities first
                entities = post.get('extracted_entities', {})
                cryptos = entities.get('cryptocurrencies', [])
                if not cryptos:
                    cryptos = post.get('crypto_relevance', {}).get('mentioned_cryptocurrencies', [])
                
                content = f"{post.get('title', '')} {post.get('content', '')}".lower()
                if symbol in cryptos or symbol.lower() in content:
                    coin_posts.append(post)
            
            # 2. Calculate sentiment over time
            sentiment_timeline = self._calculate_sentiment_timeline(
                coin_articles + coin_posts, hours
            )
            
            # 3. Get knowledge graph context
            kg_context = kg.get_entity_context(symbol)
            
            # 4. Get related entities
            related_entities = self._extract_related_entities(coin_articles, coin_posts, symbol)
            
            # 5. Get key narratives using RAG
            narratives = self._extract_narratives(rag_engine, symbol, hours)
            
            # 6. Source analysis
            source_analysis = self._analyze_sources(coin_articles, coin_posts)
            
            # 7. Calculate buzz score
            buzz_score = self._calculate_buzz_score(
                len(coin_articles), len(coin_posts), hours
            )
            
            return Response({
                'coin': symbol,
                'time_period': f'Last {hours} hours',
                'generated_at': timezone.now().isoformat(),
                
                'overview': {
                    'total_mentions': len(coin_articles) + len(coin_posts),
                    'news_mentions': len(coin_articles),
                    'social_mentions': len(coin_posts),
                    'buzz_score': buzz_score,
                    'buzz_trend': self._calculate_buzz_trend(coin_articles, coin_posts, hours)
                },
                
                'sentiment': {
                    'current': self._get_current_sentiment(coin_articles, coin_posts),
                    'timeline': sentiment_timeline,
                    'by_source': source_analysis.get('sentiment_by_source', {})
                },
                
                'narratives': narratives,
                
                'related_entities': related_entities,
                
                'knowledge_graph': {
                    'entity_type': kg_context.get('entity', {}).get('type', 'cryptocurrency'),
                    'related_events': kg_context.get('recent_events', [])[:5],
                    'connections': kg_context.get('related_entities', [])[:10]
                },
                
                'top_content': {
                    'news': self._format_top_articles(coin_articles[:5]),
                    'social': self._format_top_posts(coin_posts[:5])
                },
                
                'source_breakdown': source_analysis.get('breakdown', {})
                
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error generating coin insights for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return Response({
                'error': f'Failed to generate insights for {symbol}',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _calculate_sentiment_timeline(self, content_list, hours):
        """Calculate sentiment over time buckets"""
        now = timezone.now()
        bucket_size = max(1, hours // 12)  # 12 data points
        buckets = {}
        
        for i in range(12):
            bucket_start = now - timedelta(hours=(i + 1) * bucket_size)
            bucket_end = now - timedelta(hours=i * bucket_size)
            bucket_key = bucket_start.strftime('%Y-%m-%d %H:00')
            buckets[bucket_key] = []
        
        for item in content_list:
            pub_date = item.get('published_at')
            if isinstance(pub_date, int):
                try:
                    pub_date = datetime.utcfromtimestamp(pub_date).replace(tzinfo=timezoneDt.utc)
                except Exception:
                    continue
            elif isinstance(pub_date, str):
                try:
                    pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                except Exception:
                    continue
            elif isinstance(pub_date, datetime):
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=timezoneDt.utc)
            else:
                continue
            
            if pub_date:
                for bucket_key in buckets:
                    bucket_start = datetime.strptime(bucket_key, '%Y-%m-%d %H:00').replace(tzinfo=timezoneDt.utc)
                    bucket_end = bucket_start + timedelta(hours=bucket_size)
                    if bucket_start <= pub_date < bucket_end:
                        sentiment = item.get('sentiment_analysis', {})
                        
                        # SKIP empty sentiments
                        if 'insufficient_content' in sentiment.get('flags', []):
                            continue
                        
                        # Use 'score' instead of 'compound_score'
                        buckets[bucket_key].append(sentiment.get('score', 0))
                        break
        
        timeline = []
        for bucket_key in sorted(buckets.keys()):
            scores = buckets[bucket_key]
            timeline.append({
                'timestamp': bucket_key,
                'sentiment': round(sum(scores) / len(scores), 3) if scores else 0,
                'count': len(scores)
            })
        
        return timeline
    
    def _extract_related_entities(self, articles, posts, exclude_symbol):
        """Extract related entities from content - USES extracted_entities"""
        entity_counts = {
            'cryptocurrencies': {},
            'exchanges': {},
            'persons': {},
            'organizations': {}
        }
        
        for item in articles + posts:
            # CORRECT: Uses extracted_entities with proper structure
            entities = item.get('extracted_entities', {})
            
            # Handle list format from text_processor
            for entity_type in entity_counts.keys():
                entities_list = entities.get(entity_type, [])
                if isinstance(entities_list, list):
                    for entity in entities_list:
                        if entity != exclude_symbol:
                            entity_counts[entity_type][entity] = entity_counts[entity_type].get(entity, 0) + 1
        
        # Sort and limit
        result = {}
        for entity_type, counts in entity_counts.items():
            sorted_entities = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
            result[entity_type] = [{'name': e[0], 'count': e[1]} for e in sorted_entities]
        
        return result
    
    def _extract_narratives(self, rag_engine, symbol, hours):
        """Extract key narratives using RAG"""
        try:
            query = f"What are the main news stories and discussions about {symbol} in the last {hours} hours?"
            result = rag_engine.generate_answer(query)
            
            return {
                'summary': result.answer if hasattr(result, 'answer') else result.get('answer', ''),
                'key_themes': self._extract_themes(result.answer if hasattr(result, 'answer') else result.get('answer', '')),
                'sources_used': len(result.sources if hasattr(result, 'sources') else result.get('sources', []))
            }
        except Exception as e:
            logger.warning(f"Failed to extract narratives: {e}")
            return {'summary': '', 'key_themes': [], 'sources_used': 0}
    
    def _extract_themes(self, text):
        """Extract key themes from narrative text"""
        # Simple keyword extraction
        theme_keywords = [
            'regulation', 'adoption', 'price', 'partnership', 'upgrade',
            'security', 'hack', 'ETF', 'institutional', 'DeFi', 'NFT',
            'mining', 'staking', 'airdrop', 'listing', 'delisting'
        ]
        found_themes = []
        text_lower = text.lower()
        for theme in theme_keywords:
            if theme.lower() in text_lower:
                found_themes.append(theme)
        return found_themes[:5]
    
    def _analyze_sources(self, articles, posts):
        """Analyze content by source"""
        breakdown = {
            'news': len(articles),
            'reddit': sum(1 for p in posts if p.get('platform') == 'reddit'),
            'twitter': sum(1 for p in posts if p.get('platform') == 'twitter'),
            'youtube': sum(1 for p in posts if p.get('platform') == 'youtube')
        }
        
        sentiment_by_source = {}
        for source, items in [('news', articles)] + [(p.get('platform'), [p]) for p in posts]:
            if source not in sentiment_by_source:
                sentiment_by_source[source] = []
            for item in items:
                sentiment = item.get('sentiment_analysis', {})
                
                # SKIP empty sentiments
                if 'insufficient_content' in sentiment.get('flags', []):
                    continue
                
                if sentiment:
                    # Use 'score' instead of 'compound_score'
                    sentiment_by_source[source].append(sentiment.get('score', 0))
        
        for source in sentiment_by_source:
            scores = sentiment_by_source[source]
            sentiment_by_source[source] = round(sum(scores) / len(scores), 3) if scores else 0
        
        return {'breakdown': breakdown, 'sentiment_by_source': sentiment_by_source}
    
    def _calculate_buzz_score(self, news_count, social_count, hours):
        """Calculate buzz score (0-100)"""
        # Normalize based on expected volume
        expected_news_per_hour = 2
        expected_social_per_hour = 5
        
        news_ratio = news_count / (expected_news_per_hour * hours)
        social_ratio = social_count / (expected_social_per_hour * hours)
        
        # Weighted average (social has more weight for buzz)
        buzz = (news_ratio * 0.4 + social_ratio * 0.6) * 50
        return min(100, round(buzz, 1))
    
    def _calculate_buzz_trend(self, articles, posts, hours):
        """Calculate if buzz is increasing or decreasing"""
        now = timezone.now()
        mid_point = now - timedelta(hours=hours // 2)
        
        first_half = 0
        second_half = 0
        
        for item in articles + posts:
            pub_date = item.get('published_at')
            if isinstance(pub_date, int):
                try:
                    pub_date = datetime.utcfromtimestamp(pub_date).replace(tzinfo=timezoneDt.utc)
                except Exception:
                    continue
            elif isinstance(pub_date, str):
                try:
                    pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                except Exception:
                    continue
            elif isinstance(pub_date, datetime):
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=timezoneDt.utc)
            else:
                continue
        
            if pub_date:
                if pub_date < mid_point:
                    first_half += 1
                else:
                    second_half += 1
        
        if first_half == 0:
            return 'new_activity'
        
        change = (second_half - first_half) / first_half
        if change > 0.3:
            return 'increasing'
        elif change < -0.3:
            return 'decreasing'
        return 'stable'
    
    def _get_current_sentiment(self, articles, posts):
        """Get current sentiment from recent content"""
        all_content = articles + posts
        all_content.sort(key=lambda x: _normalize_pubdate(x.get('published_at')), reverse=True)
        
        recent = all_content[:20]
        scores = []
        
        for item in recent:
            sentiment = item.get('sentiment_analysis', {})
            
            # SKIP empty sentiments
            if 'insufficient_content' in sentiment.get('flags', []):
                continue
            
            # Use 'score' instead of 'compound_score'
            scores.append(sentiment.get('score', 0))
        
        if not scores:
            return {'score': 0, 'label': 'neutral', 'sample_size': 0}
        
        avg = sum(scores) / len(scores)
        label = 'neutral'
        if avg > 0.2:
            label = 'bullish'
        elif avg > 0.4:
            label = 'very_bullish'
        elif avg < -0.2:
            label = 'bearish'
        elif avg < -0.4:
            label = 'very_bearish'
        
        return {'score': round(avg, 3), 'label': label, 'sample_size': len(scores)}
    
    def _format_top_articles(self, articles):
        """Format top articles for response"""
        return [{
            'title': a.get('title', ''),
            'source': a.get('source', {}).get('title', 'Unknown'),
            'trust_score': round(a.get('trust_score', 0), 2),
            'sentiment': a.get('sentiment_analysis', {}).get('label', 'neutral'),
            'url': a.get('url'),
            'published_at': a.get('published_at')
        } for a in articles]
    
    def _format_top_posts(self, posts):
        """Format top posts for response"""
        return [{
            'title': p.get('title', p.get('content', '')[:100]),
            'platform': p.get('platform', 'unknown'),
            'author': p.get('author_username', 'Unknown'),
            'trust_score': round(p.get('trust_score', 0), 2),
            'sentiment': p.get('sentiment_analysis', {}).get('label', 'neutral'),
            'url': p.get('url'),
            'published_at': p.get('published_at')
        } for p in posts]

class MarketAlertsAPI(APIView):
    """
    Get market alerts and anomalies
    """
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get market alerts including sentiment shifts, unusual activity, and breaking news.",
        manual_parameters=[
            openapi.Parameter('hours', openapi.IN_QUERY, 
                description="Hours to analyze", 
                type=openapi.TYPE_INTEGER, default=6),
            openapi.Parameter('severity', openapi.IN_QUERY, 
                description="Minimum severity (low, medium, high, critical)", 
                type=openapi.TYPE_STRING, default='medium'),
        ],
        responses={200: "Market alerts"},
        tags=['Market Insights']
    )
    def get(self, request):
        try:
            from myapp.services.mongo_manager import get_mongo_manager
            
            hours = int(request.GET.get('hours', 6))
            min_severity = request.GET.get('severity', 'medium')
            
            mongo_manager = get_mongo_manager()
            alerts = []
            
            # 1. Check for sentiment shifts
            sentiment_alerts = self._detect_sentiment_shifts(mongo_manager, hours)
            alerts.extend(sentiment_alerts)
            
            # 2. Check for unusual volume
            volume_alerts = self._detect_volume_anomalies(mongo_manager, hours)
            alerts.extend(volume_alerts)
            
            # 3. Check for high-impact breaking news
            breaking_alerts = self._detect_breaking_news(mongo_manager, hours)
            alerts.extend(breaking_alerts)
            
            # 4. Check for viral social content
            viral_alerts = self._detect_viral_content(mongo_manager, hours)
            alerts.extend(viral_alerts)
            
            # Filter by severity
            severity_order = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            min_level = severity_order.get(min_severity, 2)
            filtered_alerts = [
                a for a in alerts
                if severity_order.get(a.get('severity', 'low'), 1) >= min_level
            ]
            
            # Sort by severity and time
            filtered_alerts.sort(
                key=lambda x: (severity_order.get(x.get('severity'), 0), x.get('timestamp', '')),
                reverse=True
            )
            
            return Response({
                'alerts': filtered_alerts,
                'total': len(filtered_alerts),
                'time_period': f'Last {hours} hours',
                'generated_at': timezone.now().isoformat()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error generating market alerts: {e}")
            return Response({
                'error': 'Failed to generate market alerts',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _detect_sentiment_shifts(self, mongo_manager, hours):
        """Detect significant sentiment shifts"""
        alerts = []
        
        current_articles = mongo_manager.get_high_credibility_articles(
            trust_score_threshold=5.0,
            limit=100, 
            hours_back=hours
        )
        previous_articles = mongo_manager.get_high_credibility_articles(
            trust_score_threshold=5.0,
            limit=100, 
            hours_back=hours * 2
        )
        
        # Filter previous to exclude current period
        cutoff = timezone.now() - timedelta(hours=hours)
        previous_only = []
        
        for a in previous_articles:
            pub_at = a.get('published_at')
            if pub_at:
                try:
                    if isinstance(pub_at, str):
                        pub_dt = datetime.fromisoformat(pub_at.replace('Z', '+00:00'))
                    elif isinstance(pub_at, datetime):
                        pub_dt = pub_at if pub_at.tzinfo else pub_at.replace(tzinfo=timezoneDt.utc)
                    else:
                        continue
                    
                    if pub_dt < cutoff:
                        previous_only.append(a)
                except Exception:
                    continue
        
        current_sentiment = self._avg_sentiment(current_articles)
        previous_sentiment = self._avg_sentiment(previous_only)
        
        shift = current_sentiment - previous_sentiment
        
        if abs(shift) > 0.3:
            severity = 'high' if abs(shift) > 0.5 else 'medium'
            direction = 'positive' if shift > 0 else 'negative'
            
            alerts.append({
                'type': 'sentiment_shift',
                'severity': severity,
                'title': f'Significant {direction} sentiment shift detected',
                'description': f'Market sentiment shifted from {previous_sentiment:.2f} to {current_sentiment:.2f} ({shift:+.2f})',
                'timestamp': timezone.now().isoformat(),
                'data': {
                    'current_sentiment': round(current_sentiment, 3),
                    'previous_sentiment': round(previous_sentiment, 3),
                    'shift': round(shift, 3),
                    'direction': direction
                }
            })
        
        return alerts
    
    def _detect_volume_anomalies(self, mongo_manager, hours):
        """Detect unusual content volume"""
        alerts = []
        
        current_count = len(mongo_manager.get_high_credibility_articles(
            trust_score_threshold=4.0,
            limit=1000, 
            hours_back=hours
        ))
        
        # Compare to typical volume (24h average over past week)
        typical_hourly = 5  # Adjust based on your data
        expected = typical_hourly * hours
        
        if current_count > expected * 2:
            alerts.append({
                'type': 'volume_spike',
                'severity': 'high',
                'title': 'Unusual news volume detected',
                'description': f'{current_count} articles in last {hours}h vs expected {expected}',
                'timestamp': timezone.now().isoformat(),
                'data': {
                    'current_volume': current_count,
                    'expected_volume': expected,
                    'ratio': round(current_count / expected, 2)
                }
            })
        
        return alerts
    
    def _detect_breaking_news(self, mongo_manager, hours):
        """Detect high-impact breaking news"""
        alerts = []
        
        articles = mongo_manager.get_high_credibility_articles(
            trust_score_threshold=8.0,
            limit=20, 
            hours_back=hours
        )
        
        for article in articles:
            sentiment = article.get('sentiment_analysis', {})
            
            # Skip empty/insufficient sentiments
            if not sentiment or 'insufficient_content' in sentiment.get('flags', []):
                continue
            
            sentiment_score = sentiment.get('score', 0)
            sentiment_label = sentiment.get('label', 'neutral')
            trust_score = article.get('trust_score', 0)
            
            # High-trust + strong sentiment = breaking news alert
            if trust_score > 8 and abs(sentiment_score) > 0.5:
                severity = 'critical' if abs(sentiment_score) > 0.7 else 'high'
                
                alerts.append({
                    'type': 'breaking_news',
                    'severity': severity,
                    'title': article.get('title', '')[:100],
                    'description': f'High-credibility news with {sentiment_label} sentiment',
                    'timestamp': article.get('published_at'),
                    'data': {
                        'trust_score': round(trust_score, 2),
                        'sentiment_score': round(sentiment_score, 3),  
                        'sentiment_label': sentiment_label,           
                        'source': article.get('source', {}).get('title', 'Unknown'),
                        'url': article.get('url')
                    }
                })
        
        return alerts
    
    def _detect_viral_content(self, mongo_manager, hours):
        """Detect viral social media content"""
        alerts = []
        
        posts = mongo_manager.get_social_posts(hours_back=hours, limit=100)
        
        for post in posts:
            engagement = post.get('engagement_metrics', {})
            platform = post.get('platform', '')
            
            # Platform-specific viral thresholds
            is_viral = False
            viral_metric = ''
            viral_value = 0
            
            if platform == 'reddit':
                if engagement.get('score', 0) > 5000:
                    is_viral = True
                    viral_metric = 'upvotes'
                    viral_value = engagement.get('score', 0)
            elif platform == 'twitter':
                if engagement.get('like_count', 0) > 10000:
                    is_viral = True
                    viral_metric = 'likes'
                    viral_value = engagement.get('like_count', 0)
            elif platform == 'youtube':
                if engagement.get('view_count', 0) > 100000:
                    is_viral = True
                    viral_metric = 'views'
                    viral_value = engagement.get('view_count', 0)
            
            if is_viral:
                sentiment = post.get('sentiment_analysis', {})
                
                alerts.append({
                    'type': 'viral_content',
                    'severity': 'medium',
                    'title': f'Viral {platform} post detected',
                    'description': f'{viral_value:,} {viral_metric} on {platform}',
                    'timestamp': post.get('published_at'),
                    'data': {
                        'platform': platform,
                        'content': post.get('title', post.get('content', ''))[:100],
                        'author': post.get('author_username', 'Unknown'),
                        'engagement_metric': viral_metric,
                        'engagement_value': viral_value,
                        'trust_score': round(post.get('trust_score', 0), 2),
                        'sentiment_label': sentiment.get('label', 'neutral'),
                        'url': post.get('url')
                    }
                })
        
        return alerts
    
    def _avg_sentiment(self, articles):
        """
        Calculate average sentiment score
        """
        scores = []
        for a in articles:
            sentiment = a.get('sentiment_analysis', {})
            
            # Skip empty sentiments
            if not sentiment or 'insufficient_content' in sentiment.get('flags', []):
                continue
            
            sentiment_score = sentiment.get('score', 0)
            scores.append(sentiment_score)
        
        return sum(scores) / len(scores) if scores else 0


class InfluencerTrackingAPI(APIView):
    """
    Track influential voices in crypto space
    ✅ ALREADY CORRECT - Uses new user_credibility field structure
    """
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Track top influencers and their recent content.",
        manual_parameters=[
            openapi.Parameter('platform', openapi.IN_QUERY, 
                description="Filter by platform", 
                type=openapi.TYPE_STRING,
                enum=['all', 'twitter', 'reddit', 'youtube'], default='all'),
            openapi.Parameter('hours', openapi.IN_QUERY, 
                description="Hours of data", 
                type=openapi.TYPE_INTEGER, default=48),
            openapi.Parameter('limit', openapi.IN_QUERY, 
                description="Number of influencers", 
                type=openapi.TYPE_INTEGER, default=20),
        ],
        responses={200: "Influencer tracking data"},
        tags=['Market Insights']
    )
    def get(self, request):
        try:
            from myapp.services.mongo_manager import get_mongo_manager
            
            platform = request.GET.get('platform', 'all')
            hours = int(request.GET.get('hours', 48))
            limit = int(request.GET.get('limit', 20))
            
            mongo_manager = get_mongo_manager()
            
            posts = mongo_manager.get_social_posts(
                platform=platform if platform != 'all' else None,
                hours_back=hours,
                limit=500
            )
            
            # Aggregate by author
            author_stats = {}
            for post in posts:
                author = post.get('author_username', 'Unknown')
                platform_name = post.get('platform', 'unknown')
                key = f"{platform_name}:{author}"
                
                if key not in author_stats:
                    author_stats[key] = {
                        'username': author,
                        'platform': platform_name,
                        'posts': [],
                        'total_engagement': 0,
                        'user_credibility': post.get('user_credibility', {})
                    }
                
                engagement = post.get('engagement_metrics', {})
                total = engagement.get('total_engagement', 0)
                
                author_stats[key]['posts'].append({
                    'title': post.get('title', post.get('content', ''))[:100],
                    'engagement': total,
                    'trust_score': post.get('trust_score', 0),
                    'sentiment': post.get('sentiment_analysis', {}).get('label', 'neutral'),
                    'url': post.get('url'),
                    'published_at': post.get('published_at')
                })
                author_stats[key]['total_engagement'] += total
            
            # Calculate influence score with new field structure
            influencers = []
            for key, data in author_stats.items():
                user_cred = data['user_credibility']
                
                # Calculate influence score based on platform
                # ✅ Uses new 'followers' field with proper fallbacks
                if data['platform'] == 'twitter':
                    followers = user_cred.get('followers', user_cred.get('followers_count', 0))
                    verified = user_cred.get('verified', False)
                    influence_score = (followers / 10000) + (10 if verified else 0) + len(data['posts'])
                    
                elif data['platform'] == 'reddit':
                    karma = user_cred.get('followers', user_cred.get('total_karma', 0))
                    is_mod = user_cred.get('is_mod', False)
                    influence_score = (karma / 10000) + (5 if is_mod else 0) + len(data['posts'])
                    
                elif data['platform'] == 'youtube':
                    subs = user_cred.get('followers', user_cred.get('subscriber_count', 0))
                    influence_score = (subs / 10000) + len(data['posts'])
                else:
                    influence_score = len(data['posts'])
                
                influencers.append({
                    'username': data['username'],
                    'platform': data['platform'],
                    'influence_score': round(influence_score, 2),
                    'post_count': len(data['posts']),
                    'total_engagement': data['total_engagement'],
                    'user_metrics': {
                        'followers': user_cred.get('followers', 0),
                        'verified': user_cred.get('verified', user_cred.get('is_mod', False)),
                        'account_age_days': user_cred.get('account_age_days', 0),
                        'influence_level': user_cred.get('influence_level', 'None')
                    },
                    'recent_posts': sorted(data['posts'], key=lambda x: x.get('engagement', 0), reverse=True)[:3]
                })
            
            # Sort by influence score
            influencers.sort(key=lambda x: x['influence_score'], reverse=True)
            
            return Response({
                'influencers': influencers[:limit],
                'total_tracked': len(influencers),
                'time_period': f'Last {hours} hours',
                'generated_at': timezone.now().isoformat()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error tracking influencers: {e}")
            import traceback
            traceback.print_exc()
            return Response({
                'error': 'Failed to track influencers',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
