"""
Comprehensive RAG System Test Command
Tests all RAG components: Vector Store, Knowledge Graph, LLM Providers, Query Chains, etc.
"""

import logging
import time
from datetime import timedelta
from django.core.management.base import BaseCommand
from django.utils import timezone as dj_timezone

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Test RAG (Retrieval-Augmented Generation) system components'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--component',
            type=str,
            choices=['all', 'embeddings', 'indexing', 'retrieval', 'generation', 'knowledge_graph', 'llm_providers', 'query_chains', 'context', 'postprocessor'],
            default='all',
            help='Which RAG component to test'
        )
        parser.add_argument(
            '--content-type',
            type=str,
            choices=['all', 'news', 'social'],
            default='all',
            help='Which content type to test'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed output'
        )
        parser.add_argument(
            '--skip-llm',
            action='store_true',
            help='Skip LLM generation tests (useful if no API keys)'
        )
        parser.add_argument(
            '--index-sample',
            action='store_true',
            help='Index sample documents for testing'
        )
    
    def handle(self, *args, **options):
        component = options['component']
        content_type = options['content_type']
        verbose = options['verbose']
        skip_llm = options['skip_llm']
        index_sample = options['index_sample']
        
        self.stdout.write(self.style.SUCCESS("\n" + "="*70))
        self.stdout.write(self.style.SUCCESS("   RAG SYSTEM - COMPREHENSIVE TEST"))
        self.stdout.write(self.style.SUCCESS(f"   Component: {component.upper()} | Content: {content_type.upper()}"))
        self.stdout.write(self.style.SUCCESS("="*70 + "\n"))
        
        results = {}
        
        if component in ['all', 'embeddings']:
            results['embeddings'] = self.test_embeddings(verbose)
        
        if component in ['all', 'indexing']:
            results['indexing'] = self.test_indexing(verbose, content_type, index_sample)
        
        if component in ['all', 'retrieval']:
            results['retrieval'] = self.test_retrieval(verbose, content_type)
        
        if component in ['all', 'generation'] and not skip_llm:
            results['generation'] = self.test_generation(verbose)
        
        if component in ['all', 'knowledge_graph']:
            results['knowledge_graph'] = self.test_knowledge_graph(verbose, content_type)
        
        if component in ['all', 'llm_providers'] and not skip_llm:
            results['llm_providers'] = self.test_llm_providers(verbose)
        
        if component in ['all', 'query_chains'] and not skip_llm:
            results['query_chains'] = self.test_query_chains(verbose)
        
        if component in ['all', 'context']:
            results['context'] = self.test_context_manager(verbose)
        
        if component in ['all', 'postprocessor']:
            results['postprocessor'] = self.test_postprocessor(verbose)
        
        # Print summary
        self.print_summary(results)
    
    # =========================================================================
    # SAMPLE DATA
    # =========================================================================
    
    def get_sample_news_articles(self):
        """Sample news articles in RAW format (matching fetcher output)"""
        return [
            {
                'id': 'rag_test_news_1',
                'source_id': 'rag_test_news_1',
                'title': 'Bitcoin Surges Past $100,000 on ETF Approval',
                'description': 'Bitcoin reached a new all-time high following SEC approval of multiple spot Bitcoin ETFs. Institutional investors are showing unprecedented interest in the cryptocurrency market.',
                'content': 'Bitcoin reached a new all-time high following SEC approval of multiple spot Bitcoin ETFs. Institutional investors are showing unprecedented interest in the cryptocurrency market. BlackRock and Fidelity lead the charge with their respective ETF offerings.',
                'url': 'https://example.com/btc-etf-approval',
                'author': 'Michael Chen',
                'platform': 'cryptopanic',
                'published_at': dj_timezone.now().isoformat(),
                # RAW source info (like fetchers return)
                'source': {'title': 'CoinDesk', 'domain': 'coindesk.com'},
                # RAW CryptoPanic fields
                'votes': {'positive': 45, 'negative': 3, 'important': 20},
                'instruments': [{'code': 'BTC'}, {'code': 'ETH'}],
                'kind': 'news',
                # Trust score may be added after credibility engine processes it
                'trust_score': 8.5,
            },
            {
                'id': 'rag_test_news_2',
                'source_id': 'rag_test_news_2',
                'title': 'Ethereum Network Upgrade Reduces Gas Fees by 50%',
                'description': 'The latest Ethereum upgrade has successfully reduced transaction costs, making DeFi more accessible to retail investors.',
                'content': 'The latest Ethereum upgrade has successfully reduced transaction costs, making DeFi more accessible to retail investors. Vitalik Buterin announced the milestone during ETH Denver conference.',
                'url': 'https://example.com/eth-upgrade',
                'author': 'Sarah Johnson',
                'platform': 'cryptocompare',
                'published_at': dj_timezone.now().isoformat(),
                # RAW source info
                'source': {'title': 'Reuters', 'domain': 'reuters.com'},
                # RAW CryptoCompare fields
                'upvotes': 38,
                'downvotes': 2,
                'tags': 'ETH|DEFI|UPGRADE',
                'categories': 'Technology|Blockchain',
                'trust_score': 9.0,
            },
            {
                'id': 'rag_test_news_3',
                'source_id': 'rag_test_news_3',
                'title': 'SEC Announces New Cryptocurrency Regulation Framework',
                'description': 'The Securities and Exchange Commission unveiled new guidelines for cryptocurrency exchanges and token offerings.',
                'content': 'The Securities and Exchange Commission unveiled new guidelines for cryptocurrency exchanges and token offerings. Gary Gensler emphasized investor protection while promoting innovation.',
                'url': 'https://example.com/sec-regulation',
                'author': 'David Williams',
                'platform': 'newsapi',
                'published_at': dj_timezone.now().isoformat(),
                # RAW source info
                'source': {'title': 'Bloomberg', 'domain': 'bloomberg.com'},
                # NewsAPI doesn't have votes/instruments
                'trust_score': 9.2,
            },
            {
                'id': 'rag_test_news_4',
                'source_id': 'rag_test_news_4',
                'title': 'Solana DeFi Ecosystem Reaches $10B TVL',
                'description': 'Solana decentralized finance protocols have collectively reached $10 billion in total value locked.',
                'content': 'Solana decentralized finance protocols have collectively reached $10 billion in total value locked. Major protocols like Marinade and Raydium lead the growth.',
                'url': 'https://example.com/solana-defi',
                'author': 'Alex Turner',
                'platform': 'messari',
                'published_at': dj_timezone.now().isoformat(),
                # RAW source info
                'source': {'title': 'Messari', 'domain': 'messari.io'},
                # RAW Messari fields
                'references': [
                    {'url': 'https://defillama.com/chain/Solana'},
                    {'url': 'https://solana.com/news'}
                ],
                'trust_score': 8.0,
            }
        ]
    
    def get_sample_social_posts(self):
        """Sample social posts in RAW format (matching fetcher output)"""
        now = dj_timezone.now()
        
        reddit_posts = [
            {
                'id': 'rag_test_reddit_1',
                'source_id': 'rag_test_reddit_1',
                'platform': 'reddit',
                'title': 'Bitcoin Technical Analysis - Key Support Levels',
                'selftext': 'Looking at the BTC daily chart, we have strong support at $95K with resistance at $105K. RSI showing oversold conditions. Accumulation zone identified. #Bitcoin $BTC',
                'url': 'https://reddit.com/r/cryptocurrency/test1',
                'author': 'crypto_analyst_pro',
                'subreddit': 'cryptocurrency',
                'created_utc': now.timestamp(),
                # RAW Reddit engagement metrics
                'score': 1250,
                'upvote_ratio': 0.94,
                'num_comments': 230,
                'total_awards_received': 5,
                # RAW author info (from fetcher enrichment)
                'author_info': {
                    'name': 'crypto_analyst_pro',
                    'created_utc': (now - timedelta(days=1500)).timestamp(),
                    'link_karma': 25000,
                    'comment_karma': 60000,
                    'is_mod': True,
                    'is_gold': True,
                    'has_verified_email': True
                },
                # RAW subreddit info
                'subreddit_info': {
                    'display_name': 'cryptocurrency',
                    'subscribers': 6500000,
                    'created_utc': (now - timedelta(days=3000)).timestamp()
                },
                'trust_score': 7.5,
            },
            {
                'id': 'rag_test_reddit_2',
                'source_id': 'rag_test_reddit_2',
                'platform': 'reddit',
                'title': 'Ethereum Staking Rewards Analysis',
                'selftext': 'Current ETH staking APY is around 4.5%. With the recent network upgrades, validator rewards have stabilized. Great time to stake!',
                'url': 'https://reddit.com/r/ethstaker/test2',
                'author': 'eth_validator',
                'subreddit': 'ethstaker',
                'created_utc': now.timestamp(),
                'score': 850,
                'upvote_ratio': 0.91,
                'num_comments': 120,
                'total_awards_received': 2,
                'author_info': {
                    'name': 'eth_validator',
                    'created_utc': (now - timedelta(days=800)).timestamp(),
                    'link_karma': 15000,
                    'comment_karma': 30000,
                    'is_mod': False,
                    'has_verified_email': True
                },
                'subreddit_info': {
                    'display_name': 'ethstaker',
                    'subscribers': 250000
                },
                'trust_score': 7.8,
            }
        ]
        
        twitter_posts = [
            {
                'id': 'rag_test_twitter_1',
                'source_id': 'rag_test_twitter_1',
                'platform': 'twitter',
                'text': 'Bitcoin showing strong accumulation patterns. Institutional inflows continue. Key levels: Support $95K, Resistance $105K. #Bitcoin #BTC $BTC',
                'created_at': now.isoformat(),
                # RAW Twitter engagement metrics
                'public_metrics': {
                    'like_count': 5200,
                    'retweet_count': 1800,
                    'reply_count': 340,
                    'quote_count': 120
                },
                # RAW user info (from fetcher enrichment)
                'user_info': {
                    'id': '123456789',
                    'username': 'verified_crypto_analyst',
                    'name': 'Crypto Analyst Pro',
                    'verified': True,
                    'verified_type': 'business',
                    'created_at': (now - timedelta(days=3000)).isoformat(),
                    'public_metrics': {
                        'followers_count': 450000,
                        'following_count': 800,
                        'tweet_count': 25000,
                        'listed_count': 500
                    }
                },
                'trust_score': 8.0,
            },
            {
                'id': 'rag_test_twitter_2',
                'source_id': 'rag_test_twitter_2',
                'platform': 'twitter',
                'text': 'Ethereum Layer 2 adoption is accelerating! Arbitrum and Optimism TVL hitting new highs. The scaling narrative is playing out. #ETH #Layer2',
                'created_at': now.isoformat(),
                'public_metrics': {
                    'like_count': 3100,
                    'retweet_count': 920,
                    'reply_count': 180,
                    'quote_count': 45
                },
                'user_info': {
                    'id': '987654321',
                    'username': 'defi_researcher',
                    'name': 'DeFi Research',
                    'verified': True,
                    'verified_type': 'blue',
                    'created_at': (now - timedelta(days=1500)).isoformat(),
                    'public_metrics': {
                        'followers_count': 125000,
                        'following_count': 500,
                        'tweet_count': 12000,
                        'listed_count': 200
                    }
                },
                'trust_score': 7.5,
            }
        ]
        
        youtube_posts = [
            {
                'id': 'rag_test_youtube_1',
                'video_id': 'rag_test_yt_1',
                'source_id': 'rag_test_youtube_1',
                'platform': 'youtube',
                'title': 'Bitcoin Price Prediction 2025 - Complete Technical Analysis',
                'description': 'In-depth analysis of Bitcoin market structure and price targets for 2025. Covering on-chain metrics, institutional flows, and macro factors.',
                'channel_id': 'UC_test_channel_1',
                'channel_title': 'CryptoEducator',
                'published_at': now.isoformat(),
                # RAW YouTube video metrics
                'view_count': 250000,
                'like_count': 12000,
                'comment_count': 850,
                'duration_seconds': 2400,
                'caption': 'Full video transcript covering Bitcoin technical analysis and price prediction...',
                # RAW channel info (from fetcher enrichment)
                'channel_info': {
                    'subscriber_count': 750000,
                    'subscriber_count_hidden': False,
                    'total_view_count': 150000000,
                    'video_count': 1200,
                    'channel_created': (now - timedelta(days=3500)).isoformat()
                },
                'trust_score': 7.0,
            },
            {
                'id': 'rag_test_youtube_2',
                'video_id': 'rag_test_yt_2',
                'source_id': 'rag_test_youtube_2',
                'platform': 'youtube',
                'title': 'Solana Ecosystem Deep Dive - Top DeFi Projects',
                'description': 'Comprehensive analysis of Solana DeFi ecosystem. Covering Marinade, Raydium, Jupiter, and emerging protocols.',
                'channel_id': 'UC_test_channel_2',
                'channel_title': 'DeFi Daily',
                'published_at': now.isoformat(),
                'view_count': 85000,
                'like_count': 4500,
                'comment_count': 320,
                'duration_seconds': 1800,
                'channel_info': {
                    'subscriber_count': 320000,
                    'subscriber_count_hidden': False,
                    'total_view_count': 45000000,
                    'video_count': 650,
                    'channel_created': (now - timedelta(days=1800)).isoformat()
                },
                'trust_score': 7.2,
            }
        ]
        
        return reddit_posts + twitter_posts + youtube_posts
    
    # =========================================================================
    # TEST METHODS
    # =========================================================================
    
    def test_embeddings(self, verbose: bool) -> dict:
        """Test embedding model initialization and functionality"""
        self.stdout.write("\n🔢 Testing Embedding Model...")
        
        try:
            from myapp.services.rag.rag_service import get_rag_engine
            
            start_time = time.time()
            rag_engine = get_rag_engine()
            init_time = time.time() - start_time
            
            # Test embedding generation
            test_texts = [
                "Bitcoin price surges to new all-time high",
                "Ethereum network upgrade reduces gas fees",
                "SEC announces new cryptocurrency regulations"
            ]
            
            embeddings = []
            for text in test_texts:
                start = time.time()
                embedding = rag_engine.embed_model.get_text_embedding(text)
                embed_time = time.time() - start
                embeddings.append({
                    'text': text[:30] + '...',
                    'dimension': len(embedding),
                    'time_ms': embed_time * 1000
                })
                
                if verbose:
                    self.stdout.write(f"   '{text[:30]}...' -> dim={len(embedding)}, time={embed_time*1000:.1f}ms")
            
            # Test similarity
            from numpy import dot
            from numpy.linalg import norm
            
            emb1 = rag_engine.embed_model.get_text_embedding(test_texts[0])
            emb2 = rag_engine.embed_model.get_text_embedding("BTC reaches record high price")
            emb3 = rag_engine.embed_model.get_text_embedding("Weather forecast for tomorrow")
            
            similarity_related = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
            similarity_unrelated = dot(emb1, emb3) / (norm(emb1) * norm(emb3))
            
            if verbose:
                self.stdout.write(f"\n   Similarity Test:")
                self.stdout.write(f"      Related texts: {similarity_related:.3f}")
                self.stdout.write(f"      Unrelated texts: {similarity_unrelated:.3f}")
            
            self.stdout.write(self.style.SUCCESS("\n   Embedding Model: PASSED"))
            self.stdout.write(f"      Model: {rag_engine.embedding_model_name}")
            self.stdout.write(f"      Dimension: {rag_engine.embedding_dim}")
            self.stdout.write(f"      Init time: {init_time:.2f}s")
            
            return {
                'status': 'passed',
                'model': rag_engine.embedding_model_name,
                'dimension': rag_engine.embedding_dim,
                'init_time': init_time,
                'similarity_test': {
                    'related': similarity_related,
                    'unrelated': similarity_unrelated,
                    'passed': similarity_related > similarity_unrelated
                }
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   Embedding Model: FAILED - {e}"))
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def test_indexing(self, verbose: bool, content_type: str, index_sample: bool) -> dict:
        """Test document indexing for news and social content"""
        self.stdout.write("\n📥 Testing Document Indexing...")
        
        try:
            from myapp.services.rag.rag_service import get_rag_engine
            
            rag_engine = get_rag_engine()
            results = {'news': {'indexed': 0, 'time_ms': 0}, 'social': {'indexed': 0, 'time_ms': 0}}
            
            # Test NEWS indexing
            if content_type in ['all', 'news']:
                self.stdout.write("\n   Testing NEWS indexing:")
                news_articles = self.get_sample_news_articles()
                
                start_time = time.time()
                if index_sample:
                    stats = rag_engine.bulk_index_articles(news_articles)
                    results['news']['indexed'] = stats['added']
                    results['news']['duplicates'] = stats['duplicates']
                else:
                    # Just test document creation without actually indexing
                    for article in news_articles:
                        doc = rag_engine._create_document_from_article(article)
                        if verbose:
                            self.stdout.write(f"      Created doc: {doc.doc_id}")
                            self.stdout.write(f"         Content length: {len(doc.text)}")
                            self.stdout.write(f"         Metadata keys: {list(doc.metadata.keys())}")
                    results['news']['indexed'] = len(news_articles)
                
                results['news']['time_ms'] = (time.time() - start_time) * 1000
                
                if verbose:
                    self.stdout.write(f"      Indexed: {results['news']['indexed']} articles")
                    self.stdout.write(f"      Time: {results['news']['time_ms']:.1f}ms")
            
            # Test SOCIAL indexing
            if content_type in ['all', 'social']:
                self.stdout.write("\n   Testing SOCIAL indexing:")
                social_posts = self.get_sample_social_posts()
                
                start_time = time.time()
                if index_sample:
                    stats = rag_engine.bulk_index_social_posts(social_posts)
                    results['social']['indexed'] = stats['added']
                    results['social']['duplicates'] = stats['duplicates']
                else:
                    # Just test document creation
                    for post in social_posts:
                        doc = rag_engine._create_document_from_social(post)
                        if verbose:
                            self.stdout.write(f"      Created doc: {doc.doc_id}")
                            self.stdout.write(f"         Platform: {doc.metadata.get('platform')}")
                            self.stdout.write(f"         Content length: {len(doc.text)}")
                    results['social']['indexed'] = len(social_posts)
                
                results['social']['time_ms'] = (time.time() - start_time) * 1000
                
                if verbose:
                    self.stdout.write(f"      Indexed: {results['social']['indexed']} posts")
                    self.stdout.write(f"      Time: {results['social']['time_ms']:.1f}ms")
            
            # Get index stats
            stats = rag_engine.get_stats()
            
            self.stdout.write(self.style.SUCCESS("\n   Document Indexing: PASSED"))
            self.stdout.write(f"      Total docs in index: {stats['vector_store']['documents_count']}")
            
            return {
                'status': 'passed',
                'news_indexed': results['news']['indexed'],
                'social_indexed': results['social']['indexed'],
                'total_in_index': stats['vector_store']['documents_count'],
                'index_sample_mode': index_sample
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   Document Indexing: FAILED - {e}"))
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def test_retrieval(self, verbose: bool, content_type: str) -> dict:
        """Test semantic search retrieval"""
        self.stdout.write("\nTesting Retrieval...")
        
        try:
            from myapp.services.rag.rag_service import get_rag_engine
            
            rag_engine = get_rag_engine()
            
            # Test queries
            test_queries = [
                ("What is happening with Bitcoin price?", ['bitcoin', 'btc', 'price']),
                ("Ethereum network upgrades", ['ethereum', 'eth', 'upgrade']),
                ("SEC cryptocurrency regulation", ['sec', 'regulation', 'crypto']),
                ("DeFi market trends", ['defi', 'market']),
            ]
            
            results = []
            
            for query, expected_terms in test_queries:
                start_time = time.time()
                search_results = rag_engine.retrieve(query, top_k=5)
                retrieval_time = (time.time() - start_time) * 1000
                
                result = {
                    'query': query,
                    'results_count': len(search_results),
                    'time_ms': retrieval_time,
                    'top_scores': [r.score for r in search_results[:3]] if search_results else []
                }
                results.append(result)
                
                if verbose:
                    self.stdout.write(f"\n   Query: '{query}'")
                    self.stdout.write(f"      Results: {len(search_results)}, Time: {retrieval_time:.1f}ms")
                    for i, r in enumerate(search_results[:3], 1):
                        self.stdout.write(f"      [{i}] Score: {r.score:.3f} | {r.metadata.get('title', 'N/A')[:40]}...")
            
            # Test with filters
            self.stdout.write("\n   Testing filtered retrieval:")
            
            # Filter by type
            if content_type in ['all', 'news']:
                news_results = rag_engine.retrieve(
                    "Bitcoin", 
                    top_k=5, 
                    filters={'type': 'news'}
                )
                if verbose:
                    self.stdout.write(f"      News filter: {len(news_results)} results")
            
            if content_type in ['all', 'social']:
                social_results = rag_engine.retrieve(
                    "Bitcoin",
                    top_k=5,
                    filters={'type': 'social'}
                )
                if verbose:
                    self.stdout.write(f"      Social filter: {len(social_results)} results")
            
            avg_time = sum(r['time_ms'] for r in results) / len(results) if results else 0
            avg_results = sum(r['results_count'] for r in results) / len(results) if results else 0
            
            self.stdout.write(self.style.SUCCESS("\n   Retrieval: PASSED"))
            self.stdout.write(f"      Avg retrieval time: {avg_time:.1f}ms")
            self.stdout.write(f"      Avg results per query: {avg_results:.1f}")
            
            return {
                'status': 'passed',
                'queries_tested': len(results),
                'avg_time_ms': avg_time,
                'avg_results': avg_results,
                'results': results
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   Retrieval: FAILED - {e}"))
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def test_generation(self, verbose: bool) -> dict:
        """Test RAG answer generation"""
        self.stdout.write("\nTesting Answer Generation...")
        
        try:
            from myapp.services.rag.rag_service import get_rag_engine
            
            rag_engine = get_rag_engine()
            
            # Check if LLM is available
            if not rag_engine.llm:
                self.stdout.write(self.style.WARNING("   ️ LLM not available, skipping generation test"))
                return {'status': 'skipped', 'reason': 'LLM not configured'}
            
            test_queries = [
                "What is the current sentiment around Bitcoin?",
                "Summarize recent Ethereum developments",
            ]
            
            results = []
            
            for query in test_queries:
                start_time = time.time()
                response = rag_engine.generate_answer(query)
                gen_time = (time.time() - start_time) * 1000
                
                result = {
                    'query': query,
                    'answer_length': len(response.answer),
                    'sources_count': len(response.sources),
                    'confidence': response.confidence,
                    'tokens_used': response.tokens_used,
                    'time_ms': gen_time
                }
                results.append(result)
                
                if verbose:
                    self.stdout.write(f"\n   Query: '{query}'")
                    self.stdout.write(f"      Answer ({len(response.answer)} chars): {response.answer[:200]}...")
                    self.stdout.write(f"      Sources: {len(response.sources)}")
                    self.stdout.write(f"      Confidence: {response.confidence:.2f}")
                    self.stdout.write(f"      Tokens: {response.tokens_used}")
                    self.stdout.write(f"      Time: {gen_time:.0f}ms")
            
            # Test with knowledge graph
            self.stdout.write("\n   Testing generation with Knowledge Graph:")
            start_time = time.time()
            kg_response = rag_engine.generate_answer_with_kg(
                "What is happening with Bitcoin?",
                use_knowledge_graph=True
            )
            kg_time = (time.time() - start_time) * 1000
            
            if verbose:
                self.stdout.write(f"      KG-enhanced answer: {kg_response.answer[:150]}...")
                self.stdout.write(f"      Time: {kg_time:.0f}ms")
            
            avg_time = sum(r['time_ms'] for r in results) / len(results) if results else 0
            
            self.stdout.write(self.style.SUCCESS("\n   Answer Generation: PASSED"))
            self.stdout.write(f"      Model: {rag_engine.llm_provider}/{rag_engine.llm_model}")
            self.stdout.write(f"      Avg generation time: {avg_time:.0f}ms")
            
            return {
                'status': 'passed',
                'model': f"{rag_engine.llm_provider}/{rag_engine.llm_model}",
                'queries_tested': len(results),
                'avg_time_ms': avg_time,
                'results': results
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   Answer Generation: FAILED - {e}"))
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def test_knowledge_graph(self, verbose: bool, content_type: str) -> dict:
        """Test Knowledge Graph functionality"""
        self.stdout.write("\n🕸️ Testing Knowledge Graph...")
        
        try:
            from myapp.services.rag.knowledge_graph import get_knowledge_graph
            
            kg = get_knowledge_graph()
            
            # Get initial stats
            initial_stats = kg.get_graph_statistics()
            
            if verbose:
                self.stdout.write(f"   Initial state:")
                self.stdout.write(f"      Entities: {initial_stats['total_entities']}")
                self.stdout.write(f"      Relationships: {initial_stats['total_relationships']}")
                self.stdout.write(f"      Events: {initial_stats['total_events']}")
            
            # Test entity lookup
            self.stdout.write("\n   Testing entity operations:")
            
            test_entities = ['bitcoin', 'ethereum', 'sec', 'binance']
            for entity_id in test_entities:
                if entity_id in kg.entities:
                    entity = kg.entities[entity_id]
                    if verbose:
                        self.stdout.write(f"      {entity.name} ({entity.entity_type})")
            
            # Test relationship creation
            self.stdout.write("\n   Testing relationship creation:")
            
            from myapp.services.rag.knowledge_graph import Relationship
            
            test_rel = Relationship(
                source_id='bitcoin',
                target_id='sec',
                relation_type='regulated_by',
                weight=1.0
            )
            kg.add_relationship(test_rel)
            
            if verbose:
                self.stdout.write(f"      Created: bitcoin -> sec (regulated_by)")
            
            # Test entity context
            self.stdout.write("\n   Testing entity context retrieval:")
            
            context = kg.get_entity_context('bitcoin', depth=2)
            
            if verbose:
                self.stdout.write(f"      Entity: {context['entity']['name']}")
                self.stdout.write(f"      Related entities: {len(context['related_entities'])}")
                self.stdout.write(f"      Recent events: {len(context['recent_events'])}")
            
            # Test article extraction and linking
            self.stdout.write("\n   Testing article extraction:")
            
            if content_type in ['all', 'news']:
                for article in self.get_sample_news_articles()[:2]:
                    extraction = kg.extract_and_link_from_content(article)
                    if verbose:
                        self.stdout.write(f"      Article: {article['title'][:40]}...")
                        self.stdout.write(f"         Entities found: {extraction['entities_found']}")
                        self.stdout.write(f"         Relationships: {len(extraction['relationships_created'])}")
            
            if content_type in ['all', 'social']:
                for post in self.get_sample_social_posts()[:2]:
                    # Add extracted_entities if not present
                    if 'extracted_entities' not in post:
                        post['extracted_entities'] = post.get('extracted_entities', {
                            'cryptocurrencies': [],
                            'organizations': [],
                            'persons': [],
                            'exchanges': []
                        })
                    extraction = kg.extract_and_link_from_content(post)
                    if verbose:
                        self.stdout.write(f"      Social: {post.get('title', 'N/A')[:40]}...")
                        self.stdout.write(f"         Entities found: {extraction['entities_found']}")
            
            # Test trending entities
            self.stdout.write("\n   Testing trending entities:")
            trending = kg.get_trending_entities(hours_back=24, limit=5)
            
            if verbose:
                for t in trending[:3]:
                    self.stdout.write(f"      {t['name']} ({t['type']}): {t['event_count']} events")
            
            # Test path finding
            self.stdout.write("\n   Testing path finding:")
            paths = kg.find_path('bitcoin', 'sec', max_depth=3)
            
            if verbose:
                self.stdout.write(f"      Paths from Bitcoin to SEC: {len(paths)}")
                if paths:
                    self.stdout.write(f"      Shortest: {' -> '.join(paths[0])}")
            
            # Final stats
            final_stats = kg.get_graph_statistics()
            
            self.stdout.write(self.style.SUCCESS("\n   Knowledge Graph: PASSED"))
            self.stdout.write(f"      Total Entities: {final_stats['total_entities']}")
            self.stdout.write(f"      Total Relationships: {final_stats['total_relationships']}")
            self.stdout.write(f"      Total Events: {final_stats['total_events']}")
            
            return {
                'status': 'passed',
                'statistics': final_stats,
                'trending_count': len(trending),
                'paths_found': len(paths) if paths else 0
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   Knowledge Graph: FAILED - {e}"))
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def test_llm_providers(self, verbose: bool) -> dict:
        """Test LLM provider management"""
        self.stdout.write("\n🔌 Testing LLM Providers...")
        
        try:
            from myapp.services.rag.llm_provider import get_llm_manager, LLMProvider
            
            manager = get_llm_manager()
            
            # Get status
            status = manager.get_status()
            
            if verbose:
                self.stdout.write(f"   Active provider: {status['active_provider']}")
                self.stdout.write(f"\n   Provider Status:")
                for provider, info in status['providers'].items():
                    icon = "✓" if info['available'] else "✗"
                    self.stdout.write(f"      {icon} {provider}: available={info['available']}")
                    if info['available']:
                        self.stdout.write(f"         Model: {info['model']['model']}")
            
            # Test available providers
            available = manager.get_available_providers()
            
            if verbose:
                self.stdout.write(f"\n   Available providers: {available}")
            
            # Test generation with active provider
            if status['active_provider']:
                self.stdout.write(f"\n   Testing generation with {status['active_provider']}:")
                
                try:
                    start_time = time.time()
                    response, tokens, provider = manager.generate(
                        "What is Bitcoin in one sentence?",
                        system_prompt="You are a helpful assistant. Be very brief."
                    )
                    gen_time = (time.time() - start_time) * 1000
                    
                    if verbose:
                        self.stdout.write(f"      Response: {response[:100]}...")
                        self.stdout.write(f"      Tokens: {tokens}")
                        self.stdout.write(f"      Time: {gen_time:.0f}ms")
                        self.stdout.write(f"      Provider used: {provider}")
                    
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f"      Generation failed: {e}"))
            
            self.stdout.write(self.style.SUCCESS("\n   LLM Providers: PASSED"))
            self.stdout.write(f"      Active: {status['active_provider']}")
            self.stdout.write(f"      Available: {len(available)}")
            
            return {
                'status': 'passed',
                'active_provider': status['active_provider'],
                'available_providers': available,
                'provider_status': status['providers']
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   LLM Providers: FAILED - {e}"))
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def test_query_chains(self, verbose: bool) -> dict:
        """Test query chain execution"""
        self.stdout.write("\n⛓️ Testing Query Chains...")
        
        try:
            from myapp.services.rag.query_chain import get_query_chain
            
            chain = get_query_chain()
            
            # Test query decomposition
            self.stdout.write("\n   Testing query decomposition:")
            
            complex_query = "Compare Bitcoin and Ethereum in terms of price performance and technology"
            sub_queries = chain.decompose_query(complex_query)
            
            if verbose:
                self.stdout.write(f"      Original: {complex_query}")
                self.stdout.write(f"      Sub-queries: {len(sub_queries)}")
                for sq in sub_queries[:3]:
                    self.stdout.write(f"         - {sq[:60]}...")
            
            # Test chain type determination
            self.stdout.write("\n   Testing chain type detection:")
            
            test_queries = [
                ("What is Bitcoin price?", "simple"),
                ("Compare BTC vs ETH", "comparison"),
                ("Analyze the impact of ETF approval on crypto market", "deep_analysis"),
            ]
            
            for query, expected_type in test_queries:
                detected = chain._determine_chain_type(query)
                match = "✓" if detected == expected_type else "✗"
                if verbose:
                    self.stdout.write(f"      {match} '{query[:30]}...' -> {detected} (expected: {expected_type})")
            
            # Test simple chain execution
            self.stdout.write("\n   Testing chain execution:")
            
            start_time = time.time()
            result = chain.execute_chain("What is happening with Bitcoin?", chain_type='simple')
            exec_time = (time.time() - start_time) * 1000
            
            if verbose:
                self.stdout.write(f"      Success: {result.success}")
                self.stdout.write(f"      Steps: {len(result.steps)}")
                self.stdout.write(f"      Time: {result.total_duration_ms:.0f}ms")
                self.stdout.write(f"      Answer preview: {result.final_answer[:100]}...")
                
                for step in result.steps:
                    self.stdout.write(f"         Step: {step.step_type.value} ({step.duration_ms:.0f}ms)")
            
            self.stdout.write(self.style.SUCCESS("\n   Query Chains: PASSED"))
            self.stdout.write(f"      Chain executed successfully")
            self.stdout.write(f"      Total steps: {len(result.steps)}")
            
            return {
                'status': 'passed',
                'chain_success': result.success,
                'steps_count': len(result.steps),
                'execution_time_ms': result.total_duration_ms,
                'decomposition_test': len(sub_queries)
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   Query Chains: FAILED - {e}"))
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def test_context_manager(self, verbose: bool) -> dict:
        """Test context and session management"""
        self.stdout.write("\n💾 Testing Context Manager...")
        
        try:
            from myapp.services.rag.context_manager import get_context_manager
            
            manager = get_context_manager()
            
            # Test session creation
            self.stdout.write("\n   Testing session management:")
            
            test_session_id = f"test_session_{int(time.time())}"
            session = manager.get_or_create_session(test_session_id, user_id='test_user')
            
            if verbose:
                self.stdout.write(f"      Created session: {session.session_id}")
                self.stdout.write(f"      User: {session.user_id}")
            
            # Test adding queries to session
            test_interactions = [
                ("What is Bitcoin?", "Bitcoin is a decentralized digital currency...", [{'title': 'Source 1'}]),
                ("How about Ethereum?", "Ethereum is a blockchain platform...", [{'title': 'Source 2'}]),
            ]
            
            for query, answer, sources in test_interactions:
                manager.update_session(test_session_id, query, answer, sources)
            
            if verbose:
                self.stdout.write(f"      Added {len(test_interactions)} queries to session")
            
            # Test context retrieval
            context = manager.get_context_prompt(test_session_id, n_context=2)
            
            if verbose:
                self.stdout.write(f"\n   Session context preview:")
                self.stdout.write(f"      {context[:200]}...")
            
            # Test query caching
            self.stdout.write("\n   Testing query caching:")
            
            test_query = "Test query for caching"
            test_result = {'answer': 'Test answer', 'sources': []}
            
            manager.cache_query_result(test_query, test_result, ttl=60)
            cached = manager.get_cached_result(test_query)
            
            cache_success = cached is not None and cached.get('answer') == test_result['answer']
            
            if verbose:
                self.stdout.write(f"      Cache write/read: {'✓' if cache_success else '✗'}")
            
            # Test performance tracking
            self.stdout.write("\n   Testing performance tracking:")
            
            for i in range(5):
                manager.record_performance(
                    query=f"test_query_{i}",
                    latency=100 + i * 10,
                    tokens_used=50 + i * 5,
                    success=True
                )
            
            perf_stats = manager.get_performance_stats()
            
            if verbose:
                self.stdout.write(f"      Total queries tracked: {perf_stats['total_queries']}")
                self.stdout.write(f"      Success rate: {perf_stats['success_rate']:.0%}")
                self.stdout.write(f"      Avg latency: {perf_stats['avg_latency']:.0f}ms")
            
            # Test adaptive prompt
            self.stdout.write("\n   Testing adaptive prompt building:")
            
            adaptive_prompt = manager.build_adaptive_prompt(
                query="What is the latest news?",
                session_id=test_session_id,
                base_system_prompt="You are a crypto news analyst."
            )
            
            if verbose:
                self.stdout.write(f"      Adaptive prompt length: {len(adaptive_prompt)} chars")
            
            self.stdout.write(self.style.SUCCESS("\n   Context Manager: PASSED"))
            self.stdout.write(f"      Session queries: {len(session.queries)}")
            self.stdout.write(f"      Cache working: {cache_success}")
            
            return {
                'status': 'passed',
                'session_queries': len(session.queries),
                'cache_working': cache_success,
                'performance_stats': perf_stats
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   Context Manager: FAILED - {e}"))
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def test_postprocessor(self, verbose: bool) -> dict:
        """Test post-processing and re-ranking"""
        self.stdout.write("\n🔄 Testing Post-processor...")
        
        try:
            from myapp.services.rag.post_processor import get_postprocessor
            from myapp.services.rag.rag_service import get_rag_engine
            
            postprocessor = get_postprocessor()
            rag_engine = get_rag_engine()
            
            # Test re-ranking
            self.stdout.write("\n   Testing re-ranking:")
            
            # Get some results to re-rank
            query = "Bitcoin price analysis"
            results = rag_engine.retrieve(query, top_k=10)
            
            if results:
                start_time = time.time()
                reranked = postprocessor.rerank_results(query, results, top_k=5)
                rerank_time = (time.time() - start_time) * 1000
                
                if verbose:
                    self.stdout.write(f"      Original results: {len(results)}")
                    self.stdout.write(f"      Re-ranked results: {len(reranked)}")
                    self.stdout.write(f"      Re-rank time: {rerank_time:.1f}ms")
                    
                    if reranked:
                        self.stdout.write(f"\n      Top re-ranked results:")
                        for r in reranked[:3]:
                            self.stdout.write(f"         [{r.final_rank}] Original: {r.original_score:.3f} -> Reranked: {r.reranked_score:.3f}")
            else:
                reranked = []
                rerank_time = 0
                if verbose:
                    self.stdout.write("      No results to re-rank (index may be empty)")
            
            # Test filtering
            self.stdout.write("\n   Testing result filtering:")
            
            if reranked:
                filtered = postprocessor.filter_results(reranked, {
                    'min_trust_score': 5.0,
                    'min_score': 0.3
                })
                
                if verbose:
                    self.stdout.write(f"      Before filter: {len(reranked)}")
                    self.stdout.write(f"      After filter: {len(filtered)}")
            
            # Test insight extraction
            self.stdout.write("\n   Testing insight extraction:")
            
            test_answer = """
Based on the analysis:

1. Bitcoin price shows bullish momentum
2. Institutional adoption is increasing
3. Regulatory clarity is improving

Key points:
- ETF approval has driven prices higher
- Trading volumes are at record levels

Overall sentiment is positive according to [Source 1] and [Source 2].
"""
            
            insights = postprocessor.extract_key_insights(test_answer)
            
            if verbose:
                self.stdout.write(f"      Key points found: {len(insights['key_points'])}")
                self.stdout.write(f"      Sentiment indicators: {len(insights['sentiment_indicators'])}")
                self.stdout.write(f"      Confidence level: {insights['confidence_level']}")
            
            # Test output formatting
            self.stdout.write("\n   Testing output formatting:")
            
            test_sources = [
                {'title': 'Source 1', 'source': 'CoinDesk', 'trust_score': 8.5, 'relevance_score': 0.9},
                {'title': 'Source 2', 'source': 'Reuters', 'trust_score': 9.0, 'relevance_score': 0.85},
            ]
            
            for format_type in ['detailed', 'brief', 'structured']:
                formatted = postprocessor.format_output(test_answer, test_sources, format_type)
                if verbose:
                    self.stdout.write(f"      {format_type}: {list(formatted.keys())}")
            
            # Check cross-encoder status
            cross_encoder_available = postprocessor.cross_encoder is not None
            
            self.stdout.write(self.style.SUCCESS("\n   Post-processor: PASSED"))
            self.stdout.write(f"      Cross-encoder: {'Available' if cross_encoder_available else 'Not available (using heuristics)'}")
            self.stdout.write(f"      Re-ranking time: {rerank_time:.1f}ms")
            
            return {
                'status': 'passed',
                'cross_encoder_available': cross_encoder_available,
                'rerank_time_ms': rerank_time,
                'insights_extracted': len(insights['key_points']),
                'formats_tested': 3
            }
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   Post-processor: FAILED - {e}"))
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    def print_summary(self, results: dict):
        """Print comprehensive test summary"""
        self.stdout.write("\n" + "="*70)
        self.stdout.write(self.style.SUCCESS("   RAG SYSTEM TEST SUMMARY"))
        self.stdout.write("="*70 + "\n")
        
        passed = sum(1 for r in results.values() if r.get('status') == 'passed')
        failed = sum(1 for r in results.values() if r.get('status') == 'failed')
        skipped = sum(1 for r in results.values() if r.get('status') == 'skipped')
        
        # Component results
        self.stdout.write("   COMPONENT RESULTS:\n")
        for component, result in results.items():
            status = result.get('status', 'unknown')
            if status == 'passed':
                icon = "✅"
                style = self.style.SUCCESS
            elif status == 'failed':
                icon = "❌"
                style = self.style.ERROR
            else:
                icon = "⏭️"
                style = self.style.WARNING
            
            self.stdout.write(style(f"      {icon} {component}: {status.upper()}"))
        
        # Summary stats
        self.stdout.write(f"\n   TOTALS:")
        self.stdout.write(f"      Passed: {passed}")
        self.stdout.write(f"      Failed: {failed}")
        self.stdout.write(f"      Skipped: {skipped}")
        self.stdout.write(f"      Total: {len(results)}")
        
        # RAG-specific stats
        if results.get('embeddings', {}).get('status') == 'passed':
            self.stdout.write(f"\n   🔢 Embedding Model:")
            self.stdout.write(f"      {results['embeddings'].get('model', 'N/A')}")
            self.stdout.write(f"      Dimension: {results['embeddings'].get('dimension', 'N/A')}")
        
        if results.get('llm_providers', {}).get('status') == 'passed':
            self.stdout.write(f"\n   🔌 LLM Providers:")
            self.stdout.write(f"      Active: {results['llm_providers'].get('active_provider', 'N/A')}")
            self.stdout.write(f"      Available: {results['llm_providers'].get('available_providers', [])}")
        
        if results.get('knowledge_graph', {}).get('status') == 'passed':
            stats = results['knowledge_graph'].get('statistics', {})
            self.stdout.write(f"\n   🕸️ Knowledge Graph:")
            self.stdout.write(f"      Entities: {stats.get('total_entities', 0)}")
            self.stdout.write(f"      Relationships: {stats.get('total_relationships', 0)}")
         
        if failed == 0:
            self.stdout.write(self.style.SUCCESS("\n   🎉 ALL RAG TESTS PASSED!"))
        else:
            self.stdout.write(self.style.ERROR(f"\n   ️ {failed} TESTS FAILED - Review errors above"))
        
        self.stdout.write("\n" + "="*70 + "\n")