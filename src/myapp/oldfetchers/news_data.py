# news_data fetchers
import os
from dotenv import load_dotenv
import feedparser
import time
import requests
import re
from collections import Counter
from textblob import TextBlob
from datetime import datetime 
 
load_dotenv() 

CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
MESSARI_API_KEY = os.getenv("MESSARI_API_KEY")

# --- Enhanced CryptoPanic with Credibility Analysis ---
def fetch_cryptopanic_news_enhanced(currencies=None, filter_type=None, kind="news", size=None, search=None, 
                                   analyze_credibility=True, max_items=100):
    """
    Fetch news from CryptoPanic API with comprehensive credibility and sentiment analysis
    OPTIMIZED FOR API QUOTA LIMITS
    
    Args:
        currencies (str): Comma-separated currency codes
        filter_type (str): Filter type ("rising", "hot", "bullish", "bearish", "important")
        kind (str): News kind ("news", "media")
        size (int): Number of items per page(pro only)
        search (str): Search keyword
        analyze_credibility (bool): Whether to perform credibility analysis
        max_items (int): Maximum items to process (quota management)
    """
    
    if not CRYPTOPANIC_API_KEY:
        print("CryptoPanic API key not provided")
        return []
    
    print(f"Fetching CryptoPanic news with credibility analysis")
    print(f"API Budget: Processing up to {max_items} items")
    
    base_url = "https://cryptopanic.com/api/developer/v2/posts/"
    
    params = {
        "auth_token": CRYPTOPANIC_API_KEY,
        "public": "true",
        "kind": kind
    }
    
    # Add optional parameters
    if currencies:
        params["currencies"] = currencies
    if filter_type:
        params["filter"] = filter_type
    if search:
        params["search"] = search
    
    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if "results" not in data:
            return []

        enhanced_news = []
        api_calls_made = 1  # Track API usage
        
        for item in data["results"][:max_items]:  # Limit processing
            try:
                # Core article data
                article = {
                    # Original fields
                    "id": item.get("id"),
                    "slug": item.get("slug"),
                    "title": item.get("title"),
                    "description": item.get("description"),
                    "url": item.get("url"),
                    "original_url": item.get("original_url"),
                    "published_at": item.get("published_at"),
                    "created_at": item.get("created_at"),
                    "kind": item.get("kind"),
                    "source": item.get("source"),
                    "image": item.get("image"),
                    "instruments": item.get("instruments", []),
                    "votes": item.get("votes"),
                    "panic_score": item.get("panic_score"),
                    "panic_score_1h": item.get("panic_score_1h"),
                    "author": item.get("author"),
                    "content": item.get("content"),
                    
                    # Enhanced fields
                    "platform": "CryptoPanic",
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
                if analyze_credibility:
                    # Perform comprehensive analysis
                    source_credibility = analyze_news_source_credibility(item)
                    content_analysis = analyze_news_content_quality(item)
                    sentiment_analysis = analyze_news_sentiment(item)
                    crypto_analysis = analyze_crypto_relevance_news(item)
                    market_impact = analyze_market_impact_potential(item)
                    
                    # Calculate overall credibility score
                    overall_credibility = calculate_news_trust_score(
                        item, source_credibility, content_analysis, 
                        sentiment_analysis, crypto_analysis, market_impact
                    )
                    
                    # Add analysis results
                    article.update({
                        "source_credibility": source_credibility,
                        "content_analysis": content_analysis,
                        "sentiment_analysis": sentiment_analysis,
                        "crypto_analysis": crypto_analysis,
                        "market_impact": market_impact,
                        "overall_credibility": overall_credibility
                    })
                
                enhanced_news.append(article)
                
            except Exception as e:
                print(f"Error processing CryptoPanic item {item.get('id')}: {e}")
                continue
        
        # Sort by credibility if analysis was performed
        if analyze_credibility and enhanced_news:
            enhanced_news.sort(
                key=lambda x: x.get('overall_credibility', {}).get('final_trust_score', 0), 
                reverse=True
            )
            
            # Analyze results
            analyze_news_results(enhanced_news, "CryptoPanic")
        
        print(f"CryptoPanic: Processed {len(enhanced_news)} articles")
        print(f"🔧 API calls made: {api_calls_made}")
        
        return enhanced_news

    except Exception as e:
        print(f"Error fetching CryptoPanic news: {e}")
        return []

# --- Enhanced CryptoCompare with Analysis ---
def fetch_cryptocompare_news_enhanced(categories=None, excludeCategories=None, feeds=None, 
                                     lTs=None, sortOrder="latest", lang="EN", analyze_credibility=True, max_items=100):
    """
    Fetch news from CryptoCompare API with enhanced credibility analysis
    """
    
    print(f"Fetching CryptoCompare news with credibility analysis")
    
    base_url = "https://min-api.cryptocompare.com/data/v2/news/"
    
    params = {
        "sortOrder": sortOrder,
        "lang": lang
    }
    
    if CRYPTOCOMPARE_API_KEY:
        params["api_key"] = CRYPTOCOMPARE_API_KEY
    
    # Add optional parameters
    if categories:
        params["categories"] = categories
    if excludeCategories:
        params["excludeCategories"] = excludeCategories
    if feeds:
        params["feeds"] = feeds
    if lTs:
        params["lTs"] = lTs

    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") == "Error":
            print(f"CryptoCompare API Error: {data.get('Message', 'Unknown error')}")
            return []

        news_data = data.get("Data", [])
        enhanced_news = []
        api_calls_made = 1
        
        for item in news_data[:max_items]:
            try:
                article = {
                    # Original fields
                    "id": item.get("id"),
                    "guid": item.get("guid"),
                    "published_on": item.get("published_on"),
                    "imageurl": item.get("imageurl"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "source": item.get("source"),
                    "body": item.get("body"),
                    "tags": item.get("tags"),
                    "categories": item.get("categories"),
                    "upvotes": item.get("upvotes"),
                    "downvotes": item.get("downvotes"),
                    "lang": item.get("lang"),
                    "source_info": item.get("source_info"),
                    
                    # Enhanced fields
                    "platform": "CryptoCompare",
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
                if analyze_credibility:
                    # Comprehensive analysis
                    source_credibility = analyze_news_source_credibility(item, "cryptocompare")
                    content_analysis = analyze_news_content_quality(item, "cryptocompare")
                    sentiment_analysis = analyze_news_sentiment(item, "cryptocompare")
                    crypto_analysis = analyze_crypto_relevance_news(item, "cryptocompare")
                    market_impact = analyze_market_impact_potential(item, "cryptocompare")
                    
                    overall_credibility = calculate_news_trust_score(
                        item, source_credibility, content_analysis,
                        sentiment_analysis, crypto_analysis, market_impact
                    )
                    
                    article.update({
                        "source_credibility": source_credibility,
                        "content_analysis": content_analysis,
                        "sentiment_analysis": sentiment_analysis,
                        "crypto_analysis": crypto_analysis,
                        "market_impact": market_impact,
                        "overall_credibility": overall_credibility
                    })
                
                enhanced_news.append(article)
                
            except Exception as e:
                print(f"Error processing CryptoCompare item {item.get('id')}: {e}")
                continue
        
        if analyze_credibility and enhanced_news:
            enhanced_news.sort(
                key=lambda x: x.get('overall_credibility', {}).get('final_trust_score', 0),
                reverse=True
            )
            analyze_news_results(enhanced_news, "CryptoCompare")
        
        print(f"CryptoCompare: Processed {len(enhanced_news)} articles")
        return enhanced_news

    except Exception as e:
        print(f"Error fetching CryptoCompare news: {e}")
        return []

# --- Enhanced NewsAPI ---
def fetch_newsapi_articles_enhanced(query="cryptocurrency", page_size=20, analyze_credibility=True, max_items=50):
    """
    Fetch articles from NewsAPI with enhanced credibility analysis
    """
    
    print(f"Fetching NewsAPI articles with credibility analysis")
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": NEWSAPI_API_KEY,
        "pageSize": min(page_size, max_items),
        "sortBy": "publishedAt",
        "language": "en"
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            print(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return []

        enhanced_articles = []
        api_calls_made = 1

        for article_data in data.get("articles", [])[:max_items]:
            try:
                article = {
                    # Original fields
                    "source": article_data.get("source"),
                    "author": article_data.get("author"),
                    "title": article_data.get("title"),
                    "description": article_data.get("description"),
                    "url": article_data.get("url"),
                    "urlToImage": article_data.get("urlToImage"),
                    "publishedAt": article_data.get("publishedAt"),
                    "content": article_data.get("content"),
                    
                    # Enhanced fields
                    "platform": "NewsAPI",
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
                if analyze_credibility:
                    source_credibility = analyze_news_source_credibility(article_data, "newsapi")
                    content_analysis = analyze_news_content_quality(article_data, "newsapi")
                    sentiment_analysis = analyze_news_sentiment(article_data, "newsapi")
                    crypto_analysis = analyze_crypto_relevance_news(article_data, "newsapi")
                    market_impact = analyze_market_impact_potential(article_data, "newsapi")
                    
                    overall_credibility = calculate_news_trust_score(
                        article_data, source_credibility, content_analysis,
                        sentiment_analysis, crypto_analysis, market_impact
                    )
                    
                    article.update({
                        "source_credibility": source_credibility,
                        "content_analysis": content_analysis,
                        "sentiment_analysis": sentiment_analysis,
                        "crypto_analysis": crypto_analysis,
                        "market_impact": market_impact,
                        "overall_credibility": overall_credibility
                    })
                
                enhanced_articles.append(article)
                
            except Exception as e:
                print(f"Error processing NewsAPI article: {e}")
                continue

        if analyze_credibility and enhanced_articles:
            enhanced_articles.sort(
                key=lambda x: x.get('overall_credibility', {}).get('final_trust_score', 0),
                reverse=True
            )
            analyze_news_results(enhanced_articles, "NewsAPI")

        print(f"NewsAPI: Processed {len(enhanced_articles)} articles")
        return enhanced_articles

    except Exception as e:
        print(f"Error fetching NewsAPI articles: {e}")
        return []

# --- Enhanced Messari News ---
def fetch_messari_news_enhanced(fields=None, limit=50, analyze_credibility=True, max_items=50):
    """
    Fetch news from Messari with enhanced analysis
    """
    
    print(f"Fetching Messari news with credibility analysis")
    
    try:
        headers = {}
        if MESSARI_API_KEY:
            headers["x-messari-api-key"] = MESSARI_API_KEY
        
        url = "https://data.messari.io/api/v1/news"
        params = {"limit": min(limit, max_items)}
        
        if fields:
            params["fields"] = ",".join(fields)
        
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        enhanced_news = []
        api_calls_made = 1
        
        for item in data.get("data", [])[:max_items]:
            try:
                article = {
                    # Original fields
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "content": item.get("content"),
                    "references": item.get("references", []),
                    "reference_title": item.get("reference_title"),
                    "published_at": item.get("published_at"),
                    "author": item.get("author", {}).get("name"),
                    "tags": item.get("tags", []),
                    "url": item.get("url"),
                    
                    # Enhanced fields
                    "platform": "Messari",
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
                if analyze_credibility:
                    source_credibility = analyze_news_source_credibility(item, "messari")
                    content_analysis = analyze_news_content_quality(item, "messari")
                    sentiment_analysis = analyze_news_sentiment(item, "messari")
                    crypto_analysis = analyze_crypto_relevance_news(item, "messari")
                    market_impact = analyze_market_impact_potential(item, "messari")
                    
                    overall_credibility = calculate_news_trust_score(
                        item, source_credibility, content_analysis,
                        sentiment_analysis, crypto_analysis, market_impact
                    )
                    
                    article.update({
                        "source_credibility": source_credibility,
                        "content_analysis": content_analysis,
                        "sentiment_analysis": sentiment_analysis,
                        "crypto_analysis": crypto_analysis,
                        "market_impact": market_impact,
                        "overall_credibility": overall_credibility
                    })
                
                enhanced_news.append(article)
                
            except Exception as e:
                print(f"Error processing Messari item {item.get('id')}: {e}")
                continue
        
        if analyze_credibility and enhanced_news:
            enhanced_news.sort(
                key=lambda x: x.get('overall_credibility', {}).get('final_trust_score', 0),
                reverse=True
            )
            analyze_news_results(enhanced_news, "Messari")
        
        print(f"Messari: Processed {len(enhanced_news)} articles")
        return enhanced_news
        
    except Exception as e:
        print(f"Error fetching Messari news: {e}")
        return []

# --- Enhanced CoinDesk RSS ---
def scrape_coindesk_enhanced(analyze_credibility=True, max_items=50):
    """
    Scrape CoinDesk RSS feed with enhanced analysis
    """
    
    print(f"Fetching CoinDesk RSS with credibility analysis")
    
    url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    try:
        feed = feedparser.parse(url)
        if feed.bozo:
            print("Warning: CoinDesk RSS feed may have parsing issues")
            
        enhanced_articles = []
        api_calls_made = 0  # No API calls for RSS
        
        for entry in feed.entries[:max_items]:
            try:
                article = {
                    # Original fields
                    "title": entry.title,
                    "link": entry.link,
                    "published": getattr(entry, "published", None),
                    "summary": getattr(entry, "summary", None),
                    "media_content": entry.get("media_content", []),
                    
                    # Enhanced fields
                    "platform": "CoinDesk",
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
                if analyze_credibility:
                    source_credibility = analyze_news_source_credibility(entry, "coindesk")
                    content_analysis = analyze_news_content_quality(entry, "coindesk")
                    sentiment_analysis = analyze_news_sentiment(entry, "coindesk")
                    crypto_analysis = analyze_crypto_relevance_news(entry, "coindesk")
                    market_impact = analyze_market_impact_potential(entry, "coindesk")
                    
                    overall_credibility = calculate_news_trust_score(
                        entry, source_credibility, content_analysis,
                        sentiment_analysis, crypto_analysis, market_impact
                    )
                    
                    article.update({
                        "source_credibility": source_credibility,
                        "content_analysis": content_analysis,
                        "sentiment_analysis": sentiment_analysis,
                        "crypto_analysis": crypto_analysis,
                        "market_impact": market_impact,
                        "overall_credibility": overall_credibility
                    })
                
                enhanced_articles.append(article)
                
            except Exception as e:
                print(f"Error processing CoinDesk entry: {e}")
                continue
        
        if analyze_credibility and enhanced_articles:
            enhanced_articles.sort(
                key=lambda x: x.get('overall_credibility', {}).get('final_trust_score', 0),
                reverse=True
            )
            analyze_news_results(enhanced_articles, "CoinDesk")
        
        print(f"CoinDesk: Processed {len(enhanced_articles)} articles")
        return enhanced_articles
        
    except Exception as e:
        print(f"Error scraping CoinDesk: {e}")
        return []

# --- CREDIBILITY ANALYSIS FUNCTIONS ---

def analyze_news_source_credibility(item, platform="cryptopanic"):
    """Analyze source credibility for news articles"""
    try:
        # Extract source information based on platform
        source_name = ""
        source_info = {}
        
        if platform == "cryptopanic":
            source_name = item.get("source", {}).get("title", "") if isinstance(item.get("source"), dict) else str(item.get("source", ""))
            source_info = item.get("source", {}) if isinstance(item.get("source"), dict) else {}
        elif platform == "cryptocompare":
            source_name = item.get("source", "")
            source_info = item.get("source_info", {})
        elif platform == "newsapi":
            source_data = item.get("source", {})
            source_name = source_data.get("name", "") if isinstance(source_data, dict) else str(source_data)
            source_info = source_data if isinstance(source_data, dict) else {}
        elif platform == "messari":
            source_name = "Messari"
            source_info = {"platform": "messari"}
        elif platform == "coindesk":
            source_name = "CoinDesk"
            source_info = {"platform": "coindesk"}
        
        # Known credible crypto news sources
        high_credibility_sources = {
            "coindesk": 9.0,
            "cointelegraph": 8.5,
            "decrypt": 8.0,
            "the block": 8.5,
            "messari": 9.0,
            "coinbase": 7.5,
            "binance": 7.0,
            "forbes": 8.0,
            "reuters": 9.5,
            "bloomberg": 9.0,
            "wall street journal": 9.0,
            "financial times": 8.5
        }
        
        medium_credibility_sources = {
            "bitcoin magazine": 7.5,
            "cryptoslate": 7.0,
            "newsbtc": 6.5,
            "cryptopotato": 6.0,
            "u.today": 6.0,
            "ambcrypto": 6.0
        }
        
        source_lower = source_name.lower()
        
        # Calculate source credibility score
        trust_score = 5.0  # Default medium credibility
        source_tier = "Unknown"
        
        for source, score in high_credibility_sources.items():
            if source in source_lower:
                trust_score = score
                source_tier = "High Credibility"
                break
        else:
            for source, score in medium_credibility_sources.items():
                if source in source_lower:
                    trust_score = score
                    source_tier = "Medium Credibility"
                    break
            else:
                if any(keyword in source_lower for keyword in ["official", "gov", ".gov", "reuters", "ap news"]):
                    trust_score = 8.5
                    source_tier = "Official Source"
                elif any(keyword in source_lower for keyword in ["blog", "personal", "unknown"]):
                    trust_score = 3.0
                    source_tier = "Low Credibility"
        
        # Additional credibility indicators
        credibility_indicators = {
            "is_known_source": trust_score > 5.0,
            "is_high_credibility": trust_score >= 8.0,
            "is_official_source": source_tier == "Official Source",
            "has_source_info": bool(source_info),
            "source_name_length": len(source_name),
            "appears_professional": not bool(re.search(r'[0-9]{3,}|xxx|pump|moon|rocket', source_lower))
        }
        
        return {
            "source_name": source_name,
            "source_info": source_info,
            "trust_score": trust_score,
            "source_tier": source_tier,
            "credibility_indicators": credibility_indicators,
            "platform": platform
        }
        
    except Exception as e:
        print(f"Error analyzing source credibility: {e}")
        return create_default_source_credibility()

def create_default_source_credibility():
    """Create default source credibility for error cases"""
    return {
        "source_name": "Unknown",
        "trust_score": 3.0,
        "source_tier": "Unknown",
        "credibility_indicators": {},
        "error": "Could not analyze source"
    }

def analyze_news_content_quality(item, platform="cryptopanic"):
    """Analyze content quality for news articles"""
    try:
        # Extract content based on platform
        title = ""
        content = ""
        description = ""
        
        if platform == "cryptopanic":
            title = item.get("title") or ""
            content = item.get("content") or item.get("description") or ""
            description = item.get("description") or ""
        elif platform == "cryptocompare":
            title = item.get("title") or ""
            content = item.get("body") or ""
            description = content[:200] if content else ""
        elif platform == "newsapi":
            title = item.get("title") or ""
            content = item.get("content") or item.get("description") or ""
            description = item.get("description") or ""
        elif platform == "messari":
            title = item.get("title") or ""
            content = item.get("content") or ""
            description = content[:200] if content else ""
        elif platform == "coindesk":
            title = getattr(item, "title", "") or ""
            content = getattr(item, "summary", "") or ""
            description = content
        
        # Ensure no None values
        title = str(title) if title else ""
        content = str(content) if content else ""
        description = str(description) if description else ""
        
        all_text = f"{title} {description} {content}"
        
        # Quality factors
        quality_factors = {
            "has_meaningful_title": len(title) > 10 and len(title) < 200,
            "has_content": len(content) > 50,
            "has_description": len(description) > 20,
            "appropriate_length": 100 < len(all_text) < 5000,
            "not_all_caps": not title.isupper() if title else True,
            "no_excessive_punctuation": len(re.findall(r'[!?]{2,}', all_text)) < 3,
            "proper_grammar": not bool(re.search(r'\b(ur|u|pls|omg|lol|rofl)\b', all_text.lower())),
            "no_spam_keywords": not bool(re.search(r'\b(click here|buy now|limited time|act fast|guaranteed|100%)\b', all_text.lower())),
            "professional_tone": not bool(re.search(r'🚀|💰|📈|MOON|PUMP|DUMP', all_text))
        }
        
        # Content analysis - handle empty text
        if all_text.strip():
            word_count = len(all_text.split())
            sentence_count = len(re.split(r'[.!?]+', all_text))
            avg_sentence_length = word_count / max(sentence_count, 1)
        else:
            word_count = 0
            sentence_count = 0
            avg_sentence_length = 0
        
        # Calculate quality score
        quality_score = sum(quality_factors.values()) / len(quality_factors) * 10
        
        # Readability factors
        readability_factors = {
            "appropriate_word_count": 50 < word_count < 1000,
            "good_sentence_length": 10 < avg_sentence_length < 25,
            "has_structure": bool(re.search(r'[.!?]', all_text)),
            "varied_vocabulary": len(set(all_text.lower().split())) / max(word_count, 1) > 0.5 if word_count > 0 else False
        }
        
        readability_score = sum(readability_factors.values()) / len(readability_factors) * 10
        
        final_quality_score = (quality_score * 0.7) + (readability_score * 0.3)
        
        return {
            "quality_factors": quality_factors,
            "quality_score": quality_score,
            "readability_factors": readability_factors,
            "readability_score": readability_score,
            "final_quality_score": final_quality_score,
            "content_stats": {
                "title_length": len(title),
                "content_length": len(content),
                "total_length": len(all_text),
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": avg_sentence_length
            },
            "quality_tier": get_content_quality_tier(final_quality_score)
        }
        
    except Exception as e:
        print(f"Error analyzing content quality: {e}")
        return create_default_content_quality()


def create_default_content_quality():
    """Create default content quality for error cases"""
    return {
        "final_quality_score": 5.0,
        "quality_tier": "Unknown",
        "error": "Could not analyze content quality"
    }

def analyze_news_sentiment(item, platform="cryptopanic"):
    """Analyze sentiment of news articles"""
    try:
        # Extract text for sentiment analysis
        title = ""
        content = ""
        
        if platform == "cryptopanic":
            title = item.get("title") or ""
            content = item.get("content") or item.get("description") or ""
        elif platform == "cryptocompare":
            title = item.get("title") or ""
            content = item.get("body") or ""
        elif platform == "newsapi":
            title = item.get("title") or ""
            content = item.get("content") or item.get("description") or ""
        elif platform == "messari":
            title = item.get("title") or ""
            content = item.get("content") or ""
        elif platform == "coindesk":
            title = getattr(item, "title", "") or ""
            content = getattr(item, "summary", "") or ""
        
        # Ensure strings are not None
        title = str(title) if title else ""
        content = str(content) if content else ""
        all_text = f"{title} {content}".strip()
        
        if not all_text:
            return create_default_sentiment()
        
        # Use TextBlob for basic sentiment analysis
        try:
            blob = TextBlob(all_text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
        except Exception as e:
            print(f"TextBlob analysis failed: {e}")
            polarity = 0.0
            subjectivity = 0.5
        
        # Rest of the function remains the same...
        # Crypto-specific sentiment keywords
        bullish_keywords = ["bullish", "bull", "rise", "up", "gain", "positive", "growth", "increase", "surge", "rally", "moon", "pump"]
        bearish_keywords = ["bearish", "bear", "fall", "down", "loss", "negative", "decline", "decrease", "crash", "dump", "drop"]
        neutral_keywords = ["stable", "consolidate", "sideways", "hold", "range", "analyze", "technical"]
        
        text_lower = all_text.lower()
        
        # Count sentiment keywords
        bullish_count = sum(text_lower.count(word) for word in bullish_keywords)
        bearish_count = sum(text_lower.count(word) for word in bearish_keywords)
        neutral_count = sum(text_lower.count(word) for word in neutral_keywords)
        
        # Calculate crypto sentiment score
        total_sentiment_words = bullish_count + bearish_count + neutral_count
        if total_sentiment_words > 0:
            crypto_sentiment = (bullish_count - bearish_count) / total_sentiment_words
        else:
            crypto_sentiment = polarity  # Fall back to TextBlob
        
        # Determine sentiment classification
        if crypto_sentiment > 0.2:
            sentiment_label = "Bullish"
        elif crypto_sentiment < -0.2:
            sentiment_label = "Bearish"
        else:
            sentiment_label = "Neutral"
        
        # Confidence calculation
        confidence = min(abs(crypto_sentiment) + (subjectivity * 0.3), 1.0)
        
        return {
            "textblob_polarity": polarity,
            "textblob_subjectivity": subjectivity,
            "crypto_sentiment_score": crypto_sentiment,
            "sentiment_label": sentiment_label,
            "confidence": confidence,
            "keyword_analysis": {
                "bullish_keywords": bullish_count,
                "bearish_keywords": bearish_count,
                "neutral_keywords": neutral_count,
                "total_sentiment_words": total_sentiment_words
            },
            "sentiment_strength": get_sentiment_strength(abs(crypto_sentiment)),
            "market_emotion": determine_market_emotion(crypto_sentiment, subjectivity)
        }
        
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return create_default_sentiment()

def create_default_sentiment():
    """Create default sentiment for error cases"""
    return {
        "sentiment_label": "Neutral",
        "crypto_sentiment_score": 0.0,
        "confidence": 0.0,
        "error": "Could not analyze sentiment"
    }

def analyze_crypto_relevance_news(item, platform="cryptopanic"):
    """Analyze crypto relevance of news articles"""
    try:
        # Extract text content
        title = ""
        content = ""
        tags = []
        categories = []
        
        if platform == "cryptopanic":
            title = item.get("title", "")
            content = item.get("content", "") or item.get("description", "")
            # CryptoPanic has instruments (cryptocurrencies mentioned)
            instruments = item.get("instruments", [])
            if instruments:
                categories = [instr.get("code", "") for instr in instruments if isinstance(instr, dict)]
        elif platform == "cryptocompare":
            title = item.get("title", "")
            content = item.get("body", "")
            tags = item.get("tags", "").split(",") if item.get("tags") else []
            categories = item.get("categories", "").split(",") if item.get("categories") else []
        elif platform == "newsapi":
            title = item.get("title", "")
            content = item.get("content", "") or item.get("description", "")
        elif platform == "messari":
            title = item.get("title", "")
            content = item.get("content", "")
            tags = item.get("tags", [])
        elif platform == "coindesk":
            title = getattr(item, "title", "")
            content = getattr(item, "summary", "")
        
        all_text = f"{title} {content}".lower()
        
        # Primary crypto keywords (high relevance)
        primary_crypto_keywords = [
            "bitcoin", "btc", "ethereum", "eth", "cryptocurrency", "crypto", "blockchain",
            "altcoin", "defi", "nft", "web3", "smart contract", "mining", "wallet"
        ]
        
        # Secondary crypto keywords (medium relevance)
        secondary_crypto_keywords = [
            "trading", "exchange", "binance", "coinbase", "market cap", "bull run", "bear market",
            "hodl", "pump", "dump", "moon", "diamond hands", "paper hands", "whale", "satoshi"
        ]
        
        # Specific cryptocurrencies
        major_cryptos = [
            "ada", "cardano", "sol", "solana", "matic", "polygon", "dot", "polkadot",
            "link", "chainlink", "uni", "uniswap", "aave", "comp", "compound", "maker", "mkr"
        ]
        
        # Count keyword occurrences
        primary_matches = sum(all_text.count(keyword) for keyword in primary_crypto_keywords)
        secondary_matches = sum(all_text.count(keyword) for keyword in secondary_crypto_keywords)
        crypto_mentions = sum(all_text.count(crypto) for crypto in major_cryptos)
        
        # Calculate relevance score
        relevance_score = (
            primary_matches * 3 +
            secondary_matches * 2 +
            crypto_mentions * 1 +
            len(categories) * 2 +  # Platform-specific crypto categories
            len(tags) * 0.5       # Tags if available
        )
        
        # Normalize to 0-10 scale
        final_relevance_score = min(relevance_score / 5, 10.0)
        
        # Determine crypto focus areas
        focus_areas = []
        if any(keyword in all_text for keyword in ["trading", "market", "price", "bull", "bear"]):
            focus_areas.append("Trading/Markets")
        if any(keyword in all_text for keyword in ["defi", "smart contract", "protocol"]):
            focus_areas.append("DeFi/Technology")
        if any(keyword in all_text for keyword in ["nft", "art", "collectible"]):
            focus_areas.append("NFTs")
        if any(keyword in all_text for keyword in ["regulation", "sec", "government", "legal"]):
            focus_areas.append("Regulation")
        if any(keyword in all_text for keyword in ["adoption", "institutional", "company"]):
            focus_areas.append("Adoption")
        
        return {
            "relevance_score": final_relevance_score,
            "primary_keyword_matches": primary_matches,
            "secondary_keyword_matches": secondary_matches,
            "crypto_mentions": crypto_mentions,
            "categories_mentioned": categories,
            "tags": tags,
            "focus_areas": focus_areas,
            "is_crypto_focused": final_relevance_score > 3.0,
            "relevance_tier": get_crypto_relevance_tier(final_relevance_score),
            "mentioned_cryptocurrencies": extract_mentioned_cryptos(all_text)
        }
        
    except Exception as e:
        print(f"Error analyzing crypto relevance: {e}")
        return create_default_crypto_relevance()

def create_default_crypto_relevance():
    """Create default crypto relevance for error cases"""
    return {
        "relevance_score": 5.0,
        "is_crypto_focused": True,
        "relevance_tier": "Unknown",
        "error": "Could not analyze crypto relevance"
    }

def analyze_market_impact_potential(item, platform="cryptopanic"):
    """Analyze potential market impact of news"""
    try:
        # Extract relevant data
        title = ""
        content = ""
        votes = 0
        panic_score = 0
        
        if platform == "cryptopanic":
            title = item.get("title", "")
            content = item.get("content", "") or item.get("description", "")
            votes = item.get("votes", {}).get("positive", 0) if isinstance(item.get("votes"), dict) else 0
            panic_score = item.get("panic_score", 0) or 0
        elif platform == "cryptocompare":
            title = item.get("title", "")
            content = item.get("body", "")
            
            # FIX: Safely handle upvotes and downvotes conversion
            upvotes = item.get("upvotes", 0)
            downvotes = item.get("downvotes", 0)
            
            # Convert to integers safely
            try:
                if upvotes is None or upvotes == "":
                    upvotes = 0
                else:
                    upvotes = int(str(upvotes))
                    
                if downvotes is None or downvotes == "":
                    downvotes = 0
                else:
                    downvotes = int(str(downvotes))
                    
                votes = upvotes - downvotes
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert votes to int: upvotes={upvotes}, downvotes={downvotes}. Error: {e}")
                votes = 0
                
        elif platform == "newsapi":
            title = item.get("title", "")
            content = item.get("content", "") or item.get("description", "")
        elif platform == "messari":
            title = item.get("title", "")
            content = item.get("content", "")
        elif platform == "coindesk":
            title = getattr(item, "title", "")
            content = getattr(item, "summary", "")
        
        # Ensure strings are not None
        title = str(title) if title else ""
        content = str(content) if content else ""
        all_text = f"{title} {content}".lower()
        
        # High impact keywords
        high_impact_keywords = [
            "breaking", "urgent", "major", "significant", "important", "critical",
            "sec", "regulation", "ban", "approval", "etf", "institutional",
            "hack", "security", "breach", "crash", "surge", "rally"
        ]
        
        # Market moving events
        market_events = [
            "etf approval", "sec decision", "regulatory", "institutional adoption",
            "major partnership", "listing", "delisting", "hack", "upgrade",
            "hard fork", "merger", "acquisition"
        ]
        
        # Count impact indicators
        high_impact_count = sum(all_text.count(keyword) for keyword in high_impact_keywords)
        market_event_count = sum(all_text.count(event) for event in market_events)
        
        # Calculate impact score
        impact_score = 0
        
        # Base score from keywords
        impact_score += high_impact_count * 2
        impact_score += market_event_count * 3
        
        # Platform-specific factors
        if platform == "cryptopanic" and panic_score:
            try:
                panic_score_num = float(panic_score) if panic_score else 0
                impact_score += panic_score_num / 20  # Panic score contribution
            except (ValueError, TypeError):
                pass
        
        if votes > 0:
            impact_score += min(votes / 10, 2)  # Vote contribution (capped)
        
        # Source credibility boost
        if any(source in all_text for source in ["reuters", "bloomberg", "sec", "official"]):
            impact_score += 2
        
        # Normalize to 0-10 scale
        final_impact_score = min(impact_score, 10.0)
        
        # Determine impact factors
        impact_factors = {
            "has_breaking_news": "breaking" in all_text,
            "regulatory_news": any(word in all_text for word in ["sec", "regulation", "regulatory"]),
            "institutional_news": "institutional" in all_text,
            "technical_news": any(word in all_text for word in ["upgrade", "fork", "protocol"]),
            "market_data": any(word in all_text for word in ["price", "volume", "market cap"]),
            "high_engagement": votes > 50 if votes else False,
            "panic_indicator": panic_score > 50 if panic_score else False
        }
        
        return {
            "impact_score": final_impact_score,
            "high_impact_keywords": high_impact_count,
            "market_events": market_event_count,
            "votes": votes,
            "panic_score": panic_score,
            "impact_factors": impact_factors,
            "impact_tier": get_market_impact_tier(final_impact_score),
            "potential_market_effect": determine_market_effect(final_impact_score, impact_factors)
        }
        
    except Exception as e:
        print(f"Error analyzing market impact: {e}")
        return create_default_market_impact()


def create_default_market_impact():
    """Create default market impact for error cases"""
    return {
        "impact_score": 3.0,
        "impact_tier": "Low Impact",
        "error": "Could not analyze market impact"
    }

def calculate_news_trust_score(item, source_credibility, content_analysis, 
                                   sentiment_analysis, crypto_analysis, market_impact):
    """Calculate overall credibility score for news articles"""
    try:
        # Weight factors for final score
        source_weight = 0.35
        content_weight = 0.25
        sentiment_weight = 0.15
        crypto_weight = 0.15
        impact_weight = 0.10
        
        # Extract individual scores
        source_score = source_credibility.get('trust_score', 5.0)
        content_score = content_analysis.get('final_quality_score', 5.0)
        sentiment_confidence = sentiment_analysis.get('confidence', 0.5) * 10
        crypto_score = crypto_analysis.get('relevance_score', 5.0)
        impact_score = market_impact.get('impact_score', 3.0)
        
        # Calculate weighted final score
        final_score = (
            source_score * source_weight +
            content_score * content_weight +
            sentiment_confidence * sentiment_weight +
            crypto_score * crypto_weight +
            impact_score * impact_weight
        )
        
        # Ensure score is within bounds
        final_score = max(0.0, min(final_score, 10.0))
        
        # Credibility factors summary
        credibility_factors = {
            "high_credibility_source": source_credibility.get('trust_score', 0) >= 8.0,
            "good_content_quality": content_analysis.get('final_quality_score', 0) >= 7.0,
            "clear_sentiment": sentiment_analysis.get('confidence', 0) >= 0.7,
            "crypto_relevant": crypto_analysis.get('is_crypto_focused', False),
            "high_market_impact": market_impact.get('impact_score', 0) >= 7.0,
            "professional_source": source_credibility.get('credibility_indicators', {}).get('appears_professional', False),
            "substantial_content": content_analysis.get('content_stats', {}).get('word_count', 0) > 100
        }
        
        return {
            "final_trust_score": final_score,
            "score_breakdown": {
                "source_credibility": source_score,
                "content_quality": content_score,
                "sentiment_confidence": sentiment_confidence,
                "crypto_relevance": crypto_score,
                "market_impact": impact_score
            },
            "weights_used": {
                "source": source_weight,
                "content": content_weight,
                "sentiment": sentiment_weight,
                "crypto": crypto_weight,
                "impact": impact_weight
            },
            "credibility_factors": credibility_factors,
            "credibility_tier": get_news_credibility_tier(final_score),
            "recommendation": get_news_recommendation(final_score, credibility_factors)
        }
        
    except Exception as e:
        print(f"Error calculating credibility score: {e}")
        return create_default_trust_score()

def create_default_trust_score():
    """Create default credibility score for error cases"""
    return {
        "final_trust_score": 5.0,
        "credibility_tier": "Medium Credibility",
        "error": "Could not calculate credibility score"
    }

# --- HELPER FUNCTIONS ---

def get_content_quality_tier(score):
    """Get content quality tier based on score"""
    if score >= 8.5:
        return "Excellent Quality"
    elif score >= 7.0:
        return "High Quality"
    elif score >= 5.5:
        return "Good Quality"
    elif score >= 4.0:
        return "Fair Quality"
    else:
        return "Poor Quality"

def get_sentiment_strength(abs_score):
    """Get sentiment strength classification"""
    if abs_score >= 0.7:
        return "Very Strong"
    elif abs_score >= 0.5:
        return "Strong"
    elif abs_score >= 0.3:
        return "Moderate"
    elif abs_score >= 0.1:
        return "Weak"
    else:
        return "Very Weak"

def determine_market_emotion(sentiment_score, subjectivity):
    """Determine market emotion from sentiment"""
    if sentiment_score > 0.5 and subjectivity > 0.7:
        return "FOMO (Fear of Missing Out)"
    elif sentiment_score < -0.5 and subjectivity > 0.7:
        return "FUD (Fear, Uncertainty, Doubt)"
    elif abs(sentiment_score) < 0.2:
        return "Calm/Analytical"
    elif sentiment_score > 0:
        return "Optimistic"
    else:
        return "Pessimistic"

def get_crypto_relevance_tier(score):
    """Get crypto relevance tier"""
    if score >= 8:
        return "Highly Relevant"
    elif score >= 6:
        return "Very Relevant"
    elif score >= 4:
        return "Relevant"
    elif score >= 2:
        return "Somewhat Relevant"
    else:
        return "Not Relevant"

def get_market_impact_tier(score):
    """Get market impact tier"""
    if score >= 8:
        return "High Impact"
    elif score >= 6:
        return "Medium-High Impact"
    elif score >= 4:
        return "Medium Impact"
    elif score >= 2:
        return "Low-Medium Impact"
    else:
        return "Low Impact"

def get_news_credibility_tier(score):
    """Get overall news credibility tier"""
    if score >= 8.5:
        return "Highly Credible"
    elif score >= 7.0:
        return "Very Credible"
    elif score >= 5.5:
        return "Credible"
    elif score >= 4.0:
        return "Moderately Credible"
    elif score >= 2.5:
        return "Low Credibility"
    else:
        return "Very Low Credibility"

def extract_mentioned_cryptos(text):
    """Extract mentioned cryptocurrencies from text"""
    crypto_patterns = {
        "Bitcoin": r"\b(bitcoin|btc)\b",
        "Ethereum": r"\b(ethereum|eth)\b",
        "Cardano": r"\b(cardano|ada)\b",
        "Solana": r"\b(solana|sol)\b",
        "Polkadot": r"\b(polkadot|dot)\b",
        "Chainlink": r"\b(chainlink|link)\b",
        "Polygon": r"\b(polygon|matic)\b",
        "Uniswap": r"\b(uniswap|uni)\b"
    }
    
    mentioned = []
    for crypto, pattern in crypto_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            mentioned.append(crypto)
    
    return mentioned

def determine_market_effect(impact_score, impact_factors):
    """Determine potential market effect"""
    if impact_score >= 8:
        if impact_factors.get("regulatory_news"):
            return "Major Regulatory Impact"
        elif impact_factors.get("institutional_news"):
            return "Institutional Movement"
        else:
            return "Significant Market Moving"
    elif impact_score >= 6:
        return "Moderate Market Influence"
    elif impact_score >= 4:
        return "Minor Market Effect"
    else:
        return "Limited Market Impact"

def get_news_recommendation(score, factors):
    """Get recommendation based on credibility"""
    if score >= 8.0:
        return "Highly Recommended - Excellent source"
    elif score >= 7.0:
        return "Recommended - Good credibility"
    elif score >= 5.0:
        return "Acceptable - Verify with other sources"
    elif score >= 3.0:
        return "Use Caution - Low credibility"
    else:
        return "Not Recommended - Very low credibility"

def analyze_news_results(news_articles, platform_name):
    """Analyze and summarize news results"""
    if not news_articles:
        return
    
    total_articles = len(news_articles)
    avg_credibility = sum(
        article.get('overall_credibility', {}).get('final_trust_score', 0) 
        for article in news_articles
    ) / total_articles
    
    # Credibility distribution
    tiers = {
        "Highly Credible": 0, "Very Credible": 0, "Credible": 0, 
        "Moderately Credible": 0, "Low Credibility": 0, "Very Low Credibility": 0
    }
    
    for article in news_articles:
        tier = article.get('overall_credibility', {}).get('credibility_tier', 'Very Low Credibility')
        if tier in tiers:
            tiers[tier] += 1
    
    # Sentiment distribution
    sentiments = {"Bullish": 0, "Bearish": 0, "Neutral": 0}
    for article in news_articles:
        sentiment = article.get('sentiment_analysis', {}).get('sentiment_label', 'Neutral')
        if sentiment in sentiments:
            sentiments[sentiment] += 1
    
    print(f"📈 {platform_name} Analysis Summary:")
    print(f"   Total Articles: {total_articles}")
    print(f"   Average Credibility: {avg_credibility:.2f}/10")
    print(f"   Sentiment Distribution: {dict(list(sentiments.items()))}")
    print(f"   Top Credibility Tiers: {dict(list(tiers.items())[:3])}")

# --- Enhanced Main Functions ---
def fetch_all_news_enhanced(max_items_per_source=50, trust_score_threshold=6.0, analyze_all=True):
    """
    Fetch news from all sources with enhanced analysis
    OPTIMIZED FOR API QUOTA MANAGEMENT
    
    Args:
        max_items_per_source (int): Maximum items per source (quota management)
        trust_score_threshold (float): Minimum credibility threshold for filtering
        analyze_all (bool): Whether to perform full credibility analysis
    """
    
    print("🚀 Starting Enhanced Multi-Source News Fetching")
    print(f"Processing up to {max_items_per_source} items per source")
    print(f"Credibility threshold: {trust_score_threshold}")
    print("=" * 60)
    
    start_time = time.time()
    all_results = {}
    
    # Fetch from all sources with quota management
    sources = [
        ("CryptoPanic", lambda: fetch_cryptopanic_news_enhanced(
            filter_type="important", 
            analyze_credibility=analyze_all, 
            max_items=max_items_per_source
        )), 
        ("CryptoCompare", lambda: fetch_cryptocompare_news_enhanced(
            categories="BTC,ETH", 
            analyze_credibility=analyze_all, 
            max_items=max_items_per_source
        )),
        ("NewsAPI", lambda: fetch_newsapi_articles_enhanced(
            query="cryptocurrency bitcoin ethereum", 
            analyze_credibility=analyze_all, 
            max_items=max_items_per_source
        )),
        ("Messari", lambda: fetch_messari_news_enhanced(
            limit=max_items_per_source, 
            analyze_credibility=analyze_all, 
            max_items=max_items_per_source
        )),
        ("CoinDesk", lambda: scrape_coindesk_enhanced(
            analyze_credibility=analyze_all, 
            max_items=max_items_per_source
        ))  
    ]
    
    for source_name, fetch_function in sources:
        try:
            print(f"\n🔄 Fetching from {source_name}...")
            articles = fetch_function()
            
            if analyze_all and articles:
                # Filter by credibility threshold
                high_credibility_articles = [
                    article for article in articles
                    if article.get('overall_credibility', {}).get('final_trust_score', 0) >= trust_score_threshold
                ]
                
                print(f"{source_name}: {len(high_credibility_articles)}/{len(articles)} articles meet credibility threshold")
                all_results[source_name.lower()] = high_credibility_articles
            else:
                all_results[source_name.lower()] = articles
            
            # Small delay between sources to respect rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error fetching from {source_name}: {e}")
            all_results[source_name.lower()] = []
            continue
    
    execution_time = time.time() - start_time
    
    # Generate comprehensive summary
    generate_enhanced_summary(all_results, execution_time, trust_score_threshold)
    
    return all_results

def generate_enhanced_summary(all_results, execution_time, trust_score_threshold):
    """Generate enhanced summary of all results"""
    total_articles = sum(len(articles) for articles in all_results.values())
    
    print(f"\nENHANCED NEWS ANALYSIS SUMMARY")
    print("=" * 50)
    print(f" Execution Time: {execution_time:.1f} seconds")
    print(f"Total High-Quality Articles: {total_articles}")
    print(f"🎚️  Credibility Threshold: {trust_score_threshold}/10")
    
    # Source breakdown
    print(f"\nSource Breakdown:")
    for source, articles in all_results.items():
        if articles:
            avg_credibility = sum(
                article.get('overall_credibility', {}).get('final_trust_score', 0) 
                for article in articles
            ) / len(articles)
            print(f"   {source.upper()}: {len(articles)} articles (avg: {avg_credibility:.2f}/10)")
    
    # Overall sentiment
    all_articles = [article for articles in all_results.values() for article in articles]
    if all_articles:
        sentiments = {"Bullish": 0, "Bearish": 0, "Neutral": 0}
        for article in all_articles:
            sentiment = article.get('sentiment_analysis', {}).get('sentiment_label', 'Neutral')
            if sentiment in sentiments:
                sentiments[sentiment] += 1
        
        print(f"\n💭 Overall Market Sentiment:")
        total_sentiment_articles = sum(sentiments.values())
        for sentiment, count in sentiments.items():
            percentage = (count / total_sentiment_articles * 100) if total_sentiment_articles > 0 else 0
            print(f"   {sentiment}: {count} articles ({percentage:.1f}%)")
        
        # Top trending topics
        all_focus_areas = []
        for article in all_articles:
            focus_areas = article.get('crypto_analysis', {}).get('focus_areas', [])
            all_focus_areas.extend(focus_areas)
        
        if all_focus_areas:
            trending_topics = Counter(all_focus_areas).most_common(5)
            print(f"\n🔥 Trending Topics:")
            for topic, count in trending_topics:
                print(f"   {topic}: {count} mentions")

# Test function for high-credibility news
def fetch_high_credibility_news(trust_score_threshold=7.0, max_items=30):
    """
    Fetch only high-credibility news with minimal API usage
    
    Args:
        trust_score_threshold (float): Minimum credibility score
        max_items (int): Maximum items per source
    """
    print(f"Fetching High-Credibility News (threshold: {trust_score_threshold})")
    print(f"Conservative API usage: {max_items} items per source max")
    
    return fetch_all_news_enhanced(
        max_items_per_source=max_items,
        trust_score_threshold=trust_score_threshold,
        analyze_all=True
    )

if __name__ == "__main__":
    print("🚀 Enhanced Crypto News Fetcher with Credibility Analysis")
    print("=" * 65)
    
    # Test high-credibility news fetching
    high_credibility_news = fetch_high_credibility_news(
        trust_score_threshold=6.0,
        max_items=20
    )
    
    # Show top articles from each source
    print(f"\n🏆 TOP HIGH-CREDIBILITY ARTICLES:")
    for source_name, articles in high_credibility_news.items():
        if articles:
            top_article = max(articles, key=lambda x: x.get('overall_credibility', {}).get('final_trust_score', 0))
            credibility = top_article.get('overall_credibility', {})
            
            print(f"\n{source_name.upper()}:")
            print(f"   Title: {top_article.get('title', 'No title')[:80]}...")
            print(f"   Credibility: {credibility.get('final_trust_score', 0):.2f}/10")
            print(f"   Sentiment: {top_article.get('sentiment_analysis', {}).get('sentiment_label', 'Unknown')}")
            print(f"   Source: {top_article.get('source_credibility', {}).get('source_name', 'Unknown')}") 