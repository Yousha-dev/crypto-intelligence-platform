# news_data fetchers - RAW DATA ONLY
"""
News Data Fetchers - Extract raw data from news APIs
All credibility/sentiment analysis happens in services
"""
import os
from dotenv import load_dotenv
import feedparser
import time
import requests
import json
from datetime import datetime
from pathlib import Path

load_dotenv()

CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
MESSARI_API_KEY = os.getenv("MESSARI_API_KEY")

# Create output directories
NEWS_DIR = Path("news")
NEWS_DIR.mkdir(exist_ok=True)


def save_to_json(data, filename):
    """Save data to JSON file in news directory"""
    filepath = NEWS_DIR / filename
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved to {filepath}")
    except Exception as e:
        print(f"❌ Error saving to {filepath}: {e}")


# ============================================================================
# CRYPTOPANIC - RAW DATA ONLY
# ============================================================================

def fetch_cryptopanic_news(currencies=None, filter_type=None, kind="news", 
                          max_items=100):
    """
    Fetch news from CryptoPanic API - RAW DATA ONLY
    
    Args:
        currencies (str): Comma-separated currency codes
        filter_type (str): Filter type ("rising", "hot", "bullish", "bearish", "important")
        kind (str): News kind ("news", "media")
        max_items (int): Maximum items to fetch
    
    Returns:
        list: Raw news articles from CryptoPanic API
    """
    if not CRYPTOPANIC_API_KEY:
        print("CryptoPanic API key not provided")
        return []
    
    print(f"Fetching CryptoPanic news (raw data)")
    
    base_url = "https://cryptopanic.com/api/developer/v2/posts/"
    
    params = {
        "auth_token": CRYPTOPANIC_API_KEY,
        "public": "true",
        "kind": kind
    }
    
    if currencies:
        params["currencies"] = currencies
    if filter_type:
        params["filter"] = filter_type

    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if "results" not in data:
            return []

        articles = []
        
        for item in data["results"][:max_items]:
            try:
                # Extract ONLY raw API fields
                article = {
                    # === RAW API FIELDS ONLY ===
                    "id": item.get("id"),
                    "slug": item.get("slug"),
                    "title": item.get("title"),
                    "description": item.get("description"),
                    "url": item.get("url"),
                    "original_url": item.get("original_url"),
                    "published_at": item.get("published_at"),
                    "created_at": item.get("created_at"),
                    "kind": item.get("kind"),
                    "source": item.get("source"),  # Contains {title, domain, path}
                    "image": item.get("image"),
                    "instruments": item.get("instruments", []),  # Cryptocurrencies mentioned
                    "votes": item.get("votes"),  # {positive, negative, important, ...}
                    "panic_score": item.get("panic_score"),
                    "panic_score_1h": item.get("panic_score_1h"),
                    "author": item.get("author"),
                    "content": item.get("content"),
                    
                    # === METADATA ===
                    "platform": "cryptopanic",
                    "fetched_at": datetime.now().isoformat()
                }
                
                articles.append(article)
                
            except Exception as e:
                print(f"Error extracting CryptoPanic item {item.get('id')}: {e}")
                continue
        
        print(f"CryptoPanic: Fetched {len(articles)} articles")
        
        # Save to JSON
        save_to_json(articles, "cryptopanic.json")
        
        return articles

    except Exception as e:
        print(f"Error fetching CryptoPanic news: {e}")
        return []


# ============================================================================
# CRYPTOCOMPARE - RAW DATA ONLY
# ============================================================================

def fetch_cryptocompare_news(categories=None, exclude_categories=None, 
                            feeds=None, sort_order="latest", lang="EN", 
                            max_items=100):
    """
    Fetch news from CryptoCompare API - RAW DATA ONLY
    
    Returns:
        list: Raw news articles from CryptoCompare API
    """
    print(f"Fetching CryptoCompare news (raw data)")
    
    base_url = "https://min-api.cryptocompare.com/data/v2/news/"
    
    params = {
        "sortOrder": sort_order,
        "lang": lang
    }
    
    if CRYPTOCOMPARE_API_KEY:
        params["api_key"] = CRYPTOCOMPARE_API_KEY
    if categories:
        params["categories"] = categories
    if exclude_categories:
        params["excludeCategories"] = exclude_categories
    if feeds:
        params["feeds"] = feeds

    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") == "Error":
            print(f"CryptoCompare API Error: {data.get('Message')}")
            return []

        articles = []
        
        for item in data.get("Data", [])[:max_items]:
            try:
                # Extract ONLY raw API fields
                article = {
                    # === RAW API FIELDS ONLY ===
                    "id": item.get("id"),
                    "guid": item.get("guid"),
                    "published_on": item.get("published_on"),  # Unix timestamp
                    "imageurl": item.get("imageurl"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "source": item.get("source"),
                    "body": item.get("body"),
                    "tags": item.get("tags"),  # Pipe-separated string
                    "categories": item.get("categories"),  # Pipe-separated string
                    "upvotes": item.get("upvotes"),
                    "downvotes": item.get("downvotes"),
                    "lang": item.get("lang"),
                    "source_info": item.get("source_info"),  # {name, lang, img}
                    
                    # === METADATA ===
                    "platform": "cryptocompare",
                    "fetched_at": datetime.now().isoformat()
                }
                
                articles.append(article)
                
            except Exception as e:
                print(f"Error extracting CryptoCompare item {item.get('id')}: {e}")
                continue
        
        print(f"CryptoCompare: Fetched {len(articles)} articles")
        
        # Save to JSON
        save_to_json(articles, "cryptocompare.json")
        
        return articles

    except Exception as e:
        print(f"Error fetching CryptoCompare news: {e}")
        return []


# ============================================================================
# NEWSAPI - RAW DATA ONLY
# ============================================================================

def fetch_newsapi_articles(query="cryptocurrency", page_size=50, max_items=50):
    """
    Fetch articles from NewsAPI - RAW DATA ONLY
    
    Returns:
        list: Raw articles from NewsAPI
    """
    if not NEWSAPI_API_KEY:
        print("NewsAPI key not provided")
        return []
    
    print(f"Fetching NewsAPI articles (raw data)")
    
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
            print(f"NewsAPI error: {data.get('message')}")
            return []

        articles = []

        for item in data.get("articles", [])[:max_items]:
            try:
                # Extract ONLY raw API fields
                article = {
                    # === RAW API FIELDS ONLY ===
                    "source": item.get("source"),  # {id, name}
                    "author": item.get("author"),
                    "title": item.get("title"),
                    "description": item.get("description"),
                    "url": item.get("url"),
                    "urlToImage": item.get("urlToImage"),
                    "publishedAt": item.get("publishedAt"),
                    "content": item.get("content"),
                    
                    # === METADATA ===
                    "platform": "newsapi",
                    "fetched_at": datetime.now().isoformat()
                }
                
                articles.append(article)
                
            except Exception as e:
                print(f"Error extracting NewsAPI article: {e}")
                continue

        print(f"NewsAPI: Fetched {len(articles)} articles")
        
        # Save to JSON
        save_to_json(articles, "newsapi.json")
        
        return articles

    except Exception as e:
        print(f"Error fetching NewsAPI articles: {e}")
        return []


# ============================================================================
# MESSARI - RAW DATA ONLY
# ============================================================================

def fetch_messari_news(limit=50, max_items=50):
    """
    Fetch news from Messari API - RAW DATA ONLY
    
    Returns:
        list: Raw news articles from Messari API
    """
    print(f"Fetching Messari news (raw data)")
    
    try:
        headers = {}
        if MESSARI_API_KEY:
            headers["x-messari-api-key"] = MESSARI_API_KEY
        
        url = "https://data.messari.io/api/v1/news"
        params = {"limit": min(limit, max_items)}
        
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        articles = []
        
        for item in data.get("data", [])[:max_items]:
            try:
                # Extract ONLY raw API fields
                article = {
                    # === RAW API FIELDS ONLY ===
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "content": item.get("content"),
                    "references": item.get("references", []),
                    "reference_title": item.get("reference_title"),
                    "published_at": item.get("published_at"),
                    "author": item.get("author", {}).get("name") if isinstance(item.get("author"), dict) else item.get("author"),
                    "tags": item.get("tags", []),
                    "url": item.get("url"),
                    
                    # === METADATA ===
                    "platform": "messari",
                    "fetched_at": datetime.now().isoformat()
                } 
                 
                articles.append(article)
                
            except Exception as e:
                print(f"Error extracting Messari item {item.get('id')}: {e}")
                continue
        
        print(f"Messari: Fetched {len(articles)} articles")
        
        # Save to JSON
        save_to_json(articles, "messari.json")
        
        return articles
        
    except Exception as e:
        print(f"Error fetching Messari news: {e}")
        return []


# ============================================================================
# COINDESK RSS - RAW DATA ONLY
# ============================================================================

def fetch_coindesk_news(max_items=50):
    """
    Fetch CoinDesk RSS feed - RAW DATA ONLY
    
    Returns:
        list: Raw articles from CoinDesk RSS feed
    """
    print(f"Fetching CoinDesk RSS (raw data)")
    
    url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    
    try:
        feed = feedparser.parse(url)
        
        if feed.bozo:
            print("️ Warning: CoinDesk RSS feed may have parsing issues")
            
        articles = []
        
        for entry in feed.entries[:max_items]:
            try:
                # Extract ONLY raw RSS fields
                article = {
                    # === RAW RSS FIELDS ONLY ===
                    "title": entry.get("title"),
                    "link": entry.get("link"),
                    "published": entry.get("published"),
                    "summary": entry.get("summary"),
                    "author": entry.get("author"),
                    "media_content": entry.get("media_content", []),
                    "tags": [tag.get("term") for tag in entry.get("tags", [])] if entry.get("tags") else [],
                    
                    # === METADATA ===
                    "platform": "coindesk",
                    "fetched_at": datetime.now().isoformat()
                }
                
                articles.append(article)
                
            except Exception as e:
                print(f"Error extracting CoinDesk entry: {e}")
                continue
        
        print(f"CoinDesk: Fetched {len(articles)} articles")
        
        # Save to JSON
        save_to_json(articles, "coindesk.json")
        
        return articles
        
    except Exception as e:
        print(f"Error fetching CoinDesk: {e}")
        return []