"""
Mock News Data Fetcher - Uses local JSON files instead of API calls
RAW DATA ONLY - All analysis happens in services
"""
import json
import time
from datetime import datetime
from pathlib import Path

MOCK_DATA_DIR = Path(__file__).parent / "mock_news"


def load_mock_data(filename: str) -> list:
    """Load mock data from JSON file"""
    filepath = MOCK_DATA_DIR / filename
    try:
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                cleaned_lines = [line for line in lines if not line.strip().startswith('//')]
                cleaned_content = '\n'.join(cleaned_lines)
                return json.loads(cleaned_content)
        else:
            print(f"️ Mock data file not found: {filepath}")
            return []
    except json.JSONDecodeError as e:
        print(f"Error parsing {filename}: {e}")
        return []
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []


# ============================================================================
# CRYPTOPANIC - RAW DATA ONLY
# ============================================================================

def fetch_cryptopanic_news(currencies=None, filter_type=None, kind="news", max_items=100):
    """
    MOCK: Fetch news from CryptoPanic API - RAW DATA ONLY
    
    Returns:
        list: Raw news articles (same structure as real API)
    """
    print(f"[MOCK] Fetching CryptoPanic news (raw data)")
    print(f"Loading from: {MOCK_DATA_DIR / 'cryptopanic.json'}")
    
    time.sleep(0.1)  # Simulate minimal delay
    
    articles = load_mock_data("cryptopanic.json")
    
    if not articles:
        print("️ No mock CryptoPanic data found")
        return []
    
    raw_articles = []
    
    for item in articles[:max_items]:
        try:
            # Extract ONLY raw API fields (matching real fetcher)
            article = {
                "id": item.get("id"),
                "slug": item.get("slug"),
                "title": item.get("title"),
                "description": item.get("description"),
                "url": item.get("url"),
                "original_url": item.get("original_url"),
                "published_at": item.get("published_at"),
                "created_at": item.get("created_at"),
                "kind": item.get("kind", kind),
                "source": item.get("source"),
                "image": item.get("image"),
                "instruments": item.get("instruments", []),
                "votes": item.get("votes"),
                "panic_score": item.get("panic_score"),
                "panic_score_1h": item.get("panic_score_1h"),
                "author": item.get("author"),
                "content": item.get("content"),
                
                # Metadata
                "platform": "cryptopanic",
                "fetched_at": datetime.now().isoformat()
            }
            
            raw_articles.append(article)
            
        except Exception as e:
            print(f"Error extracting CryptoPanic item {item.get('id')}: {e}")
            continue
    
    print(f"[MOCK] CryptoPanic: Fetched {len(raw_articles)} articles")
    return raw_articles


# ============================================================================
# CRYPTOCOMPARE - RAW DATA ONLY
# ============================================================================

def fetch_cryptocompare_news(categories=None, exclude_categories=None, 
                            feeds=None, sort_order="latest", lang="EN", 
                            max_items=100):
    """
    MOCK: Fetch news from CryptoCompare API - RAW DATA ONLY
    
    Returns:
        list: Raw news articles (same structure as real API)
    """
    print(f"[MOCK] Fetching CryptoCompare news (raw data)")
    print(f"Loading from: {MOCK_DATA_DIR / 'cryptocompare.json'}")
    
    time.sleep(0.1)
    
    articles = load_mock_data("cryptocompare.json")
    
    if not articles:
        print("️ No mock CryptoCompare data found")
        return []
    
    raw_articles = []
    
    for item in articles[:max_items]:
        try:
            # Extract ONLY raw API fields
            article = {
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
                "lang": item.get("lang", lang),
                "source_info": item.get("source_info"),
                
                # Metadata
                "platform": "cryptocompare",
                "fetched_at": datetime.now().isoformat()
            }
            
            raw_articles.append(article)
            
        except Exception as e:
            print(f"Error extracting CryptoCompare item {item.get('id')}: {e}")
            continue
    
    print(f"[MOCK] CryptoCompare: Fetched {len(raw_articles)} articles")
    return raw_articles


# ============================================================================
# NEWSAPI - RAW DATA ONLY
# ============================================================================

def fetch_newsapi_articles(query="cryptocurrency", page_size=50, max_items=50):
    """
    MOCK: Fetch articles from NewsAPI - RAW DATA ONLY
    
    Returns:
        list: Raw articles (same structure as real API)
    """
    print(f"[MOCK] Fetching NewsAPI articles (raw data)")
    print(f"Loading from: {MOCK_DATA_DIR / 'newsapi.json'}")
    
    time.sleep(0.1)
    
    articles = load_mock_data("newsapi.json")
    
    if not articles:
        print("️ No mock NewsAPI data found")
        return []
    
    raw_articles = []
    
    for item in articles[:max_items]:
        try:
            # Extract ONLY raw API fields
            article = {
                "source": item.get("source"),
                "author": item.get("author"),
                "title": item.get("title"),
                "description": item.get("description"),
                "url": item.get("url"),
                "urlToImage": item.get("urlToImage"),
                "publishedAt": item.get("publishedAt"),
                "content": item.get("content"),
                
                # Metadata
                "platform": "newsapi",
                "fetched_at": datetime.now().isoformat()
            }
            
            raw_articles.append(article)
            
        except Exception as e:
            print(f"Error extracting NewsAPI article: {e}")
            continue
    
    print(f"[MOCK] NewsAPI: Fetched {len(raw_articles)} articles")
    return raw_articles


# ============================================================================
# MESSARI - RAW DATA ONLY
# ============================================================================

def fetch_messari_news(limit=50, max_items=50):
    """
    MOCK: Fetch news from Messari API - RAW DATA ONLY
    
    Returns:
        list: Raw news articles (same structure as real API)
    """
    print(f"[MOCK] Fetching Messari news (raw data)")
    print(f"Loading from: {MOCK_DATA_DIR / 'messari.json'}")
    
    time.sleep(0.1)
    
    articles = load_mock_data("messari.json")
    
    if not articles:
        print("️ No mock Messari data found")
        return []
    
    raw_articles = []
    
    for item in articles[:max_items]:
        try:
            # Extract ONLY raw API fields
            article = {
                "id": item.get("id"),
                "title": item.get("title"),
                "content": item.get("content"),
                "references": item.get("references", []),
                "reference_title": item.get("reference_title"),
                "published_at": item.get("published_at"),
                "author": item.get("author", {}).get("name") if isinstance(item.get("author"), dict) else item.get("author"),
                "tags": item.get("tags", []),
                "url": item.get("url"),
                
                # Metadata
                "platform": "messari",
                "fetched_at": datetime.now().isoformat()
            }
            
            raw_articles.append(article)
            
        except Exception as e:
            print(f"Error extracting Messari item {item.get('id')}: {e}")
            continue
    
    print(f"[MOCK] Messari: Fetched {len(raw_articles)} articles")
    return raw_articles


# ============================================================================
# COINDESK RSS - RAW DATA ONLY
# ============================================================================

def fetch_coindesk_news(max_items=50):
    """
    MOCK: Fetch CoinDesk RSS feed - RAW DATA ONLY
    
    Returns:
        list: Raw articles (same structure as real RSS feed)
    """
    print(f"[MOCK] Fetching CoinDesk RSS (raw data)")
    print(f"Loading from: {MOCK_DATA_DIR / 'coindesk.json'}")
    
    time.sleep(0.1)
    
    articles = load_mock_data("coindesk.json")
    
    if not articles:
        print("️ No mock CoinDesk data found")
        return []
    
    raw_articles = []
    
    for entry in articles[:max_items]:
        try:
            # Extract ONLY raw RSS fields
            article = {
                "title": entry.get("title"),
                "link": entry.get("link") or entry.get("url"),
                "published": entry.get("published") or entry.get("published_at"),
                "summary": entry.get("summary") or entry.get("description"),
                "author": entry.get("author"),
                "media_content": entry.get("media_content", []),
                "tags": entry.get("tags", []),
                
                # Metadata
                "platform": "coindesk",
                "fetched_at": datetime.now().isoformat()
            }
            
            raw_articles.append(article)
            
        except Exception as e:
            print(f"Error extracting CoinDesk entry: {e}")
            continue
    
    print(f"[MOCK] CoinDesk: Fetched {len(raw_articles)} articles")
    return raw_articles


# ============================================================================
# MAIN FETCH FUNCTION
# ============================================================================

def fetch_all_news(max_items_per_source=50):
    """
    MOCK: Fetch news from all sources - RAW DATA ONLY
    
    Returns:
        dict: Raw news from all sources
    """
    print("🚀 [MOCK] Starting Multi-Source News Fetching (Raw Data)")
    print("=" * 60)
    
    start_time = time.time()
    all_results = {}
    
    sources = [
        ("cryptopanic", lambda: fetch_cryptopanic_news(
            filter_type="important", 
            max_items=max_items_per_source
        )),
        ("cryptocompare", lambda: fetch_cryptocompare_news(
            categories="BTC,ETH", 
            max_items=max_items_per_source
        )),
        ("newsapi", lambda: fetch_newsapi_articles(
            query="cryptocurrency bitcoin ethereum", 
            max_items=max_items_per_source
        )),
        ("messari", lambda: fetch_messari_news(
            max_items=max_items_per_source
        )),
        ("coindesk", lambda: fetch_coindesk_news(
            max_items=max_items_per_source
        ))
    ]
    
    for source_name, fetch_function in sources:
        try:
            print(f"\n🔄 [MOCK] Fetching from {source_name}...")
            articles = fetch_function()
            all_results[source_name] = articles
            
        except Exception as e:
            print(f"Error fetching from {source_name}: {e}")
            all_results[source_name] = []
    
    execution_time = time.time() - start_time
    
    # Summary
    total_articles = sum(len(articles) for articles in all_results.values())
    print(f"\n{'=' * 60}")
    print(f"[MOCK] Fetched {total_articles} total articles in {execution_time:.1f}s")
    for source, articles in all_results.items():
        print(f"   {source}: {len(articles)} articles")
    
    return all_results


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🔥 [MOCK] News Fetcher - Raw Data Only")
    print("=" * 65) 
    
    all_news = fetch_all_news(max_items_per_source=5)
    
    for source, articles in all_news.items():
        if articles:
            print(f"\n{source.upper()} Sample:")
            article = articles[0]
            print(f"   Title: {article.get('title', 'N/A')[:60]}...")
            print(f"   URL: {article.get('url', article.get('link', 'N/A'))[:50]}...")
            print(f"   Fields: {list(article.keys())}")