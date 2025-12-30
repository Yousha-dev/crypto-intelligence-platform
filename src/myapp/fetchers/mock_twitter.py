"""
Mock Twitter Fetcher - Uses local JSON files instead of API calls
RAW DATA ONLY - All analysis happens in services
"""
import json
import time
from datetime import datetime
from django.utils import timezone
from pathlib import Path

MOCK_DATA_DIR = Path(__file__).parent / "mock_social"


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
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []


def fetch_twitter_posts(queries=None, max_results=50, hours_back=24):
    """
    MOCK: Fetch cryptocurrency tweets using Twitter API v2 - RAW DATA ONLY
    
    Args:
        queries (list): List of search queries
        max_results (int): Maximum results per query (10-100)
        hours_back (int): How many hours back to search
    
    Returns:
        list: Raw tweets (same structure as real API)
    """
    if queries is None:
        queries = [
            "bitcoin OR BTC cryptocurrency",
            "ethereum OR ETH crypto",
            "cryptocurrency market analysis"
        ]
    
    print(f"[MOCK] Fetching Twitter posts from {len(queries)} queries (raw data)")
    print(f"Loading from: {MOCK_DATA_DIR / 'twitter.json'}")
    
    time.sleep(0.1)
    
    tweets = load_mock_data("twitter.json")
    
    if not tweets: 
        print("️ No mock Twitter data found")
        return []
     
    print(f"Loaded tweets count: {len(tweets)}") 
    
    all_tweets = []
    
    for query in queries:
        for tweet in tweets[:max_results]:
            try:
                user_info = tweet.get('user_info', {})
                public_metrics = tweet.get('public_metrics', {})
                
                # Build raw user info
                raw_user_info = None
                if user_info:
                    raw_user_info = {
                        "id": user_info.get("id"),
                        "username": user_info.get("username"),
                        "name": user_info.get("name"),
                        "description": user_info.get("description"),
                        "verified": user_info.get("verified", False),
                        "verified_type": user_info.get("verified_type"),
                        "created_at": user_info.get("created_at"),
                        "location": user_info.get("location"),
                        "profile_image_url": user_info.get("profile_image_url"),
                        "protected": user_info.get("protected", False),
                        "url": user_info.get("url"),
                        "public_metrics": user_info.get("public_metrics", {}),
                    }
                
                # Extract ONLY raw API fields
                tweet_data = {
                    # Raw tweet fields
                    "id": tweet.get("id"),
                    "text": tweet.get("text", ""),
                    "created_at": tweet.get("created_at", timezone.now().isoformat()),
                    "author_id": tweet.get("author_id", user_info.get("id")),
                    "conversation_id": tweet.get("conversation_id"),
                    "lang": tweet.get("lang", "en"),
                    "possibly_sensitive": tweet.get("possibly_sensitive", False),
                    "source": tweet.get("source"),
                    "reply_settings": tweet.get("reply_settings"),
                    "in_reply_to_user_id": tweet.get("in_reply_to_user_id"),
                    
                    # Raw engagement metrics (from API)
                    "public_metrics": public_metrics,
                    
                    # Raw context data
                    "context_annotations": tweet.get("context_annotations", []),
                    "entities": tweet.get("entities", {}),
                    "referenced_tweets": tweet.get("referenced_tweets"),
                    "geo": tweet.get("geo"),
                    
                    # Raw user info
                    "user_info": raw_user_info,
                    
                    # Convenience fields (derived from raw)
                    "username": user_info.get("username") if user_info else None,
                    "url": f"https://twitter.com/{user_info.get('username', 'user') if user_info else 'user'}/status/{tweet.get('id', '')}",
                    
                    # Metadata
                    "platform": "twitter",
                    "query": query,
                    "fetched_at": datetime.now().isoformat()
                }
                
                all_tweets.append(tweet_data)
                
            except Exception as e:
                print(f"Error extracting tweet {tweet.get('id')}: {e}")
                continue
    
    # Remove duplicates by tweet ID
    unique_tweets = {t['id']: t for t in all_tweets if t.get('id')}
    all_tweets = list(unique_tweets.values())
    
    print(f"[MOCK] Total Twitter posts fetched: {len(all_tweets)}")
    
    return all_tweets

 
# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🔥 [MOCK] Twitter Fetcher - Raw Data Only")
    print("=" * 65)
    
    tweets = fetch_twitter_posts(
        queries=["bitcoin OR BTC"],
        max_results=10,
        hours_back=24
    )
    
    if tweets:
        print(f"\n📝 Sample Tweet:")
        tweet = tweets[0]
        print(f"   Text: {tweet.get('text', '')[:60]}...")
        print(f"   Author: @{tweet.get('username')}")
        print(f"   Metrics: {tweet.get('public_metrics')}")
        print(f"   Fields: {list(tweet.keys())}")