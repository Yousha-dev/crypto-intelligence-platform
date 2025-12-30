"""
Mock Reddit Fetcher - Uses local JSON files instead of API calls
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


def fetch_reddit_posts(subreddits=None, limit=50, time_filter="day", sort_type="hot"):
    """
    MOCK: Fetch cryptocurrency posts from Reddit - RAW DATA ONLY
    
    Args:
        subreddits (list): List of subreddit names
        limit (int): Number of posts per subreddit
        time_filter (str): Time filter for top posts
        sort_type (str): Sort type (hot, new, rising, top)
    
    Returns:
        list: Raw Reddit posts (same structure as real API)
    """
    if subreddits is None:
        subreddits = ["CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets", "altcoin"]
    
    print(f"[MOCK] Fetching Reddit posts from {len(subreddits)} subreddits (raw data)")
    print(f"Loading from: {MOCK_DATA_DIR / 'reddit.json'}")
    
    time.sleep(0.1)
    
    posts = load_mock_data("reddit.json")
    
    if not posts:
        print("️ No mock Reddit data found")
        return []
    
    raw_posts = []
    
    for post in posts[:limit]:
        try:
            # Build subreddit info (raw)
            subreddit_data = post.get('subreddit_info', {})
            subreddit_info = {
                "name": subreddit_data.get("name", post.get("subreddit", "CryptoCurrency")),
                "subscribers": subreddit_data.get("subscribers", 100000),
                "created_utc": subreddit_data.get("created_utc", 1400000000),
                "public_description": subreddit_data.get("public_description", ""),
                "over18": subreddit_data.get("over18", False),
                "subreddit_type": subreddit_data.get("subreddit_type", "public"),
                "active_user_count": subreddit_data.get("active_user_count", 1000),
                "accounts_active": subreddit_data.get("accounts_active", 1000),
            } 
            
            # Build author info (raw)
            author_data = post.get('author_info', {})
            author_info = None
            if author_data and post.get('author') != "[deleted]":
                author_info = {
                    "name": author_data.get("name", post.get("author", "unknown")),
                    "link_karma": author_data.get("link_karma", 1000),
                    "comment_karma": author_data.get("comment_karma", 1000),
                    "created_utc": author_data.get("created_utc", 1500000000),
                    "is_gold": author_data.get("is_gold", False),
                    "is_mod": author_data.get("is_mod", False),
                    "has_verified_email": author_data.get("has_verified_email", True),
                }
            
            # Extract ONLY raw API fields
            post_data = {
                # Raw post fields
                "id": post.get("id"),
                "title": post.get("title"),
                "selftext": post.get("selftext", ""),
                "url": post.get("url"),
                "permalink": post.get("permalink", f"https://reddit.com/r/{post.get('subreddit', 'unknown')}/comments/{post.get('id', '')}"),
                "subreddit": post.get("subreddit"),
                "author": post.get("author", "[deleted]"),
                
                # Raw timestamps
                "created_utc": post.get("created_utc", int(timezone.now().timestamp())),
                
                # Raw engagement metrics (from API)
                "score": post.get("score", 0),
                "upvote_ratio": post.get("upvote_ratio", 0.5),
                "num_comments": post.get("num_comments", 0),
                "gilded": post.get("gilded", 0),
                "total_awards_received": post.get("total_awards_received", 0),
                
                # Raw post attributes
                "link_flair_text": post.get("link_flair_text"),
                "is_self": post.get("is_self", True),
                "is_video": post.get("is_video", False),
                "over_18": post.get("over_18", False),
                "spoiler": post.get("spoiler", False),
                "stickied": post.get("stickied", False),
                "locked": post.get("locked", False),
                "distinguished": post.get("distinguished"),
                "is_original_content": post.get("is_original_content", False),
                
                # Raw media info
                "domain": post.get("domain", "self.reddit"),
                "thumbnail": post.get("thumbnail"),
                "has_media": post.get("has_media", False),
                
                # Raw related data
                "author_info": author_info,
                "subreddit_info": subreddit_info,
                
                # Metadata
                "platform": "reddit",
                "fetched_at": datetime.now().isoformat()
            }
            
            raw_posts.append(post_data)
            
        except Exception as e:
            print(f"Error extracting post {post.get('id')}: {e}")
            continue
    
     # Remove duplicates by ID
    unique_posts = {p['id']: p for p in raw_posts if p.get('id')}
    all_posts = list(unique_posts.values())
    
    print(f"[MOCK] Total Reddit posts fetched: {len(all_posts)}")
    
    return all_posts

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🔥 [MOCK] Reddit Fetcher - Raw Data Only")
    print("=" * 65)
    
    posts = fetch_reddit_posts( 
        subreddits=["CryptoCurrency", "Bitcoin"],
        limit=5
    )
    
    if posts:
        print(f"\n📝 Sample Post:")
        post = posts[0]
        print(f"   Title: {post.get('title', '')[:60]}...")
        print(f"   Score: {post.get('score')}")
        print(f"   Comments: {post.get('num_comments')}")
        print(f"   Author: {post.get('author')}")
        print(f"   Fields: {list(post.keys())}")