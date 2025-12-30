# reddit fetcher - RAW DATA ONLY
"""
Reddit Fetcher - Extract raw data from Reddit API
All credibility/sentiment analysis happens in services
"""
import os
from dotenv import load_dotenv
import praw
import json
from datetime import datetime
from django.utils import timezone
import time
from pathlib import Path

load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "CryptoNewsBot/1.0")

# Create output directories
SOCIAL_DIR = Path("social")
SOCIAL_DIR.mkdir(exist_ok=True)


def save_to_json(data, filename):
    """Save data to JSON file in social directory"""
    filepath = SOCIAL_DIR / filename
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved to {filepath}")
    except Exception as e:
        print(f"❌ Error saving to {filepath}: {e}")


def fetch_reddit_posts(subreddits=None, limit=50, time_filter="day", sort_type="hot"):
    """
    Fetch cryptocurrency posts from Reddit - RAW DATA ONLY
    
    Args:
        subreddits (list): List of subreddit names
        limit (int): Number of posts per subreddit
        time_filter (str): Time filter for top posts
        sort_type (str): Sort type (hot, new, rising, top)
    
    Returns:
        list: Raw Reddit posts with all available API fields
    """
    if subreddits is None:
        subreddits = ["CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets", "altcoin"]
    
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        print("Reddit API credentials not provided")
        return []
    
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            check_for_async=False
        )
        
        print(f"Fetching Reddit posts from {len(subreddits)} subreddits (raw data)")
        
        all_posts = []
        api_calls = 0
        
        for subreddit_name in subreddits:
            try:
                print(f"  📌 Fetching r/{subreddit_name}...")
                
                subreddit = reddit.subreddit(subreddit_name)
                api_calls += 1
                
                # Get subreddit info (raw)
                subreddit_info = {
                    "name": subreddit.display_name,
                    "subscribers": subreddit.subscribers,
                    "created_utc": subreddit.created_utc,
                    "public_description": getattr(subreddit, 'public_description', ''),
                    "over18": getattr(subreddit, 'over18', False),
                    "subreddit_type": getattr(subreddit, 'subreddit_type', 'public'),
                    "active_user_count": getattr(subreddit, 'active_user_count', 0),
                    "accounts_active": getattr(subreddit, 'accounts_active', 0),
                }
                
                # Get posts based on sort type
                if sort_type == "hot":
                    posts = subreddit.hot(limit=limit)
                elif sort_type == "new":
                    posts = subreddit.new(limit=limit)
                elif sort_type == "rising":
                    posts = subreddit.rising(limit=limit)
                elif sort_type == "top":
                    posts = subreddit.top(time_filter=time_filter, limit=limit)
                else:
                    posts = subreddit.hot(limit=limit)
                
                api_calls += 1
                
                for post in posts:
                    try:
                        # Extract raw author info if available
                        author_info = None
                        if post.author:
                            try:
                                author = post.author
                                author_info = {
                                    "name": str(author),
                                    "link_karma": getattr(author, 'link_karma', 0),
                                    "comment_karma": getattr(author, 'comment_karma', 0),
                                    "created_utc": getattr(author, 'created_utc', 0),
                                    "is_gold": getattr(author, 'is_gold', False),
                                    "is_mod": getattr(author, 'is_mod', False),
                                    "has_verified_email": getattr(author, 'has_verified_email', False),
                                }
                            except Exception:
                                author_info = {"name": str(post.author), "unavailable": True}
                         
                        # Extract ONLY raw API fields
                        post_data = {
                            # === RAW POST FIELDS ===
                            "id": post.id,
                            "title": post.title,
                            "selftext": post.selftext,
                            "url": post.url,
                            "permalink": f"https://reddit.com{post.permalink}",
                            "subreddit": str(post.subreddit),
                            "author": str(post.author) if post.author else "[deleted]",
                            
                            # === RAW TIMESTAMPS ===
                            "created_utc": post.created_utc,
                            
                            # === RAW ENGAGEMENT METRICS (from API) ===
                            "score": post.score,
                            "upvote_ratio": post.upvote_ratio,
                            "num_comments": post.num_comments,
                            "gilded": post.gilded,
                            "total_awards_received": post.total_awards_received,
                            
                            # === RAW POST ATTRIBUTES ===
                            "link_flair_text": post.link_flair_text,
                            "is_self": post.is_self,
                            "is_video": post.is_video,
                            "over_18": post.over_18,
                            "spoiler": post.spoiler,
                            "stickied": post.stickied,
                            "locked": post.locked,
                            "distinguished": post.distinguished,
                            "is_original_content": getattr(post, 'is_original_content', False),
                            
                            # === RAW MEDIA INFO ===
                            "domain": post.domain,
                            "thumbnail": post.thumbnail,
                            "has_media": bool(post.media),
                            
                            # === RAW RELATED DATA ===
                            "author_info": author_info,
                            "subreddit_info": subreddit_info,
                            
                            # === METADATA ===
                            "platform": "reddit",
                            "fetched_at": datetime.now().isoformat()
                        }
                        
                        all_posts.append(post_data)
                        
                    except Exception as e:
                        print(f"    Error extracting post {post.id}: {e}")
                        continue
                
                print(f"    Fetched {len([p for p in all_posts if p.get('subreddit') == subreddit_name])} posts")
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"    Error fetching r/{subreddit_name}: {e}")
                continue
        
        print(f"Total Reddit posts fetched: {len(all_posts)}")
        print(f"API calls made: {api_calls}")
        
        # Save to JSON
        save_to_json(all_posts, "reddit.json")
        
        return all_posts
        
    except Exception as e:
        print(f"Error initializing Reddit client: {e}")
        return []