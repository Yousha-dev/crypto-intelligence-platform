# twitter fetcher - RAW DATA ONLY
"""
Twitter Fetcher - Extract raw data from Twitter API v2
All credibility/sentiment analysis happens in services
"""
import os
from dotenv import load_dotenv
import tweepy
import json
from datetime import datetime, timedelta
from django.utils import timezone
import time
from pathlib import Path

load_dotenv()

TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

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


def fetch_twitter_posts(queries=None, max_results=50, hours_back=24):
    """
    Fetch cryptocurrency tweets using Twitter API v2 - RAW DATA ONLY
    
    Args:
        queries (list): List of search queries
        max_results (int): Maximum results per query (10-100)
        hours_back (int): How many hours back to search
    
    Returns:
        list: Raw tweets with all available API fields
    """
    if not TWITTER_BEARER_TOKEN:
        print("Twitter Bearer Token not provided")
        return []
    
    if queries is None:
        queries = [
            "bitcoin OR BTC cryptocurrency",
            "ethereum OR ETH crypto",
            "cryptocurrency market analysis"
        ]
    
    try:
        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
        
        print(f"Fetching Twitter posts from {len(queries)} queries (raw data)")
        
        all_tweets = []
        start_time = timezone.now() - timedelta(hours=hours_back)
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        api_calls = 0
        
        for query in queries:
            try:
                print(f"  Searching: {query[:50]}...")
                
                response = client.search_recent_tweets(
                    query=f"{query} -is:retweet lang:en",
                    tweet_fields=[
                        'created_at', 'author_id', 'public_metrics',
                        'context_annotations', 'entities', 'geo',
                        'in_reply_to_user_id', 'referenced_tweets',
                        'reply_settings', 'source', 'possibly_sensitive',
                        'conversation_id', 'lang'
                    ],
                    user_fields=[
                        'username', 'name', 'description', 'public_metrics',
                        'verified', 'created_at', 'location', 'profile_image_url',
                        'protected', 'url', 'verified_type'
                    ],
                    expansions=['author_id', 'referenced_tweets.id'],
                    start_time=start_time_str,
                    max_results=min(max_results, 100)
                )
                
                api_calls += 1
                
                if not response.data:
                    continue
                
                # Build user lookup
                users_dict = {}
                if response.includes and 'users' in response.includes:
                    users_dict = {user.id: user for user in response.includes['users']}
                
                for tweet in response.data:
                    try:
                        user_info = users_dict.get(tweet.author_id)
                        
                        # Extract raw user info
                        raw_user_info = None
                        if user_info:
                            raw_user_info = {
                                "id": user_info.id,
                                "username": user_info.username,
                                "name": user_info.name,
                                "description": user_info.description,
                                "verified": user_info.verified,
                                "verified_type": getattr(user_info, 'verified_type', None),
                                "created_at": user_info.created_at.isoformat() if user_info.created_at else None,
                                "location": user_info.location,
                                "profile_image_url": user_info.profile_image_url,
                                "protected": getattr(user_info, 'protected', False),
                                "url": getattr(user_info, 'url', None),
                                "public_metrics": user_info.public_metrics,  # {followers_count, following_count, tweet_count, listed_count}
                            }
                        
                        # Extract ONLY raw API fields
                        tweet_data = {
                            # === RAW TWEET FIELDS ===
                            "id": tweet.id,
                            "text": tweet.text,
                            "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
                            "author_id": tweet.author_id,
                            "conversation_id": tweet.conversation_id,
                            "lang": tweet.lang,
                            "possibly_sensitive": getattr(tweet, 'possibly_sensitive', False),
                            "source": tweet.source,
                            "reply_settings": tweet.reply_settings,
                            "in_reply_to_user_id": tweet.in_reply_to_user_id,
                            
                            # === RAW ENGAGEMENT METRICS (from API) ===
                            "public_metrics": tweet.public_metrics,  # {retweet_count, reply_count, like_count, quote_count}
                            
                            # === RAW CONTEXT DATA ===
                            "context_annotations": tweet.context_annotations,
                            "entities": tweet.entities,
                            "referenced_tweets": [
                                {"type": rt.type, "id": rt.id} 
                                for rt in (tweet.referenced_tweets or [])
                            ] if tweet.referenced_tweets else None,
                            "geo": tweet.geo,
                            
                            # === RAW USER INFO ===
                            "user_info": raw_user_info,
                            
                            # === CONVENIENCE FIELDS (derived from raw) ===
                            "username": user_info.username if user_info else None,
                            "url": f"https://twitter.com/{user_info.username if user_info else 'user'}/status/{tweet.id}",
                            
                            # === METADATA ===
                            "platform": "twitter",
                            "query": query,
                            "fetched_at": datetime.now().isoformat()
                        }
                        
                        all_tweets.append(tweet_data)
                        
                    except Exception as e:
                        print(f"    Error extracting tweet {tweet.id}: {e}")
                        continue
                
                print(f"    Fetched {len([t for t in all_tweets if t.get('query') == query])} tweets")
                
                # Rate limiting
                time.sleep(3)
                
            except tweepy.TooManyRequests:
                print(f"    ⏳ Rate limited, waiting...")
                time.sleep(60)
                continue
            except Exception as e:
                print(f"    Error fetching query '{query}': {e}")
                continue
        
        # Remove duplicates by tweet ID
        unique_tweets = {t['id']: t for t in all_tweets}
        all_tweets = list(unique_tweets.values())
        
        print(f"Total Twitter posts fetched: {len(all_tweets)}")
        print(f"API calls made: {api_calls}")
        
        # Save to JSON
        save_to_json(all_tweets, "twitter.json")
        
        return all_tweets
        
    except Exception as e:
        print(f"Error initializing Twitter client: {e}")
        return []