import os
from dotenv import load_dotenv
import feedparser
import praw
import tweepy
from datetime import datetime, timedelta, timezone
import time
import requests
import json
import base64
from telethon import TelegramClient
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.functions.messages import GetHistoryRequest
import asyncio
import aiohttp

load_dotenv()

CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")

# Reddit API credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "CryptoNewsBot/1.0")

# Twitter API credentials
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

# Additional APIs
CRYPTONEWS_API_KEY = os.getenv("CRYPTONEWS_API_KEY")
SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY")
MESSARI_API_KEY = os.getenv("MESSARI_API_KEY")

# Telegram API credentials
TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Discord Bot Token
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# YouTube API
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# --- CryptoPanic ---
def fetch_cryptopanic_news(currencies=None, filter_type=None, kind="news", size=None, search=None):
    """
    Fetch news from CryptoPanic API with optional filtering
    
    Args:
        currencies (str): Comma-separated currency codes (e.g., "BTC,ETH")
        filter_type (str): Filter type ("rising", "hot", "bullish", "bearish", "important", "saved", "lol")
        kind (str): News kind ("news", "media")
        size (int): Number of items per page (1-500, only available for Enterprise)
        search (str): Search keyword (only available for Enterprise)
    """
    
    if not CRYPTOPANIC_API_KEY:
        print("CryptoPanic API key not provided")
        return []
    
    base_url = "https://cryptopanic.com/api/developer/v2/posts/"
    
    # Build parameters
    params = {
        "auth_token": CRYPTOPANIC_API_KEY,
        "public": "true",  # Use public mode for general apps
        "kind": kind
    }
    
    # Add optional parameters if provided
    if currencies:
        params["currencies"] = currencies
    if filter_type:
        params["filter"] = filter_type
    if size:
        params["size"] = size
    if search:
        params["search"] = search
    
    try:
        print(f"Fetching CryptoPanic news with params: {params}")
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print(f"API Response Status: {response.status_code}")
        print(f"Number of results: {len(data.get('results', []))}")
        
        if "results" not in data:
            print("No 'results' key in API response")
            return []

        # Extract all available fields from the API response
        news_items = []
        for item in data["results"]:
            news_item = {
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
                "content": item.get("content")
            }
            news_items.append(news_item)
        
        # Print first item for debugging
        if news_items:
            print("Sample news item:")
            print(f"Title: {news_items[0]['title']}")
            print(f"Source: {news_items[0]['source']}")
            print(f"Instruments: {news_items[0]['instruments']}")
        
        return news_items

    except requests.exceptions.RequestException as e:
        print(f"Request error fetching CryptoPanic news: {e}")
        return []
    except ValueError as e:
        print(f"JSON parsing error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error fetching CryptoPanic news: {e}")
        return []

# --- Enhanced fetch with pagination support ---
def fetch_all_cryptopanic_news(max_pages=3, **kwargs):
    """
    Fetch multiple pages of CryptoPanic news
    
    Args:
        max_pages (int): Maximum number of pages to fetch
        **kwargs: Additional parameters to pass to fetch_cryptopanic_news
    """
    all_news = []
    base_url = "https://cryptopanic.com/api/developer/v2/posts/"
    
    params = {
        "auth_token": CRYPTOPANIC_API_KEY,
        "public": "true",
        "kind": kwargs.get("kind", "news")
    }
    
    # Add optional parameters
    for key, value in kwargs.items():
        if value is not None:
            params[key] = value
    
    current_url = base_url
    page_count = 0
    
    while current_url and page_count < max_pages:
        try:
            if page_count == 0:
                response = requests.get(current_url, params=params, timeout=10)
            else:
                # For subsequent pages, use the full next URL
                response = requests.get(current_url, timeout=10)
                
            response.raise_for_status()
            data = response.json()
            
            if "results" in data:
                all_news.extend(data["results"])
                print(f"Fetched page {page_count + 1}: {len(data['results'])} items")
            
            # Get next page URL
            current_url = data.get("next")
            page_count += 1
            
        except Exception as e:
            print(f"Error fetching page {page_count + 1}: {e}")
            break
    
    print(f"Total items fetched: {len(all_news)}")
    return all_news

# --- Reddit Crypto Posts ---
def fetch_reddit_crypto_posts(subreddits=None, limit=50, time_filter="day", sort_type="hot"):
    """
    Fetch cryptocurrency posts from Reddit
    
    Args:
        subreddits (list): List of subreddit names (default: crypto-related subreddits)
        limit (int): Number of posts to fetch per subreddit
        time_filter (str): Time filter ("hour", "day", "week", "month", "year", "all")
        sort_type (str): Sort type ("hot", "new", "rising", "top")
    """
    if subreddits is None:
        subreddits = [
            "CryptoCurrency",
            "Bitcoin",
            "ethereum", 
            "CryptoMarkets",
            "altcoin",
            "defi",
            "NFT",
            "dogecoin",
            "binance",
            "cardano"
        ]
    
    try:
        # Initialize Reddit instance
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            check_for_async=False
        )
        
        print(f"Fetching Reddit posts from {len(subreddits)} subreddits")
        all_posts = []
        
        for subreddit_name in subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                print(f"Fetching from r/{subreddit_name}...")
                
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
                
                subreddit_posts = []
                for post in posts:
                    try:
                        # Get post details
                        post_data = {
                            "id": post.id,
                            "title": post.title,
                            "selftext": post.selftext,
                            "url": post.url,
                            "permalink": f"https://reddit.com{post.permalink}",
                            "subreddit": str(post.subreddit),
                            "author": str(post.author) if post.author else "[deleted]",
                            "created_utc": post.created_utc,
                            "created_datetime": datetime.fromtimestamp(post.created_utc),
                            "score": post.score,
                            "upvote_ratio": post.upvote_ratio,
                            "num_comments": post.num_comments,
                            "flair_text": post.link_flair_text,
                            "is_self": post.is_self,
                            "is_video": post.is_video,
                            "over_18": post.over_18,
                            "spoiler": post.spoiler,
                            "stickied": post.stickied,
                            "locked": post.locked,
                            "distinguished": post.distinguished,
                            "gilded": post.gilded,
                            "total_awards_received": post.total_awards_received,
                            "media": post.media,
                            "preview": getattr(post, 'preview', None),
                            "thumbnail": post.thumbnail,
                            "domain": post.domain
                        }
                        subreddit_posts.append(post_data)
                        
                    except Exception as e:
                        print(f"Error processing post {post.id}: {e}")
                        continue
                
                all_posts.extend(subreddit_posts)
                print(f"Fetched {len(subreddit_posts)} posts from r/{subreddit_name}")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error fetching from r/{subreddit_name}: {e}")
                continue
        
        print(f"Total Reddit posts fetched: {len(all_posts)}")
        return all_posts
        
    except Exception as e:
        print(f"Error initializing Reddit client: {e}")
        return []

# --- Reddit Comments Analysis ---
def fetch_reddit_post_comments(post_id, limit=50):
    """
    Fetch comments for a specific Reddit post
    
    Args:
        post_id (str): Reddit post ID
        limit (int): Maximum number of comments to fetch
    """
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            check_for_async=False
        )
        
        submission = reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)  # Remove "more comments" objects
        
        comments = []
        for comment in submission.comments.list()[:limit]:
            try:
                comment_data = {
                    "id": comment.id,
                    "body": comment.body,
                    "author": str(comment.author) if comment.author else "[deleted]",
                    "created_utc": comment.created_utc,
                    "created_datetime": datetime.fromtimestamp(comment.created_utc),
                    "score": comment.score,
                    "parent_id": comment.parent_id,
                    "is_submitter": comment.is_submitter,
                    "stickied": comment.stickied,
                    "gilded": comment.gilded,
                    "total_awards_received": comment.total_awards_received,
                    "depth": comment.depth
                }
                comments.append(comment_data)
                
            except Exception as e:
                print(f"Error processing comment {comment.id}: {e}")
                continue
        
        return comments
        
    except Exception as e:
        print(f"Error fetching comments for post {post_id}: {e}")
        return []

def fetch_twitter_crypto_posts(queries=None, max_results=100, hours_back=24):
    """
    Fetch cryptocurrency tweets using Twitter API v2
    
    Args:
        queries (list): List of search queries (default: crypto-related terms)
        max_results (int): Maximum results per query (10-100 for recent search)
        hours_back (int): How many hours back to search
    """
    if not TWITTER_BEARER_TOKEN:
        print("Twitter Bearer Token not provided")
        return []
        
    if queries is None:
        queries = [
            "bitcoin OR BTC",
            "ethereum OR ETH", 
            "cryptocurrency OR crypto",
            "dogecoin OR DOGE",
            "cardano OR ADA",
            "binance OR BNB",
            "solana OR SOL",
            "polygon OR MATIC",
            "chainlink OR LINK",
            "defi OR \"decentralized finance\""
        ]
    
    try:
        # Initialize Twitter client with Bearer Token (API v2)
        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
        
        print(f"Fetching Twitter posts for {len(queries)} queries")
        all_tweets = []
        
        # Calculate start time - Fix deprecated datetime.utcnow()
        start_time = timezone.now() - timedelta(hours=hours_back)
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        for query in queries:
            try:
                print(f"Searching Twitter for: {query}")
                
                # Search recent tweets
                response = client.search_recent_tweets(
                    query=f"{query} -is:retweet lang:en",  # Exclude retweets, English only
                    tweet_fields=[
                        'created_at', 'author_id', 'public_metrics', 
                        'context_annotations', 'entities', 'geo', 
                        'in_reply_to_user_id', 'referenced_tweets',
                        'reply_settings', 'source', 'withheld'
                    ],
                    user_fields=[
                        'username', 'name', 'description', 'public_metrics',
                        'verified', 'created_at', 'location', 'profile_image_url'
                    ],
                    expansions=['author_id', 'referenced_tweets.id', 'geo.place_id'],
                    start_time=start_time_str,
                    max_results=min(max_results, 100)  # API limit
                )
                
                if not response.data:
                    continue
                
                # Create user lookup dictionary
                users_dict = {}
                if response.includes and 'users' in response.includes:
                    users_dict = {user.id: user for user in response.includes['users']}
                
                query_tweets = []
                for tweet in response.data:
                    try:
                        # Get user info from includes
                        user_info = users_dict.get(tweet.author_id)
                        
                        tweet_data = {
                            "id": tweet.id,
                            "text": tweet.text,
                            "created_at": tweet.created_at,
                            "author_id": tweet.author_id,
                            "query": query,
                            
                            # Public metrics
                            "retweet_count": tweet.public_metrics.get('retweet_count', 0),
                            "like_count": tweet.public_metrics.get('like_count', 0),
                            "reply_count": tweet.public_metrics.get('reply_count', 0),
                            "quote_count": tweet.public_metrics.get('quote_count', 0),
                            
                            # User information
                            "username": user_info.username if user_info else None,
                            "user_name": user_info.name if user_info else None,
                            "user_description": user_info.description if user_info else None,
                            "user_verified": user_info.verified if user_info else None,
                            "user_followers": user_info.public_metrics.get('followers_count', 0) if user_info else 0,
                            "user_following": user_info.public_metrics.get('following_count', 0) if user_info else 0,
                            "user_tweet_count": user_info.public_metrics.get('tweet_count', 0) if user_info else 0,
                            "user_location": user_info.location if user_info else None,
                            "user_profile_image": user_info.profile_image_url if user_info else None,
                            
                            # Tweet metadata
                            "context_annotations": tweet.context_annotations,
                            "entities": tweet.entities,
                            "in_reply_to_user_id": tweet.in_reply_to_user_id,
                            "referenced_tweets": tweet.referenced_tweets,
                            "reply_settings": tweet.reply_settings,
                            "source": tweet.source,
                            "geo": tweet.geo,
                            
                            # Computed fields
                            "engagement_rate": (
                                tweet.public_metrics.get('retweet_count', 0) + 
                                tweet.public_metrics.get('like_count', 0) + 
                                tweet.public_metrics.get('reply_count', 0) + 
                                tweet.public_metrics.get('quote_count', 0)
                            ),
                            "url": f"https://twitter.com/user/status/{tweet.id}"
                        }
                        query_tweets.append(tweet_data)
                        
                    except Exception as e:
                        print(f"Error processing tweet {tweet.id}: {e}")
                        continue
                
                all_tweets.extend(query_tweets)
                print(f"Fetched {len(query_tweets)} tweets for query: {query}")
                
                # Better rate limiting - Twitter API v2 allows 300 requests per 15 minutes
                # That's 1 request every 3 seconds to be safe
                time.sleep(3)
                
            except tweepy.TooManyRequests as e:
                print(f"Rate limit exceeded for query '{query}': {e}")
                print("Waiting 15 minutes before continuing...")
                time.sleep(15 * 60)  # Wait 15 minutes
                continue
            except Exception as e:
                print(f"Error fetching tweets for query '{query}': {e}")
                continue
        
        print(f"Total Twitter posts fetched: {len(all_tweets)}")
        return all_tweets
        
    except Exception as e:
        print(f"Error initializing Twitter client: {e}")
        return []

# --- CoinGecko News and Trends ---
def fetch_coingecko_news_trends():
    """
    Fetch trending searches and news from CoinGecko
    """
    try:
        headers = {}
        base_url = "https://api.coingecko.com/api/v3"
        
        # Fetch trending searches
        trending_response = requests.get(f"{base_url}/search/trending", headers=headers, timeout=10)
        trending_response.raise_for_status()
        trending_data = trending_response.json()
        
        print("CoinGecko trending data fetched successfully")
        
        # Process trending data
        trending_coins = []
        if "coins" in trending_data:
            for coin in trending_data["coins"]:
                coin_data = coin.get("item", {})
                trending_coins.append({
                    "id": coin_data.get("id"),
                    "name": coin_data.get("name"),
                    "symbol": coin_data.get("symbol"),
                    "market_cap_rank": coin_data.get("market_cap_rank"),
                    "thumb": coin_data.get("thumb"),
                    "score": coin_data.get("score")
                })
        
        return {
            "trending_coins": trending_coins,
            "trending_nfts": trending_data.get("nfts", []),
            "trending_categories": trending_data.get("categories", [])
        }
        
    except Exception as e:
        print(f"Error fetching CoinGecko data: {e}")
        return {}

# --- CryptoNews API ---
def fetch_cryptonews_api(query="", page=1, items=50):
    """
    Fetch news from CryptoNews API
    
    Args:
        query (str): Search query
        page (int): Page number
        items (int): Items per page
    """
    try:
        base_url = "https://cryptonews-api.com/api/v1/news"
        params = {
            "tickers": query,
            "items": items,
            "page": page,
            "token": CRYPTONEWS_API_KEY
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print(f"CryptoNews API: Fetched {len(data.get('data', []))} articles")
        
        return [
            {
                "news_url": article.get("news_url"),
                "image_url": article.get("image_url"),
                "title": article.get("title"),
                "text": article.get("text"),
                "source_name": article.get("source_name"),
                "date": article.get("date"),
                "topics": article.get("topics"),
                "sentiment": article.get("sentiment"),
                "type": article.get("type"),
                "tickers": article.get("tickers")
            }
            for article in data.get("data", [])
        ]
        
    except Exception as e:
        print(f"Error fetching CryptoNews API: {e}")
        return []

# --- Santiment Social Data ---
def fetch_santiment_social_data(assets=None, metrics=None, from_date=None, to_date=None):
    """
    Fetch social data from Santiment
    
    Args:
        assets (list): List of asset slugs (e.g., ["bitcoin", "ethereum"])
        metrics (list): List of metrics to fetch
        from_date (str): Start date (ISO format)
        to_date (str): End date (ISO format)
    """
    if not SANTIMENT_API_KEY:
        print("Santiment API key not provided")
        return []
    
    if assets is None:
        assets = ["bitcoin", "ethereum", "cardano", "polkadot", "chainlink"]
    
    if metrics is None:
        metrics = [
            "social_volume_total",
            "social_dominance_total", 
            "sentiment_positive_total",
            "sentiment_negative_total",
            "sentiment_balance_total"
        ]
    
    if not from_date:
        from_date = (datetime.now() - timedelta(days=7)).isoformat()
    if not to_date:
        to_date = datetime.now().isoformat()
    
    try:
        headers = {
            "Authorization": f"Apikey {SANTIMENT_API_KEY}",
            "Content-Type": "application/json"
        }
        
        query = """
        query($slug: String!, $metric: String!, $from: DateTime!, $to: DateTime!) {
          getMetric(metric: $metric) {
            timeseriesData(
              slug: $slug
              from: $from
              to: $to
              interval: "1d"
            ) {
              datetime
              value
            }
          }
        }
        """
        
        all_data = []
        
        for asset in assets:
            for metric in metrics:
                try:
                    variables = {
                        "slug": asset,
                        "metric": metric,
                        "from": from_date,
                        "to": to_date
                    }
                    
                    response = requests.post(
                        "https://api.santiment.net/graphql",
                        json={"query": query, "variables": variables},
                        headers=headers,
                        timeout=15
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    if "data" in data and data["data"]["getMetric"]["timeseriesData"]:
                        for point in data["data"]["getMetric"]["timeseriesData"]:
                            all_data.append({
                                "asset": asset,
                                "metric": metric,
                                "datetime": point["datetime"],
                                "value": point["value"]
                            })
                    
                    time.sleep(0.2)  # Rate limiting
                    
                except Exception as e:
                    print(f"Error fetching {metric} for {asset}: {e}")
                    continue
        
        print(f"Santiment: Fetched {len(all_data)} data points")
        return all_data
        
    except Exception as e:
        print(f"Error fetching Santiment data: {e}")
        return []

# --- Messari News ---
def fetch_messari_news(fields=None, limit=50):
    """
    Fetch news from Messari
    
    Args:
        fields (list): List of fields to include
        limit (int): Number of news items to fetch
    """
    try:
        headers = {}
        if MESSARI_API_KEY:
            headers["x-messari-api-key"] = MESSARI_API_KEY
        
        url = "https://data.messari.io/api/v1/news"
        params = {"limit": limit}
        
        if fields:
            params["fields"] = ",".join(fields)
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        news_items = []
        for item in data.get("data", []):
            news_items.append({
                "id": item.get("id"),
                "title": item.get("title"),
                "content": item.get("content"),
                "references": item.get("references", []),
                "reference_title": item.get("reference_title"),
                "published_at": item.get("published_at"),
                "author": item.get("author", {}).get("name"),
                "tags": item.get("tags", []),
                "url": item.get("url")
            })
        
        print(f"Messari: Fetched {len(news_items)} news items")
        return news_items
        
    except Exception as e:
        print(f"Error fetching Messari news: {e}")
        return []

# --- YouTube Crypto Content ---
def fetch_youtube_crypto_content(query="cryptocurrency", max_results=25, published_after=None):
    """
    Fetch crypto-related content from YouTube
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of results
        published_after (str): ISO 8601 datetime string
    """
    if not YOUTUBE_API_KEY:
        print("YouTube API key not provided")
        return []
    
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "key": YOUTUBE_API_KEY,
            "order": "relevance",
            "relevanceLanguage": "en"
        }
        
        if published_after:
            params["publishedAfter"] = published_after
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        videos = []
        video_ids = []
        
        for item in data.get("items", []):
            video_id = item["id"]["videoId"]
            video_ids.append(video_id)
            
            videos.append({
                "video_id": video_id,
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "channel_title": item["snippet"]["channelTitle"],
                "channel_id": item["snippet"]["channelId"],
                "published_at": item["snippet"]["publishedAt"],
                "thumbnail": item["snippet"]["thumbnails"].get("high", {}).get("url"),
                "url": f"https://www.youtube.com/watch?v={video_id}"
            })
        
        # Get video statistics
        if video_ids:
            stats_url = "https://www.googleapis.com/youtube/v3/videos"
            stats_params = {
                "part": "statistics,contentDetails",
                "id": ",".join(video_ids),
                "key": YOUTUBE_API_KEY
            }
            
            stats_response = requests.get(stats_url, params=stats_params, timeout=10)
            if stats_response.status_code == 200:
                stats_data = stats_response.json()
                stats_dict = {item["id"]: item for item in stats_data.get("items", [])}
                
                # Add statistics to videos
                for video in videos:
                    video_id = video["video_id"]
                    if video_id in stats_dict:
                        stats = stats_dict[video_id]["statistics"]
                        video.update({
                            "view_count": int(stats.get("viewCount", 0)),
                            "like_count": int(stats.get("likeCount", 0)),
                            "comment_count": int(stats.get("commentCount", 0)),
                            "duration": stats_dict[video_id]["contentDetails"].get("duration")
                        })
        
        print(f"YouTube: Fetched {len(videos)} videos")
        return videos
        
    except Exception as e:
        print(f"Error fetching YouTube content: {e}")
        return []

# --- Telegram Channel Posts ---
async def fetch_telegram_crypto_posts_async(channels=None, limit=50):
    """
    Fetch posts from Telegram crypto channels
    
    Args:
        channels (list): List of channel usernames
        limit (int): Number of posts per channel
    """
    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
        print("Telegram API credentials not provided")
        return []
    
    if channels is None:
        channels = [
            "cryptonews",
            "bitcoinnews", 
            "ethereum_news",
            "binanceexchange",
            "CoinDeskNews",
            "cointelegraph"
        ]
    
    try:
        client = TelegramClient('session', TELEGRAM_API_ID, TELEGRAM_API_HASH)
        await client.start()
        
        all_posts = []
        
        for channel in channels:
            try:
                entity = await client.get_entity(channel)
                posts = await client(GetHistoryRequest(
                    peer=entity,
                    limit=limit,
                    offset_date=None,
                    offset_id=0,
                    max_id=0,
                    min_id=0,
                    add_offset=0,
                    hash=0
                ))
                
                channel_posts = []
                for message in posts.messages:
                    if message.message:  # Only text messages
                        post_data = {
                            "id": message.id,
                            "message": message.message,
                            "date": message.date,
                            "channel": channel,
                            "views": message.views,
                            "forwards": message.forwards,
                            "replies": message.replies.replies if message.replies else 0,
                            "media": bool(message.media),
                            "url": f"https://t.me/{channel}/{message.id}"
                        }
                        channel_posts.append(post_data)
                
                all_posts.extend(channel_posts)
                print(f"Telegram: Fetched {len(channel_posts)} posts from {channel}")
                
                await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error fetching from Telegram channel {channel}: {e}")
                continue
        
        await client.disconnect()
        return all_posts
        
    except Exception as e:
        print(f"Error with Telegram client: {e}")
        return []

def fetch_telegram_crypto_posts(channels=None, limit=50):
    """
    Synchronous wrapper for Telegram fetching
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(fetch_telegram_crypto_posts_async(channels, limit))
    except Exception as e:
        print(f"Error running Telegram async function: {e}")
        return []
def fetch_twitter_user_timeline(username, max_results=50):
    """
    Fetch recent tweets from a specific user
    
    Args:
        username (str): Twitter username (without @)
        max_results (int): Maximum number of tweets to fetch
    """
    try:
        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
        
        # Get user ID
        user = client.get_user(username=username)
        if not user.data:
            print(f"User {username} not found")
            return []
        
        # Get user timeline
        tweets = client.get_users_tweets(
            id=user.data.id,
            tweet_fields=[
                'created_at', 'public_metrics', 'context_annotations',
                'entities', 'referenced_tweets'
            ],
            max_results=max_results
        )
        
        if not tweets.data:
            return []
        
        timeline_tweets = []
        for tweet in tweets.data:
            tweet_data = {
                "id": tweet.id,
                "text": tweet.text,
                "created_at": tweet.created_at,
                "username": username,
                "retweet_count": tweet.public_metrics.get('retweet_count', 0),
                "like_count": tweet.public_metrics.get('like_count', 0),
                "reply_count": tweet.public_metrics.get('reply_count', 0),
                "quote_count": tweet.public_metrics.get('quote_count', 0),
                "context_annotations": tweet.context_annotations,
                "entities": tweet.entities,
                "referenced_tweets": tweet.referenced_tweets,
                "url": f"https://twitter.com/{username}/status/{tweet.id}"
            }
            timeline_tweets.append(tweet_data)
        
        return timeline_tweets
        
    except Exception as e:
        print(f"Error fetching timeline for @{username}: {e}")
        return []
def fetch_cryptocompare_news(categories=None, excludeCategories=None, feeds=None, lTs=None, sortOrder="latest", lang="EN"):
    """
    Fetch news from CryptoCompare API
    
    Args:
        categories (str): Comma separated list of categories to get news from (e.g., "BTC,ETH,Trading")
        excludeCategories (str): Comma separated list of categories to exclude
        feeds (str): Comma separated list of specific news sources
        lTs (int): Timestamp to get news after this time
        sortOrder (str): Order of results ("latest" or "popular")
        lang (str): Language for news ("EN", "DE", "FR", "IT", "PT", "RU", "ES", "ZH")
    """
    base_url = "https://min-api.cryptocompare.com/data/v2/news/"
    
    # Build parameters
    params = {
        "sortOrder": sortOrder,
        "lang": lang
    }
    
    # Add API key if available (some endpoints work without it)
    if CRYPTOCOMPARE_API_KEY:
        params["api_key"] = CRYPTOCOMPARE_API_KEY
    
    # Add optional parameters if provided
    if categories:
        params["categories"] = categories
    if excludeCategories:
        params["excludeCategories"] = excludeCategories
    if feeds:
        params["feeds"] = feeds
    if lTs:
        params["lTs"] = lTs
    
    try:
        print(f"Fetching CryptoCompare news with params: {params}")
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print(f"CryptoCompare API Response Status: {response.status_code}")
        
        # Check for API errors
        if data.get("Response") == "Error":
            print(f"CryptoCompare API Error: {data.get('Message', 'Unknown error')}")
            return []
        
        news_data = data.get("Data", [])
        print(f"Number of CryptoCompare news items: {len(news_data)}")
        
        # Extract news items
        news_items = []
        for item in news_data:
            news_item = {
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
                "source_info": item.get("source_info")
            }
            news_items.append(news_item)
        
        # Print first item for debugging
        if news_items:
            print("Sample CryptoCompare news item:")
            print(f"Title: {news_items[0]['title']}")
            print(f"Source: {news_items[0]['source']}")
            print(f"Categories: {news_items[0]['categories']}")
        
        return news_items

    except requests.exceptions.RequestException as e:
        print(f"Request error fetching CryptoCompare news: {e}")
        return []
    except ValueError as e:
        print(f"JSON parsing error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error fetching CryptoCompare news: {e}")
        return []
def fetch_newsapi_articles(query="cryptocurrency", page_size=20):
    """
    Fetch articles from NewsAPI
    
    Args:
        query (str): Search query
        page_size (int): Number of articles per page
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": NEWSAPI_API_KEY,
        "pageSize": page_size,
        "sortBy": "publishedAt"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            print(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return []

        return [
            {
                "source": article.get("source"),
                "author": article.get("author"),
                "title": article.get("title"),
                "description": article.get("description"),
                "url": article.get("url"),
                "urlToImage": article.get("urlToImage"),
                "publishedAt": article.get("publishedAt"),
                "content": article.get("content")
            }
            for article in data.get("articles", [])
        ]

    except Exception as e:
        print(f"Error fetching NewsAPI articles: {e}")
        return []

# --- CoinDesk RSS ---
def scrape_coindesk():
    """Scrape CoinDesk RSS feed"""
    url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    try:
        feed = feedparser.parse(url)
        if feed.bozo:
            print("Warning: CoinDesk RSS feed may have parsing issues")
            
        return [
            {
                "title": entry.title,
                "link": entry.link,
                "published": getattr(entry, "published", None),
                "summary": getattr(entry, "summary", None),
                "media_content": entry.get("media_content", [])
            }
            for entry in feed.entries
        ]
    except Exception as e:
        print(f"Error scraping CoinDesk: {e}")
        return []

# --- Analysis Functions ---
def analyze_all_sources(news_data):
    """
    Comprehensive analysis of all news sources
    
    Args:
        news_data (dict): Dictionary containing all news sources data
    """
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "total_sources": len(news_data),
        "source_stats": {},
        "content_analysis": {},
        "social_sentiment": {},
        "trending_topics": {},
        "market_indicators": {}
    }
    
    # Analyze each source
    for source_name, source_data in news_data.items():
        if isinstance(source_data, list) and source_data:
            analysis["source_stats"][source_name] = {
                "total_items": len(source_data),
                "avg_content_length": calculate_avg_content_length(source_data),
                "time_range": get_time_range(source_data),
                "unique_sources": get_unique_sources(source_data)
            }
    
    # Social sentiment analysis
    if "reddit" in news_data and "twitter" in news_data:
        analysis["social_sentiment"] = analyze_comprehensive_sentiment(
            news_data.get("reddit", []),
            news_data.get("twitter", [])
        )
    
    # Content analysis
    all_content = extract_all_content(news_data)
    analysis["content_analysis"] = analyze_content_themes(all_content)
    
    # Market indicators
    analysis["market_indicators"] = extract_market_indicators(news_data)
    
    # Trending topics
    analysis["trending_topics"] = extract_trending_topics(all_content)
    
    return analysis

def calculate_avg_content_length(data_list):
    """Calculate average content length for a data source"""
    total_length = 0
    count = 0
    
    for item in data_list:
        content_fields = ['title', 'description', 'content', 'text', 'body', 'selftext', 'message']
        for field in content_fields:
            if field in item and item[field]:
                total_length += len(str(item[field]))
                count += 1
                break
    
    return total_length / count if count > 0 else 0

def get_time_range(data_list):
    """Get time range of data"""
    timestamps = []
    time_fields = ['published_at', 'created_at', 'created_utc', 'publishedAt', 'date', 'published_on']
    
    for item in data_list:
        for field in time_fields:
            if field in item and item[field]:
                try:
                    if field == 'created_utc':
                        timestamps.append(datetime.fromtimestamp(item[field]))
                    elif isinstance(item[field], str):
                        # Try parsing various date formats
                        try:
                            timestamps.append(datetime.fromisoformat(item[field].replace('Z', '+00:00')))
                        except:
                            timestamps.append(datetime.fromisoformat(item[field]))
                    break
                except:
                    continue
    
    if timestamps:
        return {
            "earliest": min(timestamps).isoformat(),
            "latest": max(timestamps).isoformat(),
            "span_hours": (max(timestamps) - min(timestamps)).total_seconds() / 3600
        }
    return None

def get_unique_sources(data_list):
    """Get unique sources from data"""
    sources = set()
    source_fields = ['source', 'source_name', 'subreddit', 'channel', 'username']
    
    for item in data_list:
        for field in source_fields:
            if field in item and item[field]:
                sources.add(str(item[field]))
                break
    
    return list(sources)

def analyze_comprehensive_sentiment(reddit_data, twitter_data):
    """Comprehensive sentiment analysis of social media data"""
    sentiment = {
        "reddit": analyze_reddit_sentiment(reddit_data),
        "twitter": analyze_twitter_sentiment(twitter_data),
        "overall_market": {}
    }
    
    # Calculate overall market sentiment
    reddit_positive = sentiment["reddit"].get("positive_ratio", 0.5)
    twitter_positive = sentiment["twitter"].get("positive_ratio", 0.5)
    
    sentiment["overall_market"] = {
        "positive_ratio": (reddit_positive + twitter_positive) / 2,
        "confidence_score": calculate_confidence_score(reddit_data, twitter_data),
        "market_mood": determine_market_mood((reddit_positive + twitter_positive) / 2)
    }
    
    return sentiment

def analyze_reddit_sentiment(reddit_data):
    """Analyze Reddit sentiment indicators"""
    if not reddit_data:
        return {}
    
    total_score = sum(post.get("score", 0) for post in reddit_data)
    total_comments = sum(post.get("num_comments", 0) for post in reddit_data)
    avg_upvote_ratio = sum(post.get("upvote_ratio", 0.5) for post in reddit_data) / len(reddit_data)
    
    # Simple sentiment indicators
    positive_posts = len([post for post in reddit_data if post.get("score", 0) > 10])
    negative_posts = len([post for post in reddit_data if post.get("score", 0) < 0])
    
    return {
        "total_posts": len(reddit_data),
        "avg_score": total_score / len(reddit_data),
        "avg_comments": total_comments / len(reddit_data),
        "avg_upvote_ratio": avg_upvote_ratio,
        "positive_posts": positive_posts,
        "negative_posts": negative_posts,
        "positive_ratio": positive_posts / len(reddit_data) if reddit_data else 0,
        "engagement_score": (total_score + total_comments) / len(reddit_data),
        "sentiment_indicator": "Bullish" if avg_upvote_ratio > 0.7 else "Bearish" if avg_upvote_ratio < 0.5 else "Neutral"
    }

def analyze_twitter_sentiment(twitter_data):
    """Analyze Twitter sentiment indicators"""
    if not twitter_data:
        return {}
    
    total_likes = sum(tweet.get("like_count", 0) for tweet in twitter_data)
    total_retweets = sum(tweet.get("retweet_count", 0) for tweet in twitter_data)
    total_engagement = sum(tweet.get("engagement_rate", 0) for tweet in twitter_data)
    verified_tweets = len([tweet for tweet in twitter_data if tweet.get("user_verified")])
    
    # High engagement tweets (above average)
    avg_engagement = total_engagement / len(twitter_data) if twitter_data else 0
    high_engagement = len([tweet for tweet in twitter_data if tweet.get("engagement_rate", 0) > avg_engagement])
    
    return {
        "total_tweets": len(twitter_data),
        "avg_likes": total_likes / len(twitter_data),
        "avg_retweets": total_retweets / len(twitter_data),
        "avg_engagement": avg_engagement,
        "verified_tweets": verified_tweets,
        "high_engagement_tweets": high_engagement,
        "positive_ratio": high_engagement / len(twitter_data) if twitter_data else 0,
        "influence_score": (verified_tweets / len(twitter_data)) * 100 if twitter_data else 0,
        "viral_potential": calculate_viral_potential(twitter_data)
    }

def calculate_confidence_score(reddit_data, twitter_data):
    """Calculate confidence score based on data volume and engagement"""
    reddit_score = len(reddit_data) * 0.3 if reddit_data else 0
    twitter_score = len(twitter_data) * 0.7 if twitter_data else 0
    
    total_score = reddit_score + twitter_score
    return min(total_score / 100, 1.0)  # Normalize to 0-1

def determine_market_mood(positive_ratio):
    """Determine overall market mood"""
    if positive_ratio > 0.7:
        return "Very Bullish"
    elif positive_ratio > 0.6:
        return "Bullish"
    elif positive_ratio > 0.4:
        return "Neutral"
    elif positive_ratio > 0.3:
        return "Bearish"
    else:
        return "Very Bearish"

def calculate_viral_potential(twitter_data):
    """Calculate viral potential of tweets"""
    if not twitter_data:
        return 0
    
    high_engagement = [tweet for tweet in twitter_data if tweet.get("engagement_rate", 0) > 100]
    viral_score = len(high_engagement) / len(twitter_data) * 100
    return min(viral_score, 100)

def extract_all_content(news_data):
    """Extract all text content from news data"""
    all_content = []
    
    for source_name, source_data in news_data.items():
        if isinstance(source_data, list):
            for item in source_data:
                content_fields = ['title', 'description', 'content', 'text', 'body', 'selftext', 'message']
                for field in content_fields:
                    if field in item and item[field]:
                        all_content.append({
                            "source": source_name,
                            "content": str(item[field]),
                            "timestamp": get_item_timestamp(item)
                        })
                        break
    
    return all_content

def get_item_timestamp(item):
    """Extract timestamp from an item"""
    time_fields = ['published_at', 'created_at', 'created_utc', 'publishedAt', 'date', 'published_on']
    
    for field in time_fields:
        if field in item and item[field]:
            try:
                if field == 'created_utc':
                    return datetime.fromtimestamp(item[field]).isoformat()
                elif isinstance(item[field], str):
                    return item[field]
            except:
                continue
    
    return datetime.now().isoformat()

def analyze_content_themes(content_list):
    """Analyze content themes and keywords"""
    import re
    from collections import Counter
    
    # Common crypto keywords
    crypto_keywords = [
        'bitcoin', 'btc', 'ethereum', 'eth', 'cryptocurrency', 'crypto', 'blockchain',
        'defi', 'nft', 'trading', 'market', 'price', 'bull', 'bear', 'hodl',
        'altcoin', 'mining', 'wallet', 'exchange', 'doge', 'ada', 'sol', 'matic'
    ]
    
    all_words = []
    keyword_counts = Counter()
    
    for content_item in content_list:
        content = content_item['content'].lower()
        words = re.findall(r'\b[a-zA-Z]+\b', content)
        all_words.extend(words)
        
        # Count crypto keywords
        for keyword in crypto_keywords:
            keyword_counts[keyword] += content.count(keyword)
    
    # Get most common words (excluding common words)
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
    filtered_words = [word for word in all_words if word not in common_words and len(word) > 2]
    
    word_counts = Counter(filtered_words)
    
    return {
        "top_keywords": dict(keyword_counts.most_common(10)),
        "top_words": dict(word_counts.most_common(20)),
        "total_content_items": len(content_list),
        "avg_content_length": sum(len(item['content']) for item in content_list) / len(content_list) if content_list else 0,
        "sources_breakdown": Counter(item['source'] for item in content_list)
    }

def extract_market_indicators(news_data):
    """Extract market indicators from news data"""
    indicators = {
        "news_volume": {},
        "source_activity": {},
        "trending_coins": [],
        "market_sentiment_signals": {}
    }
    
    # News volume by source
    for source_name, source_data in news_data.items():
        if isinstance(source_data, list):
            indicators["news_volume"][source_name] = len(source_data)
    
    # Extract trending coins from CoinGecko data
    if "coingecko_trends" in news_data and news_data["coingecko_trends"]:
        indicators["trending_coins"] = news_data["coingecko_trends"].get("trending_coins", [])
    
    # Market sentiment signals
    if "cryptopanic" in news_data and news_data["cryptopanic"]:
        panic_scores = [item.get("panic_score", 0) for item in news_data["cryptopanic"] if item.get("panic_score")]
        if panic_scores:
            indicators["market_sentiment_signals"]["avg_panic_score"] = sum(panic_scores) / len(panic_scores)
            indicators["market_sentiment_signals"]["max_panic_score"] = max(panic_scores)
    
    return indicators

def extract_trending_topics(content_list):
    """Extract trending topics from content"""
    import re
    from collections import Counter
    
    # Extract hashtags and mentions
    hashtags = []
    mentions = []
    
    for content_item in content_list:
        content = content_item['content']
        hashtags.extend(re.findall(r'#\w+', content))
        mentions.extend(re.findall(r'@\w+', content))
    
    return {
        "trending_hashtags": dict(Counter(hashtags).most_common(10)),
        "trending_mentions": dict(Counter(mentions).most_common(10)),
        "topic_clusters": identify_topic_clusters(content_list)
    }

def identify_topic_clusters(content_list):
    """Identify topic clusters in content"""
    # Simple topic clustering based on common keywords
    topics = {
        "trading": ["trading", "trade", "buy", "sell", "market", "price"],
        "technology": ["blockchain", "protocol", "smart contract", "defi", "nft"],
        "regulation": ["regulation", "sec", "government", "legal", "compliance"],
        "adoption": ["adoption", "institutional", "company", "investment", "fund"]
    }
    
    topic_scores = {topic: 0 for topic in topics}
    
    for content_item in content_list:
        content = content_item['content'].lower()
        for topic, keywords in topics.items():
            for keyword in keywords:
                topic_scores[topic] += content.count(keyword)
    
    return topic_scores

def generate_summary_report(analysis_data):
    """Generate a comprehensive summary report"""
    report = []
    report.append("=" * 50)
    report.append("CRYPTOCURRENCY NEWS & SOCIAL MEDIA ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"Generated: {analysis_data.get('timestamp', 'Unknown')}")
    report.append(f"Total Sources Analyzed: {analysis_data.get('total_sources', 0)}")
    report.append("")
    
    # Source Statistics
    report.append("SOURCE STATISTICS")
    report.append("-" * 20)
    source_stats = analysis_data.get('source_stats', {})
    for source, stats in source_stats.items():
        report.append(f"{source.upper()}: {stats['total_items']} items")
        if stats.get('time_range'):
            report.append(f"  Time span: {stats['time_range']['span_hours']:.1f} hours")
        report.append(f"  Avg content length: {stats['avg_content_length']:.0f} characters")
    report.append("")
    
    # Social Sentiment
    social_sentiment = analysis_data.get('social_sentiment', {})
    if social_sentiment:
        report.append("SOCIAL SENTIMENT ANALYSIS")
        report.append("-" * 25)
        
        if 'overall_market' in social_sentiment:
            overall = social_sentiment['overall_market']
            report.append(f"Overall Market Mood: {overall.get('market_mood', 'Unknown')}")
            report.append(f"Positive Sentiment Ratio: {overall.get('positive_ratio', 0):.1%}")
            report.append(f"Confidence Score: {overall.get('confidence_score', 0):.2f}")
        
        if 'reddit' in social_sentiment:
            reddit = social_sentiment['reddit']
            report.append(f"\nReddit Analysis:")
            report.append(f"  Posts analyzed: {reddit.get('total_posts', 0)}")
            report.append(f"  Average score: {reddit.get('avg_score', 0):.1f}")
            report.append(f"  Sentiment indicator: {reddit.get('sentiment_indicator', 'Unknown')}")
        
        if 'twitter' in social_sentiment:
            twitter = social_sentiment['twitter']
            report.append(f"\nTwitter Analysis:")
            report.append(f"  Tweets analyzed: {twitter.get('total_tweets', 0)}")
            report.append(f"  Average engagement: {twitter.get('avg_engagement', 0):.1f}")
            report.append(f"  Influence score: {twitter.get('influence_score', 0):.1f}%")
    report.append("")
    
    # Content Analysis
    content_analysis = analysis_data.get('content_analysis', {})
    if content_analysis:
        report.append("CONTENT ANALYSIS")
        report.append("-" * 16)
        report.append(f"Total content items: {content_analysis.get('total_content_items', 0)}")
        report.append(f"Average content length: {content_analysis.get('avg_content_length', 0):.0f} characters")
        
        top_keywords = content_analysis.get('top_keywords', {})
        if top_keywords:
            report.append("\nTop Crypto Keywords:")
            for keyword, count in list(top_keywords.items())[:5]:
                report.append(f"  {keyword}: {count} mentions")
    report.append("")
    
    # Market Indicators
    market_indicators = analysis_data.get('market_indicators', {})
    if market_indicators:
        report.append("MARKET INDICATORS")
        report.append("-" * 17)
        
        trending_coins = market_indicators.get('trending_coins', [])
        if trending_coins:
            report.append("Top Trending Coins:")
            for i, coin in enumerate(trending_coins[:5], 1):
                report.append(f"  {i}. {coin.get('name', 'Unknown')} ({coin.get('symbol', 'N/A')})")
        
        sentiment_signals = market_indicators.get('market_sentiment_signals', {})
        if sentiment_signals:
            avg_panic = sentiment_signals.get('avg_panic_score')
            if avg_panic:
                report.append(f"\nAverage Panic Score: {avg_panic:.2f}")
    report.append("")
    
    # Trending Topics
    trending_topics = analysis_data.get('trending_topics', {})
    if trending_topics:
        report.append("TRENDING TOPICS")
        report.append("-" * 15)
        
        topic_clusters = trending_topics.get('topic_clusters', {})
        if topic_clusters:
            report.append("Topic Activity:")
            for topic, score in sorted(topic_clusters.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {topic.capitalize()}: {score} mentions")
    
    report.append("")
    report.append("=" * 50)
    report.append("End of Report")
    report.append("=" * 50)
    
    return "\n".join(report)

def export_to_json(data, filename):
    """Export data to JSON file"""
    import json
    
    try:
        # Convert datetime objects to strings for JSON serialization
        json_data = convert_datetime_to_string(data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Data exported to {filename}")
        return filename
        
    except Exception as e:
        print(f"Error exporting to JSON: {e}")
        return None

def convert_datetime_to_string(obj):
    """Recursively convert datetime objects to strings"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_datetime_to_string(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetime_to_string(item) for item in obj]
    else:
        return obj

# --- Main Fetch Function ---
def fetch_news(cryptopanic_params=None, reddit_params=None, twitter_params=None):
    """
    Fetch news from multiple sources including social media
    
    Args:
        cryptopanic_params (dict): Parameters for CryptoPanic API
        reddit_params (dict): Parameters for Reddit API
        twitter_params (dict): Parameters for Twitter API
    """
    if cryptopanic_params is None:
        cryptopanic_params = {
            "filter_type": "important",
            "kind": "news"
        }
    
    if reddit_params is None:
        reddit_params = {
            "limit": 25,
            "time_filter": "day",
            "sort_type": "hot"
        }
    
    if twitter_params is None:
        twitter_params = {
            "max_results": 50,
            "hours_back": 24
        }
    
    return {
        "cryptopanic": fetch_cryptopanic_news(**cryptopanic_params),
        "cryptocompare": fetch_cryptocompare_news(),
        "newsapi": fetch_newsapi_articles(),
        "coindesk": scrape_coindesk(),
        "reddit": fetch_reddit_crypto_posts(**reddit_params),
        "twitter": fetch_twitter_crypto_posts(**twitter_params)
    }

# --- Social Media Analytics ---
def analyze_social_sentiment(social_data):
    """
    Basic analysis of social media data
    
    Args:
        social_data (dict): Dictionary containing reddit and twitter data
    """
    analysis = {
        "reddit": {},
        "twitter": {}
    }
    
    # Reddit analysis
    if "reddit" in social_data and social_data["reddit"]:
        reddit_posts = social_data["reddit"]
        analysis["reddit"] = {
            "total_posts": len(reddit_posts),
            "total_score": sum(post.get("score", 0) for post in reddit_posts),
            "total_comments": sum(post.get("num_comments", 0) for post in reddit_posts),
            "avg_score": sum(post.get("score", 0) for post in reddit_posts) / len(reddit_posts),
            "avg_upvote_ratio": sum(post.get("upvote_ratio", 0) for post in reddit_posts) / len(reddit_posts),
            "top_subreddits": {},
            "top_posts": sorted(reddit_posts, key=lambda x: x.get("score", 0), reverse=True)[:5]
        }
        
        # Count posts by subreddit
        for post in reddit_posts:
            subreddit = post.get("subreddit", "unknown")
            analysis["reddit"]["top_subreddits"][subreddit] = analysis["reddit"]["top_subreddits"].get(subreddit, 0) + 1
    
    # Twitter analysis
    if "twitter" in social_data and social_data["twitter"]:
        twitter_posts = social_data["twitter"]
        analysis["twitter"] = {
            "total_tweets": len(twitter_posts),
            "total_likes": sum(tweet.get("like_count", 0) for tweet in twitter_posts),
            "total_retweets": sum(tweet.get("retweet_count", 0) for tweet in twitter_posts),
            "total_replies": sum(tweet.get("reply_count", 0) for tweet in twitter_posts),
            "avg_engagement": sum(tweet.get("engagement_rate", 0) for tweet in twitter_posts) / len(twitter_posts),
            "top_tweets": sorted(twitter_posts, key=lambda x: x.get("engagement_rate", 0), reverse=True)[:5],
            "verified_users": len([tweet for tweet in twitter_posts if tweet.get("user_verified")])
        }
    
    return analysis
    
# --- Example Usage ---
if __name__ == "__main__":
    # Test different configurations
    # print("=== Testing CryptoPanic Basic ===")
    # basic_news = fetch_cryptopanic_news()
    # print(f"Basic fetch: {len(basic_news)} items")
    
    # print("\n=== Testing CryptoPanic with BTC filter ===")
    # btc_news = fetch_cryptopanic_news(currencies="BTC", filter_type="important")
    # print(f"BTC important news: {len(btc_news)} items")
    
    # print("\n=== Testing CryptoCompare News ===")
    # cc_news = fetch_cryptocompare_news(categories="BTC,ETH")
    # print(f"CryptoCompare news: {len(cc_news)} items")
    
    # print("\n=== Testing CoinGecko Trends ===")
    # cg_trends = fetch_coingecko_news_trends()
    # print(f"CoinGecko trending coins: {len(cg_trends.get('trending_coins', []))} items")
    
    # print("\n=== Testing YouTube Content ===")
    # youtube_videos = fetch_youtube_crypto_content(query="bitcoin news", max_results=10)
    # print(f"YouTube videos: {len(youtube_videos)} items")
    
    # print("\n=== Testing Messari News ===")
    # messari_news = fetch_messari_news(limit=10)
    # print(f"Messari news: {len(messari_news)} items")
    
    # print("\n=== Testing Reddit Posts ===")
    # reddit_posts = fetch_reddit_crypto_posts(subreddits=["CryptoCurrency", "Bitcoin"], limit=10)
    # print(f"Reddit posts: {len(reddit_posts)} items")
    
    # print("\n=== Testing Twitter Posts ===")
    # twitter_posts = fetch_twitter_crypto_posts(queries=["bitcoin OR BTC"], max_results=20)
    # print(f"Twitter posts: {len(twitter_posts)} items")
    
    print("\n=== Testing All Sources ===")
    all_news = fetch_news()
    for source, articles in all_news.items():
        if isinstance(articles, list):
            print(f"{source}: {len(articles)} items")
        elif isinstance(articles, dict):
            print(f"{source}: {sum(len(v) if isinstance(v, list) else 1 for v in articles.values())} items")
        else:
            print(f"{source}: {'✓' if articles else '✗'}")
    
    # Comprehensive analysis
    print("\n=== Comprehensive Analysis ===")
    analysis_results = analyze_all_sources(all_news)
    
    # Generate and print summary report
    summary_report = generate_summary_report(analysis_results)
    print(summary_report)
    
    # Export data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file = export_to_json(all_news, f"crypto_news_{timestamp}.json")
    analysis_file = export_to_json(analysis_results, f"crypto_analysis_{timestamp}.json")
    
    # Save report to file
    try:
        with open(f"crypto_report_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write(summary_report)
        print(f"\nReport saved to crypto_report_{timestamp}.txt")
    except Exception as e:
        print(f"Error saving report: {e}")
    
    print(f"\nData exported to: {json_file}")
    print(f"Analysis exported to: {analysis_file}")
    
    # Print quick stats
    total_items = sum(len(articles) if isinstance(articles, list) else 
                     (sum(len(v) if isinstance(v, list) else 1 for v in articles.values()) 
                      if isinstance(articles, dict) else (1 if articles else 0))
                     for articles in all_news.values())
    print(f"\nTotal items collected: {total_items}")
    
    # Print top trending coins if available
    if 'coingecko_trends' in all_news and all_news['coingecko_trends'].get('trending_coins'):
        print("\nTop Trending Coins:")
        for i, coin in enumerate(all_news['coingecko_trends']['trending_coins'][:5], 1):
            print(f"{i}. {coin['name']} ({coin['symbol']})")
    
    # Print social sentiment summary
    social_sentiment = analysis_results.get("social_sentiment", {})
    if social_sentiment.get("overall_market"):
        positive_ratio = social_sentiment["overall_market"]["positive_ratio"]
        print(f"\nOverall Market Sentiment: {positive_ratio:.1%} positive")
        
    if social_sentiment.get("reddit"):
        reddit_sentiment = social_sentiment["reddit"]["sentiment_indicator"]
        print(f"Reddit Sentiment: {reddit_sentiment}")
        
    if social_sentiment.get("twitter"):
        influence_score = social_sentiment["twitter"]["influence_score"]
        print(f"Twitter Influence Score: {influence_score:.1f}%")
        
