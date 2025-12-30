"""
Fetcher Module - RAW DATA ONLY
All credibility/sentiment analysis happens in services
Toggle between MOCK and REAL fetchers with USE_MOCK_FETCHERS env var
"""
import os
  
USE_MOCK_FETCHERS = os.getenv("USE_MOCK_FETCHERS", "false").lower() == "true"

if USE_MOCK_FETCHERS:
    print("[FETCHERS] Using MOCK fetchers (no API calls)")
    
    from .mock_news_data import (
        fetch_cryptopanic_news,
        fetch_cryptocompare_news,
        fetch_newsapi_articles,
        fetch_messari_news,
        fetch_coindesk_news,
        fetch_all_news
    )
    
    from .mock_reddit import (
        fetch_reddit_posts
    )
    
    from .mock_twitter import (
        fetch_twitter_posts
    )
    
    from .mock_youtube import (
        fetch_youtube_videos,
        get_video_transcript
    )
    
else:
    print("🌐 [FETCHERS] Using REAL fetchers (API calls enabled)")
    
    from .news_data import (
        fetch_cryptopanic_news,
        fetch_cryptocompare_news,
        fetch_newsapi_articles,
        fetch_messari_news,
        fetch_coindesk_news
    )
    
    from .reddit import (
        fetch_reddit_posts
    )
    
    from .twitter import (
        fetch_twitter_posts
    )
    
    from .youtube import (
        fetch_youtube_videos,
        get_video_transcript
    )


__all__ = [
    # New clean functions (raw data only)
    'fetch_cryptopanic_news',
    'fetch_cryptocompare_news',
    'fetch_newsapi_articles',
    'fetch_messari_news',
    'fetch_coindesk_news',
    'fetch_all_news',
    'fetch_reddit_posts',
    'fetch_twitter_posts',
    'fetch_youtube_videos',
    'get_video_transcript',
    
    # Config
    'USE_MOCK_FETCHERS'
]