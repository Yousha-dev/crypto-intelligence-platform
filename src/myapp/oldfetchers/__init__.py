"""
Fetcher Module - Toggle between MOCK and REAL fetchers
Set USE_MOCK_FETCHERS=True in environment or settings to use mock data
"""
import os

# Read from environment variable (default to True for mock mode)
USE_MOCK_FETCHERS = os.getenv("USE_MOCK_FETCHERS", "true").lower() == "true"

if USE_MOCK_FETCHERS:
    print("[FETCHERS] Using MOCK fetchers (no API calls)")
    
    # News fetchers
    from .mock_news_data import (
        fetch_cryptopanic_news_enhanced,
        fetch_cryptocompare_news_enhanced,
        fetch_newsapi_articles_enhanced,
        fetch_messari_news_enhanced,
        scrape_coindesk_enhanced,
        fetch_all_news_enhanced,
        fetch_high_credibility_news
    )
    
    # Social fetchers - Reddit
    from .mock_reddit import (
        fetch_reddit_crypto_posts_with_credibility,
        fetch_high_credibility_reddit_posts
    )
    
    # Social fetchers - Twitter
    from .mock_twitter import (
        fetch_twitter_crypto_posts,
        fetch_high_credibility_twitter_posts
    )
    
    # Social fetchers - YouTube
    from .mock_youtube import (
        search_latest_high_credibility_crypto_videos_optimized,
        fetch_high_credibility_youtube_videos
    )
    
else:
    print("🌐 [FETCHERS] Using REAL fetchers (API calls enabled)")
    
    # News fetchers
    from .news_data import (
        fetch_cryptopanic_news_enhanced,
        fetch_cryptocompare_news_enhanced,
        fetch_newsapi_articles_enhanced,
        fetch_messari_news_enhanced,
        scrape_coindesk_enhanced,
        fetch_all_news_enhanced,
        fetch_high_credibility_news
    )
    
    # Social fetchers - Reddit
    from .reddit import (
        fetch_reddit_crypto_posts_with_credibility,
        fetch_high_credibility_reddit_posts
    )
    
    # Social fetchers - Twitter
    from .twitter import (
        fetch_twitter_crypto_posts,
        fetch_high_credibility_twitter_posts
    )
    
    # Social fetchers - YouTube
    from .youtube import (
        search_latest_high_credibility_crypto_videos_optimized,
        fetch_high_credibility_youtube_videos
    )


# Export all functions
__all__ = [
    # News
    'fetch_cryptopanic_news_enhanced',
    'fetch_cryptocompare_news_enhanced',
    'fetch_newsapi_articles_enhanced',
    'fetch_messari_news_enhanced',
    'scrape_coindesk_enhanced',
    'fetch_all_news_enhanced',
    'fetch_high_credibility_news',
    # Reddit
    'fetch_reddit_crypto_posts_with_credibility',
    'fetch_high_credibility_reddit_posts',
    # Twitter
    'fetch_twitter_crypto_posts',
    'fetch_high_credibility_twitter_posts',
    # YouTube
    'search_latest_high_credibility_crypto_videos_optimized',
    'fetch_high_credibility_youtube_videos',
    # Config
    'USE_MOCK_FETCHERS'
]