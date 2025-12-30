"""
Mock Twitter Fetcher - Uses local JSON files instead of API calls
EXACT match to real twitter.py output structure
"""
import json
import time
import re
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


# ============================================================================
# ANALYSIS FUNCTIONS (Exact copies from real twitter.py)
# ============================================================================

def calculate_twitter_user_credibility(user_info: dict) -> dict:
    """Calculate user credibility score based on account metrics - EXACT match to real"""
    if not user_info:
        return create_basic_user_profile()
    
    try:
        # Account age calculation
        created_at = user_info.get('created_at', '')
        if created_at:
            try:
                if isinstance(created_at, str):
                    account_created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    account_created = created_at
                account_age_days = (timezone.now() - account_created).days
            except:
                account_age_days = 365
        else:
            account_age_days = 365
        
        # User metrics
        public_metrics = user_info.get('public_metrics', {})
        followers_count = public_metrics.get('followers_count', 0)
        following_count = public_metrics.get('following_count', 0)
        tweet_count = public_metrics.get('tweet_count', 0)
        listed_count = public_metrics.get('listed_count', 0)
        
        verified = user_info.get('verified', False)
        verified_type = user_info.get('verified_type')
        
        # Credibility indicators
        credibility_indicators = {
            "verified_account": verified,
            "blue_verified": verified_type == 'blue',
            "business_verified": verified_type == 'business',
            "government_verified": verified_type == 'government',
            "established_account": account_age_days > 365,
            "mature_account": account_age_days > 180,
            "high_followers": followers_count > 10000,
            "very_high_followers": followers_count > 100000,
            "balanced_follow_ratio": (following_count / max(followers_count, 1)) < 2 if followers_count > 100 else True,
            "active_tweeter": tweet_count > 100,
            "prolific_tweeter": tweet_count > 1000,
            "listed_user": listed_count > 10,
            "has_description": bool(user_info.get('description')),
            "has_url": bool(user_info.get('url')),
            "has_location": bool(user_info.get('location')),
            "not_protected": not user_info.get('protected', False),
            "profile_complete": bool(user_info.get('description') and user_info.get('location'))
        }
        
        # Calculate credibility score
        trust_score = calculate_user_trust_score(
            credibility_indicators, account_age_days, followers_count, tweet_count
        )
        
        return {
            "exists": True,
            "username": user_info.get('username', 'unknown'),
            "account_age_days": account_age_days,
            "followers_count": followers_count,
            "following_count": following_count,
            "tweet_count": tweet_count,
            "listed_count": listed_count,
            "verified": verified,
            "verified_type": verified_type,
            "credibility_indicators": credibility_indicators,
            "trust_score": trust_score,
            "account_tier": get_twitter_account_tier(account_age_days, followers_count, verified),
            "influence_level": calculate_influence_level(followers_count, listed_count, verified)
        }
        
    except Exception as e:
        print(f"Error calculating user credibility: {e}")
        return create_basic_user_profile()


def create_basic_user_profile() -> dict:
    """Create basic user profile for missing/deleted accounts"""
    return {
        "exists": False,
        "username": "[deleted]",
        "trust_score": 0,
        "account_tier": "Unknown",
        "influence_level": "None",
        "credibility_indicators": {},
        "reason": "Account deleted or unavailable"
    }


def calculate_user_trust_score(indicators: dict, account_age_days: int, 
                                     followers_count: int, tweet_count: int) -> float:
    """Calculate user credibility score"""
    score = 0
    
    if account_age_days > 1095:
        score += 2
    elif account_age_days > 730:
        score += 1.5
    elif account_age_days > 365:
        score += 1
    elif account_age_days > 90:
        score += 0.5
    
    if followers_count > 1000000:
        score += 3
    elif followers_count > 100000:
        score += 2.5
    elif followers_count > 10000:
        score += 2
    elif followers_count > 1000:
        score += 1
    elif followers_count > 100:
        score += 0.5
    
    if tweet_count > 10000:
        score += 1
    elif tweet_count > 1000:
        score += 0.5
    
    if indicators.get('verified_account'):
        score += 2
    if indicators.get('blue_verified'):
        score += 1
    if indicators.get('business_verified') or indicators.get('government_verified'):
        score += 2
    
    if indicators.get('balanced_follow_ratio'):
        score += 0.5
    if indicators.get('profile_complete'):
        score += 0.5
    if indicators.get('listed_user'):
        score += 0.5
    
    return min(score, 10)


def get_twitter_account_tier(account_age_days: int, followers_count: int, verified: bool) -> str:
    """Determine Twitter account tier"""
    if verified and followers_count > 100000:
        return "Verified Influencer"
    elif followers_count > 1000000:
        return "Mega Influencer"
    elif followers_count > 100000:
        return "Major Influencer"
    elif followers_count > 10000:
        return "Micro Influencer"
    elif account_age_days > 730 and followers_count > 1000:
        return "Established User"
    elif account_age_days > 365:
        return "Regular User"
    elif account_age_days > 90:
        return "New User"
    else:
        return "Very New User"


def calculate_influence_level(followers_count: int, listed_count: int, verified: bool) -> str:
    """Calculate influence level"""
    if verified and followers_count > 500000:
        return "Very High"
    elif followers_count > 100000 or (verified and followers_count > 10000):
        return "High"
    elif followers_count > 10000 or listed_count > 100:
        return "Medium"
    elif followers_count > 1000:
        return "Low"
    else:
        return "None"


def analyze_twitter_content_quality(tweet_data: dict, user_info: dict) -> dict:
    """Analyze tweet content quality - EXACT match to real"""
    # Mock uses dict access (correct for JSON data)
    text = tweet_data.get('text', '')
    public_metrics = tweet_data.get('public_metrics', {})
    entities = tweet_data.get('entities', {})
    context_annotations = tweet_data.get('context_annotations', [])
    
    quality_factors = {
        "meaningful_length": len(text) > 20,
        "not_too_long": len(text) < 250,
        "has_context_annotations": bool(context_annotations),
        "has_entities": bool(entities),
        "not_all_caps": not text.isupper(),
        "no_excessive_punctuation": not bool(re.search(r'[!?]{3,}', text)),
        "no_excessive_hashtags": len(re.findall(r'#\w+', text)) <= 3,
        "no_excessive_mentions": len(re.findall(r'@\w+', text)) <= 2,
        "not_possibly_sensitive": not tweet_data.get('possibly_sensitive', False),
        "is_original": not tweet_data.get('in_reply_to_user_id'),
        "has_engagement": (public_metrics.get('like_count', 0) + 
                          public_metrics.get('retweet_count', 0)) > 0
    }
    
    spam_indicators = {
        "repeated_characters": bool(re.search(r'(.)\1{4,}', text)),
        "excessive_emojis": len(re.findall(r'[😀-🿿]', text)) > 5,
        "suspicious_urls": len(re.findall(r'bit\.ly|tinyurl|t\.co', text)) > 1,
        "promotional_keywords": bool(re.search(r'\b(buy now|click here|limited time|act fast)\b', text.lower())),
        "excessive_caps": len(re.findall(r'[A-Z]{5,}', text)) > 2
    }
    
    quality_score = sum(quality_factors.values()) / len(quality_factors) * 10
    spam_score = sum(spam_indicators.values()) / len(spam_indicators) * 10
    
    return {
        "quality_factors": quality_factors,
        "quality_score": max(0, quality_score - spam_score),
        "spam_indicators": spam_indicators,
        "spam_score": spam_score,
        "text_analysis": {
            "length": len(text),
            "word_count": len(text.split()),
            "hashtag_count": len(re.findall(r'#\w+', text)),
            "mention_count": len(re.findall(r'@\w+', text)),
            "url_count": len(re.findall(r'http[s]?://', text)),
            "has_media": bool(entities.get('media'))
        }
    }


def calculate_twitter_engagement_metrics(tweet_data: dict, user_info: dict) -> dict:
    """Calculate engagement metrics for tweet - EXACT match to real"""
    public_metrics = tweet_data.get('public_metrics', {})
    like_count = public_metrics.get('like_count', 0)
    retweet_count = public_metrics.get('retweet_count', 0)
    reply_count = public_metrics.get('reply_count', 0)
    quote_count = public_metrics.get('quote_count', 0)
    
    total_engagement = like_count + retweet_count + reply_count + quote_count
    
    user_public_metrics = user_info.get('public_metrics', {}) if user_info else {}
    follower_count = user_public_metrics.get('followers_count', 1)
    engagement_rate = (total_engagement / max(follower_count, 1)) * 100
    
    # Tweet age for velocity calculation
    created_at = tweet_data.get('created_at', '')
    try:
        if isinstance(created_at, str):
            tweet_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        else:
            tweet_time = created_at
        tweet_age_hours = (timezone.now() - tweet_time).total_seconds() / 3600
    except:
        tweet_age_hours = 1
    
    engagement_velocity = total_engagement / max(tweet_age_hours, 0.1)
    
    return {
        "total_engagement": total_engagement,
        "engagement_rate": engagement_rate,
        "engagement_velocity": engagement_velocity,
        "like_to_follower_ratio": (like_count / max(follower_count, 1)) * 100,
        "retweet_to_like_ratio": (retweet_count / max(like_count, 1)),
        "reply_engagement": reply_count > 0,
        "viral_potential": calculate_tweet_viral_potential(tweet_data, user_info),
        "engagement_quality_score": min(engagement_rate * 2, 10)
    }


def calculate_tweet_viral_potential(tweet_data: dict, user_info: dict) -> float:
    """Calculate viral potential of tweet"""
    public_metrics = tweet_data.get('public_metrics', {})
    like_count = public_metrics.get('like_count', 0)
    retweet_count = public_metrics.get('retweet_count', 0)
    reply_count = public_metrics.get('reply_count', 0)
    
    created_at = tweet_data.get('created_at', '')
    try:
        if isinstance(created_at, str):
            tweet_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        else:
            tweet_time = created_at
        tweet_age_hours = (timezone.now() - tweet_time).total_seconds() / 3600
    except:
        tweet_age_hours = 1
    
    viral_score = 0
    if retweet_count / max(tweet_age_hours, 1) > 10:
        viral_score += 3
    if like_count / max(tweet_age_hours, 1) > 50:
        viral_score += 2
    if reply_count > 10:
        viral_score += 1
    if user_info and user_info.get('verified'):
        viral_score += 2
    if tweet_data.get('context_annotations'):
        viral_score += 1
    
    return min(viral_score, 10)


def calculate_twitter_crypto_relevance(tweet_data: dict, search_query: str) -> dict:
    """Calculate crypto relevance for tweet - EXACT match to real"""
    text = tweet_data.get('text', '').lower()
    
    primary_crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "cryptocurrency", "crypto", "blockchain"]
    secondary_crypto_keywords = ["defi", "nft", "trading", "altcoin", "web3", "mining", "wallet"]
    context_keywords = ["bull", "bear", "hodl", "moon", "pump", "dump", "dyor"]
    
    relevance_score = 0
    keyword_matches = {}
    
    primary_matches = []
    for keyword in primary_crypto_keywords:
        count = text.count(keyword)
        if count > 0:
            primary_matches.append({"keyword": keyword, "count": count})
            relevance_score += count * 3
    
    secondary_matches = []
    for keyword in secondary_crypto_keywords:
        count = text.count(keyword)
        if count > 0:
            secondary_matches.append({"keyword": keyword, "count": count})
            relevance_score += count * 2
    
    context_matches = []
    for keyword in context_keywords:
        count = text.count(keyword)
        if count > 0:
            context_matches.append({"keyword": keyword, "count": count})
            relevance_score += count * 1
    
    # Context annotations bonus
    context_annotations = tweet_data.get('context_annotations', [])
    if context_annotations:
        for annotation in context_annotations:
            domain = annotation.get('domain', {}).get('name', '').lower()
            entity = annotation.get('entity', {}).get('name', '').lower()
            if any(crypto_term in f"{domain} {entity}" for crypto_term in ['crypto', 'bitcoin', 'blockchain']):
                relevance_score += 2
    
    if search_query:
        query_words = search_query.lower().split()
        for word in query_words:
            if word in text:
                relevance_score += 1
    
    final_score = min(relevance_score / 10, 10.0)
    
    return {
        "crypto_relevance_score": final_score,
        "keyword_matches": {
            "primary": primary_matches,
            "secondary": secondary_matches,
            "context": context_matches
        },
        "context_crypto_detected": any(
            'crypto' in str(annotation).lower() for annotation in (context_annotations or [])
        ),
        "is_crypto_focused": final_score > 3.0,
        "relevance_tier": get_crypto_relevance_tier(final_score)
    }


def get_crypto_relevance_tier(score: float) -> str:
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


def calculate_twitter_trust_score(tweet_data: dict, user_info: dict, user_credibility: dict,
                                        content_analysis: dict, engagement_metrics: dict,
                                        crypto_relevance: dict) -> dict:
    """Calculate comprehensive Twitter credibility score - EXACT match to real"""
    
    user_score = user_credibility.get('trust_score', 0) * 0.4
    content_score = content_analysis.get('quality_score', 0) * 0.3
    engagement_score = min(engagement_metrics.get('engagement_quality_score', 0) * 0.2, 2)
    relevance_score = crypto_relevance.get('crypto_relevance_score', 0) * 0.1
    
    final_score = min(user_score + content_score + engagement_score + relevance_score, 10.0)
    
    credibility_factors = {
        "verified_author": user_info.get('verified', False) if user_info else False,
        "established_account": user_credibility.get('credibility_indicators', {}).get('established_account', False),
        "high_quality_content": content_analysis.get('quality_score', 0) > 7,
        "good_engagement": engagement_metrics.get('engagement_rate', 0) > 1,
        "crypto_relevant": crypto_relevance.get('is_crypto_focused', False),
        "not_spam": content_analysis.get('spam_score', 10) < 3,
        "authentic_engagement": engagement_metrics.get('reply_engagement', False),
        "influencer_level": user_credibility.get('influence_level', 'None') in ['High', 'Very High']
    }
    
    return {
        "final_trust_score": final_score,
        "score_breakdown": {
            "user_authority": user_score,
            "content_quality": content_score,
            "engagement_quality": engagement_score,
            "crypto_relevance": relevance_score
        },
        "credibility_factors": credibility_factors,
        "credibility_tier": get_twitter_credibility_tier(final_score),
        "analysis_timestamp": datetime.now().isoformat()
    }


def get_twitter_credibility_tier(score: float) -> str:
    """Get Twitter credibility tier"""
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


# ============================================================================
# MOCK FETCHER FUNCTIONS (EXACT same signatures as real fetchers)
# ============================================================================

def fetch_twitter_crypto_posts(queries=None, max_results=50, hours_back=24, analyze_credibility=True):
    """
    MOCK: Fetch cryptocurrency tweets using Twitter API v2 with comprehensive credibility analysis
    EXACT same signature and output structure as real fetcher
    """
    if queries is None:
        queries = [
            "bitcoin OR BTC cryptocurrency",
            "ethereum OR ETH crypto",
            "cryptocurrency market analysis"
        ]
    
    print(f"[MOCK] Fetching Twitter posts from {len(queries)} queries with credibility analysis")
    print(f"Loading from: {MOCK_DATA_DIR / 'twitter.json'}")
    print(f"API Budget: Twitter allows 300 requests per 15 minutes")
    
    time.sleep(0.3)
    
    tweets = load_mock_data("twitter.json")
    
    if not tweets:
        print("️ No mock Twitter data found")
        return []
    
    all_tweets = []
    
    for query in queries:
        query_tweets = []
        
        for tweet in tweets[:max_results // len(queries)]:
            try:
                user_info = tweet.get('user_info', {})
                public_metrics = tweet.get('public_metrics', {})
                
                # Calculate user credibility
                user_credibility = calculate_twitter_user_credibility(user_info)
                
                # Analyze content and engagement
                content_analysis = analyze_twitter_content_quality(tweet, user_info)
                engagement_metrics = calculate_twitter_engagement_metrics(tweet, user_info)
                crypto_relevance = calculate_twitter_crypto_relevance(tweet, query)
                
                # Calculate credibility score
                if analyze_credibility:
                    trust_score = calculate_twitter_trust_score(
                        tweet, user_info, user_credibility, content_analysis,
                        engagement_metrics, crypto_relevance
                    )
                else:
                    trust_score = {"final_trust_score": 0}
                
                # Build tweet data matching EXACT real structure
                tweet_data = {
                    # Core tweet data
                    "id": tweet.get('id'),
                    "text": tweet.get('text', ''),
                    "created_at": tweet.get('created_at', timezone.now().isoformat()),
                    "author_id": tweet.get('author_id', user_info.get('id')),
                    "query": query,
                    "conversation_id": tweet.get('conversation_id'),
                    "lang": tweet.get('lang', 'en'),
                    "possibly_sensitive": tweet.get('possibly_sensitive', False),
                    
                    # Public metrics
                    "retweet_count": public_metrics.get('retweet_count', 0),
                    "like_count": public_metrics.get('like_count', 0),
                    "reply_count": public_metrics.get('reply_count', 0),
                    "quote_count": public_metrics.get('quote_count', 0),
                    
                    # User information
                    "username": user_info.get('username'),
                    "user_name": user_info.get('name'),
                    "user_description": user_info.get('description'),
                    "user_verified": user_info.get('verified', False),
                    "user_verified_type": user_info.get('verified_type'),
                    "user_followers": user_info.get('public_metrics', {}).get('followers_count', 0),
                    "user_following": user_info.get('public_metrics', {}).get('following_count', 0),
                    "user_tweet_count": user_info.get('public_metrics', {}).get('tweet_count', 0),
                    "user_listed_count": user_info.get('public_metrics', {}).get('listed_count', 0),
                    "user_location": user_info.get('location'),
                    "user_profile_image": user_info.get('profile_image_url'),
                    "user_created_at": user_info.get('created_at'),
                    "user_protected": user_info.get('protected', False),
                    "user_url": user_info.get('url'),
                    
                    # Tweet metadata
                    "context_annotations": tweet.get('context_annotations', []),
                    "entities": tweet.get('entities', {}),
                    "in_reply_to_user_id": tweet.get('in_reply_to_user_id'),
                    "referenced_tweets": tweet.get('referenced_tweets', []),
                    "reply_settings": tweet.get('reply_settings'),
                    "source": tweet.get('source'),
                    "geo": tweet.get('geo'),
                    
                    # Computed metrics
                    "total_engagement": (
                        public_metrics.get('retweet_count', 0) +
                        public_metrics.get('like_count', 0) +
                        public_metrics.get('reply_count', 0) +
                        public_metrics.get('quote_count', 0)
                    ),
                    "engagement_rate": engagement_metrics.get('engagement_rate', 0),
                    "url": f"https://twitter.com/{user_info.get('username', 'user')}/status/{tweet.get('id', '')}",
                    
                    # Credibility analysis
                    "user_credibility": user_credibility,
                    "content_analysis": content_analysis,
                    "engagement_metrics": engagement_metrics,
                    "crypto_relevance": crypto_relevance,
                    "credibility_analysis": trust_score,
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
                query_tweets.append(tweet_data)
                
            except Exception as e:
                print(f"Error processing tweet {tweet.get('id')}: {e}")
                continue
        
        all_tweets.extend(query_tweets)
        print(f"[MOCK] Fetched {len(query_tweets)} tweets for query: {query}")
    
    # Remove duplicates and sort by credibility
    if analyze_credibility:
        unique_tweets = {}
        for tweet in all_tweets:
            tweet_id = tweet.get('id')
            if tweet_id not in unique_tweets or tweet.get('credibility_analysis', {}).get('final_trust_score', 0) > unique_tweets[tweet_id].get('credibility_analysis', {}).get('final_trust_score', 0):
                unique_tweets[tweet_id] = tweet
        
        all_tweets = list(unique_tweets.values())
        all_tweets.sort(key=lambda x: x.get('credibility_analysis', {}).get('final_trust_score', 0), reverse=True)
        
        analyze_twitter_results(all_tweets)
    
    print(f"[MOCK] Total Twitter posts fetched: {len(all_tweets)}")
    
    return all_tweets


def fetch_high_credibility_twitter_posts(max_tweets=50, trust_score_threshold=6.0, hours_back=12):
    """
    MOCK: Fetch high-credibility Twitter posts with minimal API calls
    EXACT same signature as real fetcher
    """
    print(f"[MOCK] Fetching high-credibility Twitter posts (threshold: {trust_score_threshold})")
    print(f"API Budget: Conservative approach with 3 queries max")
    
    all_tweets = fetch_twitter_crypto_posts(
        queries=[
            "bitcoin OR BTC price analysis",
            "ethereum OR ETH crypto news",
            "cryptocurrency market update"
        ],
        max_results=max_tweets // 3,
        hours_back=hours_back,
        analyze_credibility=True
    )
    
    high_credibility_tweets = [
        tweet for tweet in all_tweets
        if tweet.get('credibility_analysis', {}).get('final_trust_score', 0) >= trust_score_threshold
    ]
    
    print(f"[MOCK] Found {len(high_credibility_tweets)} high-credibility tweets")
    
    return high_credibility_tweets[:max_tweets]


def analyze_twitter_results(tweets: list):
    """Analyze Twitter results"""
    if not tweets:
        return
    
    total = len(tweets)
    avg_credibility = sum(
        t.get('credibility_analysis', {}).get('final_trust_score', 0) 
        for t in tweets
    ) / total
    verified_count = sum(1 for t in tweets if t.get('user_credibility', {}).get('verified', False))
    
    print(f"📈 [MOCK] Twitter Analysis:")
    print(f"   Total Tweets: {total}")
    print(f"   Average Credibility: {avg_credibility:.2f}/10")
    print(f"   Verified Users: {verified_count}")


if __name__ == "__main__":
    print("🧪 Testing Mock Twitter Fetcher")
    print("=" * 55)
    
    tweets = fetch_high_credibility_twitter_posts(max_tweets=10, trust_score_threshold=5.0)
    
    if tweets:
        print(f"\nFound {len(tweets)} high-credibility tweets!")
        
        for i, tweet in enumerate(tweets[:3], 1):
            credibility = tweet.get('credibility_analysis', {})
            
            print(f"\n{i}. @{tweet.get('username', 'unknown')}")
            print(f"   Tweet: {tweet.get('text', '')[:100]}...")
            print(f"   Credibility: {credibility.get('final_trust_score', 0):.2f}/10")
            print(f"   Tier: {credibility.get('credibility_tier', 'Unknown')}")
            print(f"   Followers: {tweet.get('user_followers', 0):,}")
            print(f"   Engagement: {tweet.get('total_engagement', 0):,}")
            print(f"   Verified: {tweet.get('user_verified', False)}")
    else:
        print("No high-credibility tweets found")