# twitter fetcher
import os
from dotenv import load_dotenv
import tweepy
from datetime import datetime, timedelta, timezone
import time
import re
load_dotenv()
  
  
# Twitter API credentials
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

def fetch_twitter_crypto_posts(queries=None, max_results=50, hours_back=24, analyze_credibility=True):
    """
    Fetch cryptocurrency tweets using Twitter API v2 with comprehensive credibility analysis
    OPTIMIZED FOR API QUOTA LIMITS
    
    Args:
        queries (list): List of search queries (default: crypto-related terms)
        max_results (int): Maximum results per query (10-100 for recent search)
        hours_back (int): How many hours back to search
        analyze_credibility (bool): Whether to perform credibility analysis
    """
    if not TWITTER_BEARER_TOKEN:
        print("Twitter Bearer Token not provided")
        return []
    
    # OPTIMIZATION 1: Reduce queries to stay under rate limits
    if queries is None:
        queries = [
            "bitcoin OR BTC cryptocurrency",
            "ethereum OR ETH crypto",
            "cryptocurrency market analysis"
        ]  # Reduced to 3 most effective queries
    
    try:
        # Initialize Twitter client with Bearer Token (API v2)
        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
        
        print(f"Fetching Twitter posts from {len(queries)} queries with credibility analysis")
        print(f"API Budget: Twitter allows 300 requests per 15 minutes")
        
        all_tweets = []
        
        # Calculate start time
        start_time = timezone.now() - timedelta(hours=hours_back)
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        api_calls_made = 0
        request_start_time = time.time()
        
        for query in queries:
            try:
                print(f"Searching Twitter for: {query}")
                
                # OPTIMIZATION 2: Get comprehensive data in single API call
                response = client.search_recent_tweets(
                    query=f"{query} -is:retweet lang:en",  # Exclude retweets, English only
                    tweet_fields=[
                        'created_at', 'author_id', 'public_metrics', 
                        'context_annotations', 'entities', 'geo', 
                        'in_reply_to_user_id', 'referenced_tweets',
                        'reply_settings', 'source', 'withheld',
                        'possibly_sensitive', 'edit_history_tweet_ids',
                        'conversation_id', 'lang'
                    ],
                    user_fields=[
                        'username', 'name', 'description', 'public_metrics',
                        'verified', 'created_at', 'location', 'profile_image_url',
                        'protected', 'url', 'verified_type', 'pinned_tweet_id'
                    ],
                    expansions=[
                        'author_id', 'referenced_tweets.id', 
                        'referenced_tweets.id.author_id', 'geo.place_id'
                    ],
                    start_time=start_time_str,
                    max_results=min(max_results, 100)  # API limit
                )
                
                api_calls_made += 1
                
                if not response.data:
                    continue
                
                # OPTIMIZATION 3: Create comprehensive lookup dictionaries
                users_dict = {}
                referenced_tweets_dict = {}
                
                if response.includes:
                    if 'users' in response.includes:
                        users_dict = {user.id: user for user in response.includes['users']}
                    
                    if 'tweets' in response.includes:
                        referenced_tweets_dict = {tweet.id: tweet for tweet in response.includes['tweets']}
                
                query_tweets = []
                for tweet in response.data:
                    try:
                        # Get comprehensive user info from includes
                        user_info = users_dict.get(tweet.author_id)
                        
                        # Calculate user credibility metrics
                        user_credibility = calculate_twitter_user_credibility(user_info) if user_info else create_basic_user_profile()
                        
                        # Analyze tweet content and engagement
                        content_analysis = analyze_twitter_content_quality(tweet, user_info)
                        engagement_metrics = calculate_twitter_engagement_metrics(tweet, user_info)
                        crypto_relevance = calculate_twitter_crypto_relevance(tweet, query)
                        
                        # Calculate overall credibility score
                        if analyze_credibility:
                            trust_score = calculate_twitter_trust_score(
                                tweet, user_info, user_credibility, content_analysis, 
                                engagement_metrics, crypto_relevance
                            )
                        else:
                            trust_score = {"final_trust_score": 0}
                        
                        tweet_data = {
                            # Core tweet data
                            "id": tweet.id,
                            "text": tweet.text,
                            "created_at": tweet.created_at,
                            "author_id": tweet.author_id,
                            "query": query,
                            "conversation_id": tweet.conversation_id,
                            "lang": tweet.lang,
                            "possibly_sensitive": getattr(tweet, 'possibly_sensitive', False),
                            
                            # Public metrics (engagement data)
                            "retweet_count": tweet.public_metrics.get('retweet_count', 0),
                            "like_count": tweet.public_metrics.get('like_count', 0),
                            "reply_count": tweet.public_metrics.get('reply_count', 0),
                            "quote_count": tweet.public_metrics.get('quote_count', 0),
                            
                            # User information (for credibility analysis)
                            "username": user_info.username if user_info else None,
                            "user_name": user_info.name if user_info else None,
                            "user_description": user_info.description if user_info else None,
                            "user_verified": user_info.verified if user_info else False,
                            "user_verified_type": getattr(user_info, 'verified_type', None) if user_info else None,
                            "user_followers": user_info.public_metrics.get('followers_count', 0) if user_info else 0,
                            "user_following": user_info.public_metrics.get('following_count', 0) if user_info else 0,
                            "user_tweet_count": user_info.public_metrics.get('tweet_count', 0) if user_info else 0,
                            "user_listed_count": user_info.public_metrics.get('listed_count', 0) if user_info else 0,
                            "user_location": user_info.location if user_info else None,
                            "user_profile_image": user_info.profile_image_url if user_info else None,
                            "user_created_at": user_info.created_at if user_info else None,
                            "user_protected": getattr(user_info, 'protected', False) if user_info else False,
                            "user_url": getattr(user_info, 'url', None) if user_info else None,
                            
                            # Tweet metadata for credibility
                            "context_annotations": tweet.context_annotations,
                            "entities": tweet.entities,
                            "in_reply_to_user_id": tweet.in_reply_to_user_id,
                            "referenced_tweets": tweet.referenced_tweets,
                            "reply_settings": tweet.reply_settings,
                            "source": tweet.source,
                            "geo": tweet.geo,
                            
                            # Computed engagement metrics
                            "total_engagement": (
                                tweet.public_metrics.get('retweet_count', 0) + 
                                tweet.public_metrics.get('like_count', 0) + 
                                tweet.public_metrics.get('reply_count', 0) + 
                                tweet.public_metrics.get('quote_count', 0)
                            ),
                            "engagement_rate": engagement_metrics.get('engagement_rate', 0),
                            "url": f"https://twitter.com/{user_info.username if user_info else 'user'}/status/{tweet.id}",
                            
                            # Credibility analysis results
                            "user_credibility": user_credibility,
                            "content_analysis": content_analysis,
                            "engagement_metrics": engagement_metrics,
                            "crypto_relevance": crypto_relevance,
                            "credibility_analysis": trust_score,
                            "analysis_timestamp": datetime.now().isoformat()
                        }
                        query_tweets.append(tweet_data)
                        
                    except Exception as e:
                        print(f"Error processing tweet {tweet.id}: {e}")
                        continue
                
                all_tweets.extend(query_tweets)
                print(f"Fetched {len(query_tweets)} tweets for query: {query}")
                print(f"API calls made: {api_calls_made}")
                
                # OPTIMIZATION 4: Smart rate limiting for Twitter API v2
                # Twitter API v2 allows 300 requests per 15 minutes = 1 request every 3 seconds
                if api_calls_made >= 10:  # Conservative limit for safety
                    elapsed_time = time.time() - request_start_time
                    if elapsed_time < 60:  # If we've made 10 calls in less than 60 seconds
                        wait_time = 60 - elapsed_time
                        print(f"⏳ Rate limiting: waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        api_calls_made = 0
                        request_start_time = time.time()
                else:
                    time.sleep(3)  # Standard 3-second delay between requests
                
            except tweepy.TooManyRequests as e:
                print(f"Rate limit exceeded for query '{query}': {e}")
                print("Waiting 15 minutes before continuing...")
                time.sleep(15 * 60)  # Wait 15 minutes
                continue
            except Exception as e:
                print(f"Error fetching tweets for query '{query}': {e}")
                continue
        
        # Remove duplicates and sort by credibility
        if analyze_credibility:
            # Remove duplicates by tweet ID
            unique_tweets = {}
            for tweet in all_tweets:
                tweet_id = tweet.get('id')
                if tweet_id not in unique_tweets or tweet.get('credibility_analysis', {}).get('final_trust_score', 0) > unique_tweets[tweet_id].get('credibility_analysis', {}).get('final_trust_score', 0):
                    unique_tweets[tweet_id] = tweet
            
            all_tweets = list(unique_tweets.values())
            
            # Sort by credibility score
            all_tweets.sort(key=lambda x: x.get('credibility_analysis', {}).get('final_trust_score', 0), reverse=True)
            
            # Analyze results
            analyze_twitter_results(all_tweets)
        
        print(f"Total Twitter posts fetched: {len(all_tweets)}")
        print(f"🔧 Total API calls made: {api_calls_made}")
        
        return all_tweets
        
    except Exception as e:
        print(f"Error initializing Twitter client: {e}")
        return []

def calculate_twitter_user_credibility(user_info):
    """Calculate user credibility score based on account metrics"""
    if not user_info:
        return create_basic_user_profile()
    
    try:
        # Account age calculation
        if user_info.created_at:
            account_created = user_info.created_at
            if isinstance(account_created, str):
                account_created = datetime.fromisoformat(account_created.replace('Z', '+00:00'))
            account_age_days = (timezone.now() - account_created).days
        else:
            account_age_days = 0
        
        # User metrics
        followers_count = user_info.public_metrics.get('followers_count', 0)
        following_count = user_info.public_metrics.get('following_count', 0)
        tweet_count = user_info.public_metrics.get('tweet_count', 0)
        listed_count = user_info.public_metrics.get('listed_count', 0)
        
        # Credibility indicators
        credibility_indicators = {
            "verified_account": user_info.verified,
            "blue_verified": getattr(user_info, 'verified_type', None) == 'blue',
            "business_verified": getattr(user_info, 'verified_type', None) == 'business',
            "government_verified": getattr(user_info, 'verified_type', None) == 'government',
            "established_account": account_age_days > 365,
            "mature_account": account_age_days > 180,
            "high_followers": followers_count > 10000,
            "very_high_followers": followers_count > 100000,
            "balanced_follow_ratio": (following_count / max(followers_count, 1)) < 2 if followers_count > 100 else True,
            "active_tweeter": tweet_count > 100,
            "prolific_tweeter": tweet_count > 1000,
            "listed_user": listed_count > 10,
            "has_description": bool(user_info.description),
            "has_url": bool(getattr(user_info, 'url', None)),
            "has_location": bool(user_info.location),
            "not_protected": not getattr(user_info, 'protected', False),
            "profile_complete": bool(user_info.description and user_info.location)
        }
        
        # Calculate credibility score
        trust_score = calculate_user_trust_score(
            credibility_indicators, account_age_days, followers_count, tweet_count
        )
        
        return {
            "exists": True,
            "username": user_info.username,
            "account_age_days": account_age_days,
            "followers_count": followers_count,
            "following_count": following_count,
            "tweet_count": tweet_count,
            "listed_count": listed_count,
            "verified": user_info.verified,
            "verified_type": getattr(user_info, 'verified_type', None),
            "credibility_indicators": credibility_indicators,
            "trust_score": trust_score,
            "account_tier": get_twitter_account_tier(account_age_days, followers_count, user_info.verified),
            "influence_level": calculate_influence_level(followers_count, listed_count, user_info.verified)
        }
        
    except Exception as e:
        print(f"Error calculating user credibility: {e}")
        return create_basic_user_profile()

def create_basic_user_profile():
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

def analyze_twitter_content_quality(tweet, user_info):
    """Analyze tweet content quality"""
    text = tweet.text
     
    quality_factors = {
        "meaningful_length": len(text) > 20,
        "not_too_long": len(text) < 250,
        "has_context_annotations": bool(tweet.context_annotations),
        "has_entities": bool(tweet.entities),
        "not_all_caps": not text.isupper(),
        "no_excessive_punctuation": not bool(re.search(r'[!?]{3,}', text)),
        "no_excessive_hashtags": len(re.findall(r'#\w+', text)) <= 3,
        "no_excessive_mentions": len(re.findall(r'@\w+', text)) <= 2,
        "not_possibly_sensitive": not getattr(tweet, 'possibly_sensitive', False),
        "is_original": not tweet.in_reply_to_user_id,
        "has_engagement": (tweet.public_metrics.get('like_count', 0) + 
                          tweet.public_metrics.get('retweet_count', 0)) > 0
    }
    
    # Spam/bot indicators
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
            "has_media": bool(tweet.entities and tweet.entities.get('media'))
        }
    }

def calculate_twitter_engagement_metrics(tweet, user_info):
    """Calculate engagement metrics for tweet"""
    like_count = tweet.public_metrics.get('like_count', 0)
    retweet_count = tweet.public_metrics.get('retweet_count', 0)
    reply_count = tweet.public_metrics.get('reply_count', 0)
    quote_count = tweet.public_metrics.get('quote_count', 0)
    
    total_engagement = like_count + retweet_count + reply_count + quote_count
    
    # Calculate rates relative to user's follower count
    follower_count = user_info.public_metrics.get('followers_count', 1) if user_info else 1
    engagement_rate = (total_engagement / max(follower_count, 1)) * 100
    
    # Tweet age for velocity calculation
    tweet_age_hours = (timezone.now() - tweet.created_at).total_seconds() / 3600
    engagement_velocity = total_engagement / max(tweet_age_hours, 0.1)
    
    return {
        "total_engagement": total_engagement,
        "engagement_rate": engagement_rate,
        "engagement_velocity": engagement_velocity,
        "like_to_follower_ratio": (like_count / max(follower_count, 1)) * 100,
        "retweet_to_like_ratio": (retweet_count / max(like_count, 1)),
        "reply_engagement": reply_count > 0,
        "viral_potential": calculate_tweet_viral_potential(tweet, user_info),
        "engagement_quality_score": min(engagement_rate * 2, 10)  # Scale to 0-10
    }

def calculate_twitter_crypto_relevance(tweet, search_query):
    """Calculate crypto relevance for tweet"""
    text = tweet.text.lower()
    
    # Crypto keywords with weights
    primary_crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "cryptocurrency", "crypto", "blockchain"]
    secondary_crypto_keywords = ["defi", "nft", "trading", "altcoin", "web3", "mining", "wallet"]
    context_keywords = ["bull", "bear", "hodl", "moon", "pump", "dump", "dyor"]
    
    relevance_score = 0
    keyword_matches = {}
    
    # Count primary keywords (high weight)
    primary_matches = []
    for keyword in primary_crypto_keywords:
        count = text.count(keyword)
        if count > 0:
            primary_matches.append({"keyword": keyword, "count": count})
            relevance_score += count * 3
    
    # Count secondary keywords (medium weight)
    secondary_matches = []
    for keyword in secondary_crypto_keywords:
        count = text.count(keyword)
        if count > 0:
            secondary_matches.append({"keyword": keyword, "count": count})
            relevance_score += count * 2
    
    # Count context keywords (low weight)
    context_matches = []
    for keyword in context_keywords:
        count = text.count(keyword)
        if count > 0:
            context_matches.append({"keyword": keyword, "count": count})
            relevance_score += count * 1
    
    # Context annotations bonus
    if tweet.context_annotations:
        for annotation in tweet.context_annotations:
            domain = annotation.get('domain', {}).get('name', '').lower()
            entity = annotation.get('entity', {}).get('name', '').lower()
            if any(crypto_term in f"{domain} {entity}" for crypto_term in ['crypto', 'bitcoin', 'blockchain']):
                relevance_score += 2
    
    # Query relevance bonus
    if search_query:
        query_words = search_query.lower().split()
        for word in query_words:
            if word in text:
                relevance_score += 1
    
    final_score = min(relevance_score / 10, 10.0)  # Scale to 0-10
    
    return {
        "crypto_relevance_score": final_score,
        "keyword_matches": {
            "primary": primary_matches,
            "secondary": secondary_matches,
            "context": context_matches
        },
        "context_crypto_detected": any(
            'crypto' in str(annotation).lower() for annotation in (tweet.context_annotations or [])
        ),
        "is_crypto_focused": final_score > 3.0,
        "relevance_tier": get_crypto_relevance_tier(final_score)
    }

def calculate_twitter_trust_score(tweet, user_info, user_credibility, content_analysis, engagement_metrics, crypto_relevance):
    """Calculate comprehensive Twitter credibility score"""
    
    # User authority (0-4 points)
    user_score = user_credibility.get('trust_score', 0) * 0.4
    
    # Content quality (0-3 points)
    content_score = content_analysis.get('quality_score', 0) * 0.3
    
    # Engagement quality (0-2 points)
    engagement_score = min(engagement_metrics.get('engagement_quality_score', 0) * 0.2, 2)
    
    # Crypto relevance (0-1 point)
    relevance_score = crypto_relevance.get('crypto_relevance_score', 0) * 0.1
    
    # Final score
    final_score = min(user_score + content_score + engagement_score + relevance_score, 10.0)
    
    # Credibility factors
    credibility_factors = {
        "verified_author": user_info.verified if user_info else False,
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

# Helper functions
def calculate_user_trust_score(indicators, account_age_days, followers_count, tweet_count):
    """Calculate user credibility score"""
    score = 0
    
    # Account age scoring
    if account_age_days > 1095:  # 3+ years
        score += 2
    elif account_age_days > 730:  # 2+ years
        score += 1.5
    elif account_age_days > 365:  # 1+ year
        score += 1
    elif account_age_days > 90:  # 3+ months
        score += 0.5
    
    # Follower scoring
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
    
    # Activity scoring
    if tweet_count > 10000:
        score += 1
    elif tweet_count > 1000:
        score += 0.5
    
    # Verification bonus
    if indicators.get('verified_account'):
        score += 2
    if indicators.get('blue_verified'):
        score += 1
    if indicators.get('business_verified') or indicators.get('government_verified'):
        score += 2
    
    # Quality indicators
    if indicators.get('balanced_follow_ratio'):
        score += 0.5
    if indicators.get('profile_complete'):
        score += 0.5
    if indicators.get('listed_user'):
        score += 0.5
    
    return min(score, 10)

def get_twitter_account_tier(account_age_days, followers_count, verified):
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

def calculate_influence_level(followers_count, listed_count, verified):
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

def calculate_tweet_viral_potential(tweet, user_info):
    """Calculate viral potential of tweet"""
    like_count = tweet.public_metrics.get('like_count', 0)
    retweet_count = tweet.public_metrics.get('retweet_count', 0)
    reply_count = tweet.public_metrics.get('reply_count', 0)
    
    # Tweet age for velocity
    tweet_age_hours = (timezone.now() - tweet.created_at).total_seconds() / 3600
    
    # Viral indicators
    viral_score = 0
    if retweet_count / max(tweet_age_hours, 1) > 10:  # 10+ retweets per hour
        viral_score += 3
    if like_count / max(tweet_age_hours, 1) > 50:  # 50+ likes per hour
        viral_score += 2
    if reply_count > 10:
        viral_score += 1
    if user_info and user_info.verified:
        viral_score += 2
    if tweet.context_annotations:  # Has topic context
        viral_score += 1
    
    return min(viral_score, 10)

def get_crypto_relevance_tier(score):
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

def get_twitter_credibility_tier(score):
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

def analyze_twitter_results(tweets):
    """Analyze Twitter results and provide summary"""
    if not tweets:
        return
    
    total_tweets = len(tweets)
    avg_credibility = sum(t.get('credibility_analysis', {}).get('final_trust_score', 0) for t in tweets) / total_tweets
    
    # Credibility distribution
    tiers = {"Highly Credible": 0, "Very Credible": 0, "Credible": 0, "Moderately Credible": 0, "Low Credibility": 0, "Very Low Credibility": 0}
    for tweet in tweets:
        tier = tweet.get('credibility_analysis', {}).get('credibility_tier', 'Very Low Credibility')
        if tier in tiers:
            tiers[tier] += 1
    
    # Account type distribution
    verified_count = sum(1 for t in tweets if t.get('user_verified'))
    high_followers = sum(1 for t in tweets if t.get('user_followers', 0) > 10000)
    
    print(f"📈 Twitter Analysis Summary:")
    print(f"   Total Tweets: {total_tweets}")
    print(f"   Average Credibility: {avg_credibility:.2f}/10")
    print(f"   Verified Users: {verified_count} ({verified_count/total_tweets*100:.1f}%)")
    print(f"   High-Follower Users: {high_followers} ({high_followers/total_tweets*100:.1f}%)")
    print(f"   Credibility Distribution: {dict(list(tiers.items())[:3])}")

# Optimized function for high-credibility tweets
def fetch_high_credibility_twitter_posts(max_tweets=50, trust_score_threshold=6.0, hours_back=12):
    """
    Fetch high-credibility Twitter posts with minimal API calls
    
    Args:
        max_tweets (int): Maximum tweets to return
        trust_score_threshold (float): Minimum credibility score
        hours_back (int): Hours back to search (reduced for better results)
    """
    print(f"Fetching high-credibility Twitter posts (threshold: {trust_score_threshold})")
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
    
    # Filter by credibility threshold
    high_credibility_tweets = [
        tweet for tweet in all_tweets 
        if tweet.get('credibility_analysis', {}).get('final_trust_score', 0) >= trust_score_threshold
    ]
    
    print(f"Found {len(high_credibility_tweets)} high-credibility tweets")
    
    return high_credibility_tweets[:max_tweets]

    
# --- Example Usage ---
if __name__ == "__main__":
    print("🐦 Twitter High-Credibility Crypto Posts Fetcher")
    print("=" * 55)
    
    # Test the high-credibility function
    high_credibility_posts = fetch_high_credibility_twitter_posts(
        max_tweets=10,
        trust_score_threshold=5.0,
        hours_back=74
    )
    
    if high_credibility_posts:
        print(f"\nFound {len(high_credibility_posts)} high-credibility tweets!")
        
        # Show top 3 tweets
        print("\n🏆 Top 3 High-Credibility Tweets:")
        for i, tweet in enumerate(high_credibility_posts[:3], 1):
            credibility = tweet.get('credibility_analysis', {})
            user_info = tweet.get('user_credibility', {})
            
            print(f"\n{i}. @{tweet.get('username', 'unknown')}")
            print(f"   Tweet: {tweet.get('text', '')[:100]}...")
            print(f"   Credibility: {credibility.get('final_trust_score', 0):.2f}/10")
            print(f"   Tier: {credibility.get('credibility_tier', 'Unknown')}")
            print(f"   Followers: {tweet.get('user_followers', 0):,}")
            print(f"   Engagement: {tweet.get('total_engagement', 0):,}")
            print(f"   Verified: {tweet.get('user_verified', False)}")
    else:
        print("No high-credibility tweets found")  