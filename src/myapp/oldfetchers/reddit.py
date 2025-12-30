# reddit fetcher
import os
from dotenv import load_dotenv
import praw
from datetime import datetime, timedelta, timezone
import time
import re
from datetime import timezone as timezoneDt
from collections import Counter

  
load_dotenv()

# Reddit API credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "CryptoNewsBot/1.0")
  

# Optimized Reddit function with minimal API calls
def fetch_reddit_crypto_posts_with_credibility(subreddits=None, limit=50, time_filter="day", sort_type="hot", analyze_credibility=True):
    """
    Fetch cryptocurrency posts from Reddit with comprehensive credibility analysis
    OPTIMIZED FOR LOW API QUOTA (10 QPM)
    
    Args:
        subreddits (list): List of subreddit names
        limit (int): Number of posts to fetch per subreddit
        time_filter (str): Time filter ("hour", "day", "week", "month", "year", "all")
        sort_type (str): Sort type ("hot", "new", "rising", "top")
        analyze_credibility (bool): Whether to perform credibility analysis
    """
    if subreddits is None:
        subreddits = [
            "CryptoCurrency",
            "Bitcoin", 
            "ethereum",
            "CryptoMarkets",
            "altcoin"
        ]  # Reduced to 5 most important subreddits
    
    try:
        # Initialize Reddit instance
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            check_for_async=False
        )
        
        print(f"Fetching Reddit posts from {len(subreddits)} subreddits with optimized credibility analysis")
        all_posts = []
        
        # Cache for author and subreddit info to avoid duplicate API calls
        author_cache = {}
        subreddit_cache = {}
        
        api_calls_made = 0
        start_time = time.time()
        
        for subreddit_name in subreddits:
            try:
                print(f"Fetching from r/{subreddit_name}...")
                
                # Single API call to get subreddit and posts
                subreddit = reddit.subreddit(subreddit_name)
                
                # Get subreddit info (1 API call per subreddit)
                if subreddit_name not in subreddit_cache:
                    subreddit_info = get_subreddit_info_efficiently(subreddit)
                    subreddit_cache[subreddit_name] = subreddit_info
                    api_calls_made += 1
                else:
                    subreddit_info = subreddit_cache[subreddit_name]
                
                # Get posts (1 API call per subreddit)
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
                
                api_calls_made += 1
                
                subreddit_posts = []
                processed_authors = set()  # Track authors we've already analyzed
                
                for post in posts:
                    try:
                        # Extract all available post data in one pass
                        post_data = extract_comprehensive_post_data_efficient(post)
                        
                        if analyze_credibility:
                            author_name = str(post.author) if post.author else "[deleted]"
                            
                            # Only fetch author info if not cached and author exists
                            if author_name not in author_cache and author_name != "[deleted]":
                                # Limit author lookups to avoid quota issues
                                if author_name not in processed_authors and api_calls_made < 8:  # Leave buffer for other calls
                                    author_info = get_author_info_efficiently(reddit, post.author)
                                    author_cache[author_name] = author_info
                                    processed_authors.add(author_name)
                                    api_calls_made += 1
                                    
                                    # Rate limiting
                                    time.sleep(6)  # 10 QPM = 6 seconds between calls
                                else:
                                    # Use basic author info without API call
                                    author_cache[author_name] = create_basic_author_profile(post.author)
                            
                            author_info = author_cache.get(author_name, create_basic_author_profile(post.author))
                            
                            # Perform all other analysis without additional API calls
                            content_analysis = analyze_reddit_post_content(post_data)
                            engagement_metrics = calculate_reddit_engagement_metrics(post_data)
                            crypto_relevance = calculate_reddit_crypto_relevance(post_data)
                            
                            # Calculate credibility score
                            trust_score = calculate_reddit_trust_score(
                                post_data, author_info, subreddit_info, 
                                content_analysis, engagement_metrics, crypto_relevance
                            )
                            
                            # Add all analysis to post data
                            post_data.update({
                                "author_info": author_info,
                                "subreddit_info": subreddit_info,
                                "content_analysis": content_analysis,
                                "engagement_metrics": engagement_metrics,
                                "crypto_relevance": crypto_relevance,
                                "credibility_analysis": trust_score
                            })
                        
                        subreddit_posts.append(post_data)
                        
                    except Exception as e:
                        print(f"Error processing post {post.id}: {e}")
                        continue
                
                all_posts.extend(subreddit_posts)
                print(f"Fetched {len(subreddit_posts)} posts from r/{subreddit_name}")
                print(f"API calls made: {api_calls_made}")
                
                # Rate limiting between subreddits
                if api_calls_made >= 8:  # Stay well under 10 QPM limit
                    wait_time = 60 - (time.time() - start_time)
                    if wait_time > 0:
                        print(f"⏳ Waiting {wait_time:.1f}s for rate limit...")
                        time.sleep(wait_time)
                        api_calls_made = 0
                        start_time = time.time()
                
            except Exception as e:
                print(f"Error fetching from r/{subreddit_name}: {e}")
                continue
        
        print(f"Total Reddit posts fetched: {len(all_posts)}")
        print(f"🔧 Total API calls made: {api_calls_made}")
        
        # Sort by credibility if analysis was performed
        if analyze_credibility and all_posts:
            all_posts.sort(key=lambda x: x.get('credibility_analysis', {}).get('final_trust_score', 0), reverse=True)
            
            # Show summary
            avg_credibility = sum(p.get('credibility_analysis', {}).get('final_trust_score', 0) for p in all_posts) / len(all_posts)
            print(f"📈 Average credibility score: {avg_credibility:.2f}/10")
        
        return all_posts
        
    except Exception as e:
        print(f"Error initializing Reddit client: {e}")
        return []

def get_subreddit_info_efficiently(subreddit):
    """Get subreddit info in single API call"""
    try:
        # All this data comes from the subreddit object without additional API calls
        return {
            "name": subreddit.display_name,
            "subscribers": subreddit.subscribers,
            "created_utc": subreddit.created_utc,
            "public_description": getattr(subreddit, 'public_description', ''),
            "over18": getattr(subreddit, 'over18', False),
            "subreddit_type": getattr(subreddit, 'subreddit_type', 'public'),
            "active_user_count": getattr(subreddit, 'active_user_count', 0),
            "accounts_active": getattr(subreddit, 'accounts_active', 0),
            "lang": getattr(subreddit, 'lang', 'en'),
            "credibility_factors": {
                "large_community": subreddit.subscribers > 100000,
                "established": (datetime.now().timestamp() - subreddit.created_utc) > (365 * 24 * 3600),
                "active_community": getattr(subreddit, 'accounts_active', 0) > 100,
                "public_subreddit": getattr(subreddit, 'subreddit_type', 'public') == 'public',
                "crypto_focused": subreddit.display_name.lower() in ['cryptocurrency', 'bitcoin', 'ethereum', 'cryptomarkets', 'altcoin', 'defi']
            }
        }
    except Exception as e:
        return {
            "name": str(subreddit),
            "error": str(e),
            "credibility_factors": {"api_error": True}
        }

def get_author_info_efficiently(reddit, author):
    """Get essential author info with minimal API overhead"""
    if not author or str(author) == "[deleted]":
        return create_basic_author_profile(None)
    
    try:
        # Single API call to get author data
        author_obj = reddit.redditor(str(author))
        
        # Calculate account age
        account_created = datetime.fromtimestamp(author_obj.created_utc, tz=timezoneDt.utc)
        account_age_days = (timezone.now() - account_created).days
        
        # Get all available data in single call
        link_karma = getattr(author_obj, 'link_karma', 0)
        comment_karma = getattr(author_obj, 'comment_karma', 0)
        total_karma = link_karma + comment_karma
        
        # Basic verification checks (no additional API calls)
        is_verified = getattr(author_obj, 'is_verified', False)
        has_verified_email = getattr(author_obj, 'has_verified_email', False)
        is_gold = getattr(author_obj, 'is_gold', False)
        is_mod = getattr(author_obj, 'is_mod', False)
        
        # Calculate credibility indicators
        credibility_indicators = {
            "established_account": account_age_days > 365,
            "high_karma": total_karma > 10000,
            "very_high_karma": total_karma > 100000,
            "balanced_karma": abs(link_karma - comment_karma) / max(total_karma, 1) < 0.8 if total_karma > 0 else False,
            "verified_status": is_verified or has_verified_email,
            "premium_user": is_gold,
            "moderator": is_mod,
            "decent_karma": total_karma > 1000,
            "mature_account": account_age_days > 180
        }
        
        # Calculate author credibility score
        trust_score = calculate_author_trust_score(credibility_indicators, account_age_days, total_karma)
        
        return {
            "exists": True,
            "username": str(author),
            "account_age_days": account_age_days,
            "link_karma": link_karma,
            "comment_karma": comment_karma,
            "total_karma": total_karma,
            "is_verified": is_verified,
            "has_verified_email": has_verified_email,
            "is_gold": is_gold,
            "is_mod": is_mod,
            "credibility_indicators": credibility_indicators,
            "trust_score": trust_score,
            "account_tier": get_account_tier(account_age_days, total_karma)
        }
        
    except Exception as e:
        print(f"Error getting author info for {author}: {e}")
        return create_basic_author_profile(author)

def create_basic_author_profile(author):
    """Create basic author profile without API calls"""
    if not author or str(author) == "[deleted]":
        return {
            "exists": False,
            "username": "[deleted]",
            "trust_score": 0,
            "account_tier": "Unknown",
            "credibility_indicators": {},
            "reason": "Account deleted or unavailable"
        }
    
    return {
        "exists": True,
        "username": str(author),
        "trust_score": 3,  # Neutral score for unknown accounts
        "account_tier": "Unknown",
        "credibility_indicators": {
            "api_limited": True
        },
        "note": "Limited analysis due to API quota constraints"
    }

def extract_comprehensive_post_data_efficient(post):
    """Extract all post data efficiently without additional API calls"""
    try:
        # Calculate post age
        post_age_hours = (timezone.now() - datetime.fromtimestamp(post.created_utc, tz=timezoneDt.utc)).total_seconds() / 3600
        
        # Get all available attributes in one pass
        post_data = {
            # Core identifiers
            "id": post.id,
            "title": post.title,
            "selftext": post.selftext,
            "url": post.url,
            "permalink": f"https://reddit.com{post.permalink}",
            "subreddit": str(post.subreddit),
            "author": str(post.author) if post.author else "[deleted]",
            
            # Timestamps
            "created_utc": post.created_utc,
            "created_datetime": datetime.fromtimestamp(post.created_utc),
            "post_age_hours": post_age_hours,
            
            # Engagement metrics (all available without additional calls)
            "score": post.score,
            "upvote_ratio": post.upvote_ratio,
            "num_comments": post.num_comments,
            "gilded": post.gilded,
            "total_awards_received": post.total_awards_received,
            
            # Post characteristics
            "flair_text": post.link_flair_text,
            "is_self": post.is_self,
            "is_video": post.is_video,
            "over_18": post.over_18,
            "spoiler": post.spoiler,
            "stickied": post.stickied,
            "locked": post.locked,
            "distinguished": post.distinguished,
            "archived": getattr(post, 'archived', False),
            "is_original_content": getattr(post, 'is_original_content', False),
            
            # Media and content
            "domain": post.domain,
            "thumbnail": post.thumbnail,
            "media": bool(post.media),
            "preview": bool(getattr(post, 'preview', None)),
            
            # Content metrics
            "title_length": len(post.title),
            "content_length": len(post.selftext) if post.selftext else 0,
            "total_text_length": len(post.title) + (len(post.selftext) if post.selftext else 0),
            
            # Quick quality indicators
            "has_content": bool(post.selftext),
            "has_flair": bool(post.link_flair_text),
            "is_external_link": not post.is_self,
            "engagement_rate": (post.num_comments / max(post.score, 1)) if post.score > 0 else 0
        }
        
        return post_data
        
    except Exception as e:
        print(f"Error extracting post data: {e}")
        return {"error": str(e), "id": getattr(post, 'id', 'unknown')}

def calculate_author_trust_score(indicators, account_age_days, total_karma):
    """Calculate author credibility score from available indicators"""
    score = 0
    
    # Account age scoring
    if account_age_days > 1095:  # 3+ years
        score += 3
    elif account_age_days > 730:  # 2+ years
        score += 2.5
    elif account_age_days > 365:  # 1+ year
        score += 2
    elif account_age_days > 90:  # 3+ months
        score += 1
    
    # Karma scoring
    if total_karma > 100000:
        score += 3
    elif total_karma > 50000:
        score += 2.5
    elif total_karma > 10000:
        score += 2
    elif total_karma > 1000:
        score += 1
    elif total_karma > 100:
        score += 0.5
    
    # Verification bonuses
    if indicators.get('verified_status'):
        score += 1
    if indicators.get('premium_user'):
        score += 0.5
    if indicators.get('moderator'):
        score += 1
    if indicators.get('balanced_karma'):
        score += 0.5
    
    return min(score, 10)

def get_account_tier(account_age_days, total_karma):
    """Determine account tier based on age and karma"""
    if account_age_days > 1095 and total_karma > 50000:
        return "Veteran"
    elif account_age_days > 730 and total_karma > 10000:
        return "Established" 
    elif account_age_days > 365 and total_karma > 1000:
        return "Trusted"
    elif account_age_days > 90 and total_karma > 100:
        return "Regular"
    elif account_age_days > 30:
        return "New"
    else:
        return "Very New"

# Optimized batch processing function
def fetch_high_credibility_reddit_posts(max_posts=100, trust_score_threshold=6.0, subreddit_limit=3):
    """
    Fetch high-credibility Reddit posts with minimal API calls
    
    Args:
        max_posts (int): Maximum posts to return
        trust_score_threshold (float): Minimum credibility score
        subreddit_limit (int): Number of subreddits to search (max 5 to stay under quota)
    """
    print(f"Fetching high-credibility Reddit posts (threshold: {trust_score_threshold})")
    print(f"API Budget: ~8-9 calls to stay under 10 QPM limit")
    
    # Limit subreddits to stay under API quota
    priority_subreddits = ["CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets", "altcoin"][:subreddit_limit]
    
    all_posts = fetch_reddit_crypto_posts_with_credibility(
        subreddits=priority_subreddits,
        limit=max_posts // len(priority_subreddits),
        analyze_credibility=True
    )
    
    # Filter by credibility threshold
    high_credibility_posts = [
        post for post in all_posts 
        if post.get('credibility_analysis', {}).get('final_trust_score', 0) >= trust_score_threshold
    ]
    
    print(f"Found {len(high_credibility_posts)} posts above credibility threshold")
    
    # Analyze results
    if high_credibility_posts:
        analyze_reddit_results(high_credibility_posts)
    
    return high_credibility_posts[:max_posts]

def analyze_reddit_results(posts):
    """Quick analysis of Reddit results"""
    if not posts:
        return
    
    total_posts = len(posts)
    avg_credibility = sum(p.get('credibility_analysis', {}).get('final_trust_score', 0) for p in posts) / total_posts
    avg_score = sum(p.get('score', 0) for p in posts) / total_posts
    avg_comments = sum(p.get('num_comments', 0) for p in posts) / total_posts
    
    print(f"📈 Reddit Analysis Summary:")
    print(f"   Total Posts: {total_posts}")
    print(f"   Average Credibility: {avg_credibility:.2f}/10")
    print(f"   Average Score: {avg_score:.0f}")
    print(f"   Average Comments: {avg_comments:.0f}")
    
    # Top subreddits
    subreddit_counts = {}
    for post in posts:
        subreddit = post.get('subreddit', 'unknown')
        subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
    
    print(f"   Top Subreddits: {dict(sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)[:3])}")
    
    # Credibility tiers
    tiers = {"Highly Credible": 0, "Credible": 0, "Moderately Credible": 0, "Low Credibility": 0}
    for post in posts:
        tier = post.get('credibility_analysis', {}).get('credibility_tier', 'Low Credibility')
        if tier in tiers:
            tiers[tier] += 1
    
    print(f"   Credibility Distribution: {tiers}")

def analyze_reddit_post_content(post_data):
    """Analyze Reddit post content for quality indicators"""
    title = post_data.get('title', '')
    content = post_data.get('selftext', '')
    
    # Content quality indicators
    quality_factors = {
        "has_meaningful_title": len(title.split()) >= 3,
        "has_content": len(content) > 0,
        "substantial_content": len(content) > 100,
        "not_all_caps": not title.isupper(),
        "no_excessive_punctuation": not bool(re.search(r'[!?]{3,}', title)),
        "has_flair": bool(post_data.get('flair_text')),
        "is_original_content": post_data.get('is_original_content', False),
        "not_nsfw": not post_data.get('over_18', False),
        "not_spoiler": not post_data.get('spoiler', False)
    }
    
    # Calculate content quality score
    quality_score = sum(quality_factors.values()) / len(quality_factors) * 10
    
    # Analyze content themes
    all_text = f"{title} {content}".lower()
    
    # Spam indicators
    spam_indicators = {
        "excessive_caps": len(re.findall(r'[A-Z]{5,}', title)) > 0,
        "excessive_emojis": len(re.findall(r'[😀-🿿]', title + content)) > 0,
        "suspicious_links": len(re.findall(r'bit\.ly|tinyurl|goo\.gl', content)) > 0,
        "repeated_text": has_repeated_phrases(all_text)
    }
    
    spam_score = sum(spam_indicators.values()) / len(spam_indicators) * 10
    
    return {
        "quality_factors": quality_factors,
        "quality_score": quality_score,
        "spam_indicators": spam_indicators,
        "spam_score": spam_score,
        "content_analysis": {
            "title_word_count": len(title.split()),
            "content_word_count": len(content.split()),
            "readability_score": calculate_simple_readability(all_text) if all_text else 0,
            "sentiment_indicators": analyze_basic_sentiment(all_text)
        }
    }

def calculate_reddit_engagement_metrics(post_data):
    """Calculate advanced Reddit engagement metrics"""
    score = post_data.get('score', 0)
    num_comments = post_data.get('num_comments', 0)
    upvote_ratio = post_data.get('upvote_ratio', 0.5)
    post_age_hours = post_data.get('post_age_hours', 1)
    awards = post_data.get('total_awards_received', 0)
    gilded = post_data.get('gilded', 0)
    
    # Calculate engagement velocity
    score_velocity = score / max(post_age_hours, 1)
    comment_velocity = num_comments / max(post_age_hours, 1)
    
    # Engagement quality indicators
    engagement_quality = {
        "high_score": score > 100,
        "good_upvote_ratio": upvote_ratio > 0.8,
        "active_discussion": num_comments > 10,
        "received_awards": awards > 0,
        "gilded_post": gilded > 0,
        "balanced_engagement": (num_comments / max(score, 1)) > 0.05  # Good comment-to-score ratio
    }
    
    return {
        "score_velocity": score_velocity,
        "comment_velocity": comment_velocity,
        "engagement_ratio": num_comments / max(score, 1),
        "quality_indicators": engagement_quality,
        "engagement_quality_score": sum(engagement_quality.values()) / len(engagement_quality) * 10,
        "viral_potential": calculate_reddit_viral_potential(score, num_comments, upvote_ratio, post_age_hours)
    }

def calculate_reddit_crypto_relevance(post_data):
    """Calculate crypto relevance for Reddit post"""
    title = post_data.get('title', '').lower()
    content = post_data.get('selftext', '').lower()
    subreddit = post_data.get('subreddit', '').lower()
    
    # Crypto relevance scoring
    crypto_keywords = {
        "primary": ["bitcoin", "btc", "ethereum", "eth", "cryptocurrency", "crypto", "blockchain"],
        "secondary": ["defi", "nft", "trading", "altcoin", "mining", "wallet", "exchange"],
        "tertiary": ["bull", "bear", "hodl", "doge", "ada", "sol", "matic", "chainlink"]
    }
    
    all_text = f"{title} {content}"
    relevance_score = 0
    keyword_matches = {}
    
    for category, keywords in crypto_keywords.items():
        matches = []
        for keyword in keywords:
            count = all_text.count(keyword)
            if count > 0:
                matches.append({"keyword": keyword, "count": count})
                if category == "primary":
                    relevance_score += count * 3
                elif category == "secondary":
                    relevance_score += count * 2
                else:
                    relevance_score += count * 1
        
        if matches:
            keyword_matches[category] = matches
    
    # Subreddit relevance bonus
    crypto_subreddits = ["cryptocurrency", "bitcoin", "ethereum", "cryptomarkets", "altcoin", "defi"]
    subreddit_bonus = 2 if any(crypto_sub in subreddit for crypto_sub in crypto_subreddits) else 0
    
    final_score = min((relevance_score + subreddit_bonus) / 10, 10.0)
    
    return {
        "crypto_relevance_score": final_score,
        "keyword_matches": keyword_matches,
        "subreddit_crypto_focused": subreddit_bonus > 0,
        "is_crypto_relevant": final_score > 3.0
    }

def calculate_reddit_trust_score(post_data, author_info, subreddit_info, content_analysis, engagement_metrics, crypto_relevance):
    """Calculate comprehensive Reddit post credibility score"""
    
    # Author credibility (0-3 points)
    author_score = 0
    if author_info.get('exists'):
        indicators = author_info.get('credibility_indicators', {})
        author_score = sum([
            2 if indicators.get('established_account') else 0,
            2 if indicators.get('high_karma') else 0,
            1 if indicators.get('verified_status') else 0,
            1 if indicators.get('balanced_karma') else 0
        ]) / 2  # Normalize to 0-3
    
    # Subreddit credibility (0-2 points)
    subreddit_factors = subreddit_info.get('credibility_factors', {})
    subreddit_score = sum([
        1 if subreddit_factors.get('large_community') else 0,
        1 if subreddit_factors.get('established') else 0,
        0.5 if subreddit_factors.get('active_community') else 0
    ])
    
    # Content quality (0-3 points)
    content_score = content_analysis.get('quality_score', 0) * 0.3
    spam_penalty = content_analysis.get('spam_score', 0) * 0.1
    content_final = max(0, content_score - spam_penalty)
    
    # Engagement quality (0-2 points)
    engagement_score = engagement_metrics.get('engagement_quality_score', 0) * 0.2
    
    # Final credibility score
    final_score = min(author_score + subreddit_score + content_final + engagement_score, 10.0)
    
    # Credibility factors summary
    credibility_factors = {
        "trusted_author": author_info.get('exists') and sum(author_info.get('credibility_indicators', {}).values()) >= 2,
        "established_subreddit": sum(subreddit_factors.values()) >= 2,
        "quality_content": content_analysis.get('quality_score', 0) > 7,
        "good_engagement": engagement_metrics.get('engagement_quality_score', 0) > 6,
        "crypto_relevant": crypto_relevance.get('is_crypto_relevant', False),
        "not_spam": content_analysis.get('spam_score', 10) < 3,
        "recent_post": post_data.get('post_age_hours', 999) < 168  # Less than 1 week
    }
    
    return {
        "final_trust_score": final_score,
        "score_breakdown": {
            "author_credibility": author_score,
            "subreddit_credibility": subreddit_score,
            "content_quality": content_final,
            "engagement_quality": engagement_score
        },
        "credibility_factors": credibility_factors,
        "credibility_tier": get_credibility_tier(final_score),
        "analysis_timestamp": datetime.now().isoformat()
    }

def get_credibility_tier(score):
    """Get credibility tier based on score"""
    if score >= 8.5:
        return "Highly Credible"
    elif score >= 7.0:
        return "Credible"
    elif score >= 5.0:
        return "Moderately Credible"
    elif score >= 3.0:
        return "Low Credibility"
    else:
        return "Very Low Credibility"

def calculate_reddit_viral_potential(score, comments, upvote_ratio, age_hours):
    """Calculate viral potential of Reddit post"""
    # Viral indicators
    score_velocity = score / max(age_hours, 1)
    comment_velocity = comments / max(age_hours, 1)
    
    viral_score = 0
    if score_velocity > 50:  # 50+ score per hour
        viral_score += 3
    elif score_velocity > 20:
        viral_score += 2
    elif score_velocity > 10:
        viral_score += 1
    
    if comment_velocity > 5:  # 5+ comments per hour
        viral_score += 2
    elif comment_velocity > 2:
        viral_score += 1
    
    if upvote_ratio > 0.9:
        viral_score += 2
    elif upvote_ratio > 0.8:
        viral_score += 1
    
    return min(viral_score, 10)

def has_repeated_phrases(text):
    """Check for repeated phrases that might indicate spam"""
    words = text.split()
    if len(words) < 10:
        return False
    
    # Check for repeated 3-word phrases
    phrases = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    phrase_counts = {}
    for phrase in phrases:
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
    
    return any(count > 2 for count in phrase_counts.values())

def analyze_basic_sentiment(text):
    """Basic sentiment analysis"""
    positive_words = ['good', 'great', 'excellent', 'amazing', 'bullish', 'up', 'gain', 'profit', 'moon']
    negative_words = ['bad', 'terrible', 'awful', 'bearish', 'down', 'loss', 'crash', 'dump']
    
    positive_count = sum(text.count(word) for word in positive_words)
    negative_count = sum(text.count(word) for word in negative_words)
    
    return {
        "positive_indicators": positive_count,
        "negative_indicators": negative_count,
        "sentiment_ratio": (positive_count - negative_count) / max(positive_count + negative_count, 1)
    }

def calculate_simple_readability(text):
    """Simple readability score calculation"""
    words = text.split()
    sentences = [s for s in text.split('.') if s.strip()]
    
    if not words or not sentences:
        return 0
    
    avg_words_per_sentence = len(words) / len(sentences)
    
    # Simple readability score (lower is more readable)
    # Scale from 0-10 where 10 is most readable
    if avg_words_per_sentence < 10:
        return 10
    elif avg_words_per_sentence < 15:
        return 8
    elif avg_words_per_sentence < 20:
        return 6
    elif avg_words_per_sentence < 25:
        return 4
    else:
        return 2

# Test function with API quota awareness
if __name__ == "__main__":
    print("🚀 Testing Optimized Reddit Fetching (API Quota Aware)")
    print("=" * 60)
    
    # Test with strict API limits
    high_credibility_posts = fetch_high_credibility_reddit_posts(
        max_posts=50,
        trust_score_threshold=6.5,
        subreddit_limit=3  # Only 3 subreddits to stay under quota
    )
    
    if high_credibility_posts:
        print(f"\n🏆 Top 3 Highest Credibility Reddit Posts:")
        for i, post in enumerate(high_credibility_posts[:3], 1):
            cred_score = post.get('credibility_analysis', {}).get('final_trust_score', 0)
            print(f"{i}. {post.get('title', '')[:80]}...")
            print(f"   r/{post.get('subreddit')} | Score: {post.get('score')} | Comments: {post.get('num_comments')}")
            print(f"   Credibility: {cred_score:.2f}/10 | Author: {post.get('author')}")
            print(f"   {post.get('permalink')}")
            print()
    else:
        print("No high-credibility posts found")   