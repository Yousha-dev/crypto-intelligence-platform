"""
Mock Reddit Fetcher - Uses local JSON files instead of API calls
EXACT match to real reddit.py output structure
"""
import json
import time
import re
from datetime import timezone as timezoneDt
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
# ANALYSIS FUNCTIONS (Exact copies from real reddit.py)
# ============================================================================

def get_subreddit_info_efficiently(subreddit_name: str, subscribers: int = 100000) -> dict:
    """Mock subreddit info matching real structure"""
    created_utc = timezone.now().timestamp() - (365 * 24 * 3600 * 3)  # 3 years old
    
    return {
        "name": subreddit_name,
        "subscribers": subscribers,
        "created_utc": created_utc,
        "public_description": f"Discussion about {subreddit_name}",
        "over18": False,
        "subreddit_type": "public",
        "active_user_count": int(subscribers * 0.01),
        "accounts_active": int(subscribers * 0.01),
        "lang": "en",
        "credibility_factors": {
            "large_community": subscribers > 100000,
            "established": True,
            "active_community": True,
            "public_subreddit": True,
            "crypto_focused": subreddit_name.lower() in ['cryptocurrency', 'bitcoin', 'ethereum', 'cryptomarkets', 'altcoin', 'defi']
        }
    }


def create_basic_author_profile(author_name: str) -> dict:
    """Create basic author profile matching real structure"""
    if not author_name or author_name == "[deleted]":
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
        "username": author_name,
        "trust_score": 3,
        "account_tier": "Unknown",
        "credibility_indicators": {
            "api_limited": True
        },
        "note": "Limited analysis due to API quota constraints"
    }


def get_author_info_efficiently(author_data: dict) -> dict:
    """Get author info matching real reddit.py structure"""
    if not author_data:
        return create_basic_author_profile(None)
    
    username = author_data.get('username', author_data.get('name', '[deleted]'))
    if username == "[deleted]":
        return create_basic_author_profile(None)
    
    account_age_days = author_data.get('account_age_days', 365)
    link_karma = author_data.get('link_karma', 1000)
    comment_karma = author_data.get('comment_karma', 1000)
    total_karma = author_data.get('total_karma', link_karma + comment_karma)
    is_verified = author_data.get('is_verified', False)
    has_verified_email = author_data.get('has_verified_email', True)
    is_gold = author_data.get('is_gold', False)
    is_mod = author_data.get('is_mod', False)
    
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
    
    trust_score = calculate_author_trust_score(credibility_indicators, account_age_days, total_karma)
    
    return {
        "exists": True,
        "username": username,
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


def calculate_author_trust_score(indicators: dict, account_age_days: int, total_karma: int) -> float:
    """Calculate author credibility score from available indicators"""
    score = 0
    
    if account_age_days > 1095:
        score += 3
    elif account_age_days > 730:
        score += 2.5
    elif account_age_days > 365:
        score += 2
    elif account_age_days > 90:
        score += 1
    
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
    
    if indicators.get('verified_status'):
        score += 1
    if indicators.get('premium_user'):
        score += 0.5
    if indicators.get('moderator'):
        score += 1
    if indicators.get('balanced_karma'):
        score += 0.5
    
    return min(score, 10)


def get_account_tier(account_age_days: int, total_karma: int) -> str:
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


def analyze_reddit_post_content(post_data: dict) -> dict:
    """Analyze Reddit post content for quality indicators - EXACT match to real"""
    title = post_data.get('title', '')
    content = post_data.get('selftext', '')
    
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
    
    quality_score = sum(quality_factors.values()) / len(quality_factors) * 10
    
    all_text = f"{title} {content}".lower()
    
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


def calculate_reddit_engagement_metrics(post_data: dict) -> dict:
    """Calculate advanced Reddit engagement metrics - EXACT match to real"""
    score = post_data.get('score', 0)
    num_comments = post_data.get('num_comments', 0)
    upvote_ratio = post_data.get('upvote_ratio', 0.5)
    post_age_hours = post_data.get('post_age_hours', 1)
    awards = post_data.get('total_awards_received', 0)
    gilded = post_data.get('gilded', 0)
    
    score_velocity = score / max(post_age_hours, 1)
    comment_velocity = num_comments / max(post_age_hours, 1)
    
    engagement_quality = {
        "high_score": score > 100,
        "good_upvote_ratio": upvote_ratio > 0.8,
        "active_discussion": num_comments > 10,
        "received_awards": awards > 0,
        "gilded_post": gilded > 0,
        "balanced_engagement": (num_comments / max(score, 1)) > 0.05
    }
    
    return {
        "score_velocity": score_velocity,
        "comment_velocity": comment_velocity,
        "engagement_ratio": num_comments / max(score, 1),
        "quality_indicators": engagement_quality,
        "engagement_quality_score": sum(engagement_quality.values()) / len(engagement_quality) * 10,
        "viral_potential": calculate_reddit_viral_potential(score, num_comments, upvote_ratio, post_age_hours)
    }


def calculate_reddit_crypto_relevance(post_data: dict) -> dict:
    """Calculate crypto relevance for Reddit post - EXACT match to real"""
    title = post_data.get('title', '').lower()
    content = post_data.get('selftext', '').lower()
    subreddit = post_data.get('subreddit', '').lower()
    
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
    
    crypto_subreddits = ["cryptocurrency", "bitcoin", "ethereum", "cryptomarkets", "altcoin", "defi"]
    subreddit_bonus = 2 if any(crypto_sub in subreddit for crypto_sub in crypto_subreddits) else 0
    
    final_score = min((relevance_score + subreddit_bonus) / 10, 10.0)
    
    return {
        "crypto_relevance_score": final_score,
        "keyword_matches": keyword_matches,
        "subreddit_crypto_focused": subreddit_bonus > 0,
        "is_crypto_relevant": final_score > 3.0
    }


def calculate_reddit_trust_score(post_data: dict, author_info: dict, subreddit_info: dict,
                                       content_analysis: dict, engagement_metrics: dict, 
                                       crypto_relevance: dict) -> dict:
    """Calculate comprehensive Reddit post credibility score - EXACT match to real"""
    
    # Author credibility (0-3 points)
    author_score = 0
    if author_info.get('exists'):
        indicators = author_info.get('credibility_indicators', {})
        author_score = sum([
            2 if indicators.get('established_account') else 0,
            2 if indicators.get('high_karma') else 0,
            1 if indicators.get('verified_status') else 0,
            1 if indicators.get('balanced_karma') else 0
        ]) / 2
    
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
        "recent_post": post_data.get('post_age_hours', 999) < 168
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


def get_credibility_tier(score: float) -> str:
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


def calculate_reddit_viral_potential(score: int, comments: int, upvote_ratio: float, age_hours: float) -> float:
    """Calculate viral potential of Reddit post"""
    score_velocity = score / max(age_hours, 1)
    comment_velocity = comments / max(age_hours, 1)
    
    viral_score = 0
    if score_velocity > 50:
        viral_score += 3
    elif score_velocity > 20:
        viral_score += 2
    elif score_velocity > 10:
        viral_score += 1
    
    if comment_velocity > 5:
        viral_score += 2
    elif comment_velocity > 2:
        viral_score += 1
    
    if upvote_ratio > 0.9:
        viral_score += 2
    elif upvote_ratio > 0.8:
        viral_score += 1
    
    return min(viral_score, 10)


def has_repeated_phrases(text: str) -> bool:
    """Check for repeated phrases that might indicate spam"""
    words = text.split()
    if len(words) < 10:
        return False
    
    phrases = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    phrase_counts = {}
    for phrase in phrases:
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
    
    return any(count > 2 for count in phrase_counts.values())


def analyze_basic_sentiment(text: str) -> dict:
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


def calculate_simple_readability(text: str) -> float:
    """Simple readability score calculation"""
    words = text.split()
    sentences = [s for s in text.split('.') if s.strip()]
    
    if not words or not sentences:
        return 0
    
    avg_words_per_sentence = len(words) / len(sentences)
    
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


def extract_comprehensive_post_data_efficient(post: dict) -> dict:
    """Extract all post data efficiently - EXACT match to real"""
    created_utc = post.get('created_utc', int(time.time()))
    post_age_hours = (timezone.now() - datetime.fromtimestamp(created_utc, tz=timezoneDt.utc)).total_seconds() / 3600
    
    return {
        "id": post.get('id'),
        "title": post.get('title', ''),
        "selftext": post.get('selftext', ''),
        "url": post.get('url', ''),
        "permalink": post.get('permalink', f"https://reddit.com/r/{post.get('subreddit', 'unknown')}/comments/{post.get('id', '')}"),
        "subreddit": post.get('subreddit', ''),
        "author": post.get('author', '[deleted]'),
        "created_utc": created_utc,
        "created_datetime": datetime.fromtimestamp(created_utc),
        "post_age_hours": post_age_hours,
        "score": post.get('score', 0),
        "upvote_ratio": post.get('upvote_ratio', 0.5),
        "num_comments": post.get('num_comments', 0),
        "gilded": post.get('gilded', 0),
        "total_awards_received": post.get('total_awards_received', 0),
        "flair_text": post.get('link_flair_text'),
        "is_self": post.get('is_self', True),
        "is_video": post.get('is_video', False),
        "over_18": post.get('over_18', False),
        "spoiler": post.get('spoiler', False),
        "stickied": post.get('stickied', False),
        "locked": post.get('locked', False),
        "distinguished": post.get('distinguished'),
        "archived": post.get('archived', False),
        "is_original_content": post.get('is_original_content', False),
        "domain": post.get('domain', 'self.reddit'),
        "thumbnail": post.get('thumbnail'),
        "media": bool(post.get('media')),
        "preview": bool(post.get('preview')),
        "title_length": len(post.get('title', '')),
        "content_length": len(post.get('selftext', '')),
        "total_text_length": len(post.get('title', '')) + len(post.get('selftext', '')),
        "has_content": bool(post.get('selftext')),
        "has_flair": bool(post.get('link_flair_text')),
        "is_external_link": not post.get('is_self', True),
        "engagement_rate": (post.get('num_comments', 0) / max(post.get('score', 1), 1)) if post.get('score', 0) > 0 else 0
    }


# ============================================================================
# MOCK FETCHER FUNCTIONS (EXACT same signatures as real fetchers)
# ============================================================================

def fetch_reddit_crypto_posts_with_credibility(subreddits=None, limit=50, time_filter="day", 
                                               sort_type="hot", analyze_credibility=True):
    """
    MOCK: Fetch cryptocurrency posts from Reddit with comprehensive credibility analysis
    EXACT same signature and output structure as real fetcher
    """
    if subreddits is None:
        subreddits = ["CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets", "altcoin"]
    
    print(f"[MOCK] Fetching Reddit posts from {len(subreddits)} subreddits with credibility analysis")
    print(f"Loading from: {MOCK_DATA_DIR / 'reddit.json'}")
    
    time.sleep(0.3)
    
    posts = load_mock_data("reddit.json")
    
    if not posts:
        print("️ No mock Reddit data found")
        return []
    
    # Cache for subreddit info
    subreddit_cache = {}
    author_cache = {}
    
    all_posts = []
    
    for post in posts[:limit]:
        try:
            # Extract comprehensive post data
            post_data = extract_comprehensive_post_data_efficient(post)
            
            if analyze_credibility:
                # Get subreddit info
                subreddit_name = post_data.get('subreddit', 'CryptoCurrency')
                if subreddit_name not in subreddit_cache:
                    subreddit_cache[subreddit_name] = get_subreddit_info_efficiently(
                        subreddit_name,
                        post.get('subreddit_info', {}).get('subscribers', 100000)
                    )
                subreddit_info = subreddit_cache[subreddit_name]
                
                # Get author info
                author_name = post_data.get('author', '[deleted]')
                if author_name not in author_cache:
                    author_cache[author_name] = get_author_info_efficiently(post.get('author_info', {}))
                author_info = author_cache[author_name]
                
                # Perform analysis
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
            
            all_posts.append(post_data)
            
        except Exception as e:
            print(f"Error processing post {post.get('id')}: {e}")
            continue
    
    # Sort by credibility if analysis was performed
    if analyze_credibility and all_posts:
        all_posts.sort(
            key=lambda x: x.get('credibility_analysis', {}).get('final_trust_score', 0),
            reverse=True
        )
        
        avg_credibility = sum(p.get('credibility_analysis', {}).get('final_trust_score', 0) for p in all_posts) / len(all_posts)
        print(f"📈 [MOCK] Average credibility score: {avg_credibility:.2f}/10")
    
    print(f"[MOCK] Reddit: Processed {len(all_posts)} posts")
    
    return all_posts


def fetch_high_credibility_reddit_posts(max_posts=100, trust_score_threshold=6.0, subreddit_limit=3):
    """
    MOCK: Fetch high-credibility Reddit posts with minimal API calls
    EXACT same signature as real fetcher
    """
    print(f"[MOCK] Fetching high-credibility Reddit posts (threshold: {trust_score_threshold})")
    print(f"API Budget: ~8-9 calls to stay under 10 QPM limit")
    
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
    
    print(f"[MOCK] Found {len(high_credibility_posts)} posts above credibility threshold")
    
    if high_credibility_posts:
        analyze_reddit_results(high_credibility_posts)
    
    return high_credibility_posts[:max_posts]


def analyze_reddit_results(posts: list):
    """Quick analysis of Reddit results - EXACT match to real"""
    if not posts:
        return
    
    total_posts = len(posts)
    avg_credibility = sum(p.get('credibility_analysis', {}).get('final_trust_score', 0) for p in posts) / total_posts
    avg_score = sum(p.get('score', 0) for p in posts) / total_posts
    avg_comments = sum(p.get('num_comments', 0) for p in posts) / total_posts
    
    print(f"📈 [MOCK] Reddit Analysis Summary:")
    print(f"   Total Posts: {total_posts}")
    print(f"   Average Credibility: {avg_credibility:.2f}/10")
    print(f"   Average Score: {avg_score:.0f}")
    print(f"   Average Comments: {avg_comments:.0f}")
    
    subreddit_counts = {}
    for post in posts:
        subreddit = post.get('subreddit', 'unknown')
        subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
    
    print(f"   Top Subreddits: {dict(sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)[:3])}")
    
    tiers = {"Highly Credible": 0, "Credible": 0, "Moderately Credible": 0, "Low Credibility": 0}
    for post in posts:
        tier = post.get('credibility_analysis', {}).get('credibility_tier', 'Low Credibility')
        if tier in tiers:
            tiers[tier] += 1
    
    print(f"   Credibility Distribution: {tiers}")


if __name__ == "__main__":
    print("🧪 Testing Mock Reddit Fetcher (Exact Match)")
    posts = fetch_high_credibility_reddit_posts(max_posts=10, trust_score_threshold=5.0)
    for post in posts[:3]:
        print(f"📝 {post.get('title', 'N/A')[:50]}...")
        print(f"   Score: {post.get('credibility_analysis', {}).get('final_trust_score', 0):.1f}")
        print(f"   Tier: {post.get('credibility_analysis', {}).get('credibility_tier', 'N/A')}")