"""
Mock YouTube Fetcher - Uses local JSON files instead of API calls
EXACT match to real youtube.py output structure
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
# ANALYSIS FUNCTIONS (Exact copies from real youtube.py)
# ============================================================================

def calculate_engagement_metrics_fast(video, channel_info):
    """Calculate engagement metrics quickly without API calls - EXACT match to real"""
    view_count = video.get('view_count', 0)
    like_count = video.get('like_count', 0)
    comment_count = video.get('comment_count', 0)
    subscriber_count = channel_info.get('subscriber_count', 1)
    
    # Basic metrics
    total_engagement = like_count + comment_count
    engagement_rate = (total_engagement / view_count * 100) if view_count > 0 else 0
    like_to_view_ratio = (like_count / view_count * 100) if view_count > 0 else 0
    comment_to_view_ratio = (comment_count / view_count * 100) if view_count > 0 else 0
    
    return {
        "engagement_rate": engagement_rate,
        "like_to_view_ratio": like_to_view_ratio,
        "comment_to_view_ratio": comment_to_view_ratio,
        "engagement_quality_score": min(engagement_rate * 2, 10)  # Scale to 0-10
    }


def analyze_content_quality_fast(video):
    """Analyze content quality quickly - EXACT match to real"""
    title = video.get('title', '')
    description = video.get('description', '')
    duration = video.get('duration_seconds', 0)
    
    quality_score = 0
    
    # Title quality
    if len(title) > 10 and not title.isupper():
        quality_score += 2
    
    # Description quality
    if len(description) > 100:
        quality_score += 2
    
    # Duration appropriateness
    if 180 < duration < 1800:  # 3-30 minutes
        quality_score += 2
    
    # Technical quality
    if video.get('caption'):
        quality_score += 1
    if video.get('embeddable', True):
        quality_score += 1
    
    # Professional indicators
    if not re.search(r'[!]{3,}|URGENT|BREAKING', title.upper()):
        quality_score += 2
    
    return {
        "content_quality_score": quality_score,
        "duration_category": categorize_video_duration(duration)
    }


def calculate_crypto_relevance_fast(video, search_query):
    """Calculate crypto relevance quickly - EXACT match to real"""
    title = video.get('title', '').lower()
    description = video.get('description', '').lower()[:500]  # Limit processing
    
    # Primary crypto keywords (high weight)
    primary_keywords = ["bitcoin", "btc", "ethereum", "eth", "cryptocurrency", "crypto", "blockchain"]
    secondary_keywords = ["defi", "trading", "altcoin", "analysis", "market", "price"]
    
    all_text = f"{title} {description}"
    
    relevance_score = 0
    
    # Count primary keywords
    for keyword in primary_keywords:
        relevance_score += all_text.count(keyword) * 3
    
    # Count secondary keywords
    for keyword in secondary_keywords:
        relevance_score += all_text.count(keyword) * 1
    
    # Query relevance bonus
    if search_query:
        query_words = search_query.lower().split()
        for word in query_words:
            if word in all_text:
                relevance_score += 1
    
    final_score = min(relevance_score / 5, 10.0)  # Scale to 0-10
    
    return {
        "crypto_relevance_score": final_score,
        "is_crypto_focused": final_score > 3.0
    }


def calculate_trust_score_fast(video, channel_info, engagement_metrics, content_quality, crypto_relevance):
    """Calculate credibility score quickly - EXACT match to real"""
    
    # Channel authority (0-4 points)
    subscriber_count = channel_info.get('subscriber_count', 0)
    channel_age_days = channel_info.get('channel_age_days', 0)
    
    channel_score = 0
    if subscriber_count > 1000000:
        channel_score += 2
    elif subscriber_count > 100000:
        channel_score += 1.5
    elif subscriber_count > 10000:
        channel_score += 1
    
    if channel_age_days > 365:
        channel_score += 1
    if channel_info.get('video_count', 0) > 50:
        channel_score += 0.5
    
    # Performance score (0-3 points)
    view_count = video.get('view_count', 0)
    performance_score = 0
    if view_count > 100000:
        performance_score += 2
    elif view_count > 10000:
        performance_score += 1.5
    elif view_count > 1000:
        performance_score += 1
    
    performance_score += min(engagement_metrics.get('engagement_quality_score', 0) / 10, 1)
    
    # Final score calculation
    final_score = (
        channel_score * 0.4 +
        content_quality.get('content_quality_score', 0) * 0.1 +
        performance_score * 0.3 +
        crypto_relevance.get('crypto_relevance_score', 0) * 0.2
    )
    
    return {
        "final_trust_score": min(final_score, 10.0),
        "channel_authority_score": channel_score,
        "performance_score": performance_score
    }


def categorize_video_duration(seconds):
    """Categorize video duration for analysis - EXACT match to real"""
    if seconds < 300:  # 5 minutes
        return "short"
    elif seconds < 1800:  # 30 minutes
        return "medium"
    elif seconds < 3600:  # 1 hour
        return "long"
    else:
        return "very_long"


def format_duration(seconds):
    """Format duration in seconds to human readable format - EXACT match to real"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds//60}m {seconds%60}s"
    else:
        return f"{seconds//3600}h {(seconds%3600)//60}m"


def get_enhanced_video_transcript(video_id):
    """Mock transcript fetching - returns mock data"""
    # Return mock transcript data
    return {
        "available": True,
        "language": "en",
        "text": "This is a mock transcript for testing purposes. Bitcoin and Ethereum analysis...",
        "word_count": 150,
        "auto_generated": True
    }


def enhance_video_efficiently(video, search_query):
    """Enhance video with credibility analysis - EXACT match to real"""
    try:
        channel_info = video.get('channel_info', {})
        
        # Calculate all metrics without API calls
        engagement_metrics = calculate_engagement_metrics_fast(video, channel_info)
        content_quality = analyze_content_quality_fast(video)
        crypto_relevance = calculate_crypto_relevance_fast(video, search_query)
        credibility_analysis = calculate_trust_score_fast(
            video, channel_info, engagement_metrics, content_quality, crypto_relevance
        )
        
        # Combine all data - EXACT match to real structure
        enhanced_video = {
            **video,
            **engagement_metrics,
            **content_quality,
            **crypto_relevance,
            **credibility_analysis,
            "search_query": search_query,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return enhanced_video
        
    except Exception as e:
        print(f"Error enhancing video {video.get('video_id')}: {e}")
        return video


# ============================================================================
# MOCK FETCHER FUNCTIONS (EXACT same signatures as real fetchers)
# ============================================================================

def search_latest_high_credibility_crypto_videos_optimized(max_results=50, days_back=7, 
                                                          min_views=1000, trust_score_threshold=7.0):
    """
    MOCK: Search for latest crypto videos with highest credibility
    EXACT same signature and output structure as real fetcher
    
    Args:
        max_results (int): Maximum number of videos to return
        days_back (int): How many days back to search
        min_views (int): Minimum view count for credibility
        trust_score_threshold (float): Minimum credibility score
    """
    print(f"[MOCK] Searching for latest high-credibility crypto videos (API quota optimized)")
    print(f"Loading from: {MOCK_DATA_DIR / 'youtube.json'}")
    print(f"Target: {max_results} videos, Min credibility: {trust_score_threshold}")
    
    time.sleep(0.3)  # Simulate API delay
    
    videos = load_mock_data("youtube.json")
    
    if not videos:
        print("️ No mock YouTube data found")
        return []
    
    # Priority crypto queries (matching real function)
    priority_crypto_queries = [
        "bitcoin price analysis today",
        "cryptocurrency market analysis", 
        "crypto trading analysis"
    ]
    
    all_videos = []
    api_calls_made = 0
    
    for query in priority_crypto_queries:
        print(f"[MOCK] Searching: '{query}'...")
        api_calls_made += 1
        
        for video in videos[:max_results // len(priority_crypto_queries)]:
            try:
                # Check minimum views
                if video.get('view_count', 0) < min_views:
                    continue
                
                # Build video data structure matching EXACT real output
                video_data = {
                    # Core data
                    "video_id": video.get('video_id', video.get('id', '')),
                    "title": video.get('title', ''),
                    "description": video.get('description', '')[:1000],
                    "published_at": video.get('published_at', timezone.now().isoformat()),
                    "published_timestamp": timezone.now().timestamp() - (days_back * 24 * 3600),
                    "channel_title": video.get('channel_title', 'Unknown'),
                    "channel_id": video.get('channel_id', ''),
                    
                    # Essential stats
                    "view_count": video.get('view_count', 0),
                    "like_count": video.get('like_count', 0),
                    "comment_count": video.get('comment_count', 0),
                    
                    # Content details
                    "duration_seconds": video.get('duration_seconds', 600),
                    "duration_formatted": format_duration(video.get('duration_seconds', 600)),
                    "caption": video.get('caption', True),
                    
                    # Status
                    "privacy_status": video.get('privacy_status', 'public'),
                    "embeddable": video.get('embeddable', True),
                    
                    # URLs
                    "url": f"https://www.youtube.com/watch?v={video.get('video_id', video.get('id', ''))}",
                    "thumbnail": video.get('thumbnail', video.get('thumbnails', {}).get('high', {}).get('url', ''))
                }
                
                # Ensure channel_info exists with proper structure
                channel_info = video.get('channel_info', {})
                if not channel_info:
                    channel_info = {
                        'channel_id': video_data.get('channel_id', ''),
                        'channel_name': video_data.get('channel_title', 'Unknown'),
                        'subscriber_count': 50000,
                        'total_view_count': 1000000,
                        'video_count': 100,
                        'channel_description': '',
                        'channel_created': '',
                        'channel_age_days': 365,
                        'country': None,
                        'subscriber_count_hidden': False,
                        'privacy_status': 'public'
                    }
                
                video_data['channel_info'] = channel_info
                
                # Enhance with credibility analysis
                enhanced_video = enhance_video_efficiently(video_data, query)
                
                # Filter by credibility threshold
                if enhanced_video.get('final_trust_score', 0) >= trust_score_threshold:
                    all_videos.append(enhanced_video)
                
            except Exception as e:
                print(f"Error processing video {video.get('video_id')}: {e}")
                continue
        
        print(f"[MOCK] Found qualifying videos for query: {query}")
        print(f"API calls made: {api_calls_made}")
    
    # Remove duplicates by video ID
    unique_videos = {}
    for video in all_videos:
        video_id = video.get('video_id')
        if video_id not in unique_videos or video.get('final_trust_score', 0) > unique_videos[video_id].get('final_trust_score', 0):
            unique_videos[video_id] = video
    
    # Sort by credibility score and recency
    sorted_videos = sorted(
        unique_videos.values(),
        key=lambda x: (x.get('final_trust_score', 0), x.get('published_timestamp', 0)),
        reverse=True
    )
    
    # Get transcripts for top videos (limited)
    final_videos = []
    transcript_count = 0
    max_transcripts = min(10, len(sorted_videos))
    
    for i, video in enumerate(sorted_videos[:max_results]):
        if transcript_count < max_transcripts:
            print(f"📝 [MOCK] Getting transcript for: {video.get('title', '')[:50]}...")
            transcript = get_enhanced_video_transcript(video.get('video_id'))
            video['transcript'] = transcript
            transcript_count += 1
        else:
            video['transcript'] = {"available": False, "reason": "Quota optimization - transcript skipped"}
        
        final_videos.append(video)
    
    print(f"[MOCK] Final result: {len(final_videos)} high-credibility crypto videos")
    print(f"🔧 Total API calls made: {api_calls_made}")
    
    # Analysis summary
    if final_videos:
        avg_credibility = sum(v.get('final_trust_score', 0) for v in final_videos) / len(final_videos)
        total_views = sum(v.get('view_count', 0) for v in final_videos)
        channels = len(set(v.get('channel_id') for v in final_videos))
        
        print(f"\n[MOCK] SUCCESS: Found {len(final_videos)} high-credibility YouTube videos")
        print(f"📈 Average credibility: {avg_credibility:.2f}/10")
        print(f"👁️  Total views: {total_views:,}")
        print(f"📺 Unique channels: {channels}")
    
    return final_videos


def fetch_high_credibility_youtube_videos(max_videos=20, trust_score_threshold=7.0, days_back=3):
    """
    MOCK: Main function optimized for API quota limits
    EXACT same signature as real fetcher
    
    Args:
        max_videos (int): Maximum videos to return (reduced default)
        trust_score_threshold (float): Minimum credibility score
        days_back (int): How many days back to search (reduced default)
    """
    print(f"🚀 [MOCK] Starting YouTube search with quota optimization")
    print(f"API Budget: 6-8 calls maximum per run")
    print(f"Target: {max_videos} videos, threshold: {trust_score_threshold}")
    
    start_time = time.time()
    
    videos = search_latest_high_credibility_crypto_videos_optimized(
        max_results=max_videos,
        days_back=days_back,
        min_views=5000,  # Higher minimum to ensure quality
        trust_score_threshold=trust_score_threshold
    )
    
    execution_time = time.time() - start_time
    
    if videos:
        print(f"\n [MOCK] Execution time: {execution_time:.1f} seconds")
        
        # Show top 3
        print(f"\n🏆 [MOCK] Top 3 Videos:")
        for i, video in enumerate(videos[:3], 1):
            print(f"{i}. {video.get('title', '')[:70]}...")
            print(f"   Score: {video.get('final_trust_score', 0):.2f} | Views: {video.get('view_count', 0):,}")
            print(f"   Channel: {video.get('channel_title', '')}")
    else:
        print("[MOCK] No high-credibility videos found")
    
    return videos


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🔥 [MOCK] YouTube High-Credibility Video Fetcher - API Optimized")
    print("=" * 65)
    
    videos = fetch_high_credibility_youtube_videos(
        max_videos=5,
        trust_score_threshold=5.0,
        days_back=5
    )
    
    for video in videos[:3]:
        print(f"\n📺 {video.get('title', 'N/A')[:50]}...")
        print(f"   Credibility: {video.get('final_trust_score', 0):.1f}/10")
        print(f"   Views: {video.get('view_count', 0):,}")
        print(f"   Channel: {video.get('channel_title', 'Unknown')}")