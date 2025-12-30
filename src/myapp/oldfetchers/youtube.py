# youtube fetcher
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import time
import re
from youtube_transcript_api import YouTubeTranscriptApi
import isodate  # For parsing YouTube duration
import requests 
import json
   
load_dotenv()
  
# YouTube API 
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def search_latest_high_credibility_crypto_videos_optimized(max_results=50, days_back=7, min_views=1000, trust_score_threshold=7.0):
    """
    Search for latest crypto videos with highest credibility - OPTIMIZED FOR 10 QPM LIMIT
    
    Args:
        max_results (int): Maximum number of videos to return
        days_back (int): How many days back to search
        min_views (int): Minimum view count for credibility
        trust_score_threshold (float): Minimum credibility score
    """
    if not YOUTUBE_API_KEY:
        print("YouTube API key not provided")
        return []
    
    print(f"Searching for latest high-credibility crypto videos (API quota optimized)")
    print(f"Target: {max_results} videos, Min credibility: {trust_score_threshold}")
    
    # OPTIMIZATION 1: Reduce to 3 most effective queries to minimize API calls
    priority_crypto_queries = [
        "bitcoin price analysis today",
        "cryptocurrency market analysis", 
        "crypto trading analysis"
    ]
    
    all_videos = []
    published_after = (timezone.now() - timedelta(days=days_back)).isoformat()
    
    api_calls_made = 0
    start_time = time.time()
    
    # OPTIMIZATION 2: Single comprehensive search per query
    for query in priority_crypto_queries:
        try:
            print(f"Searching: '{query}'...")
            
            # Single API call for search
            search_response = make_youtube_search_call(query, published_after, min(50, max_results))
            api_calls_made += 1
            
            if not search_response or not search_response.get("items"):
                continue
            
            # Extract video IDs
            video_ids = [item["id"]["videoId"] for item in search_response["items"]]
            
            if not video_ids:
                continue
            
            # OPTIMIZATION 3: Get all video details + channel info in batched calls
            video_details_with_channels = get_comprehensive_details_batched(video_ids)
            api_calls_made += len(video_details_with_channels.get('api_calls', []))
            
            # Process videos with minimal additional processing
            for video in video_details_with_channels.get('videos', []):
                if video.get('view_count', 0) >= min_views:
                    # Enhance with credibility analysis (NO additional API calls)
                    enhanced_video = enhance_video_efficiently(video, query)
                    
                    if enhanced_video.get('final_trust_score', 0) >= trust_score_threshold:
                        all_videos.append(enhanced_video)
            
            print(f"Found {len([v for v in video_details_with_channels.get('videos', []) if v.get('view_count', 0) >= min_views])} qualifying videos")
            print(f"API calls made: {api_calls_made}")
            
            # OPTIMIZATION 4: Strict rate limiting
            if api_calls_made >= 8:  # Stay under 10 QPM with buffer
                wait_time = 60 - (time.time() - start_time)
                if wait_time > 0:
                    print(f"⏳ Rate limit: waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    api_calls_made = 0
                    start_time = time.time()
            else:
                time.sleep(2)  # Small delay between queries
                
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
            continue
    
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
    
    # OPTIMIZATION 5: Get transcripts only for top videos to save processing time
    final_videos = []
    transcript_count = 0
    max_transcripts = min(10, len(sorted_videos))  # Limit transcript fetching
    
    for i, video in enumerate(sorted_videos[:max_results]):
        if transcript_count < max_transcripts:
            print(f"📝 Getting transcript for: {video.get('title', '')[:50]}...")
            transcript = get_enhanced_video_transcript(video.get('video_id'))
            video['transcript'] = transcript
            transcript_count += 1
        else:
            video['transcript'] = {"available": False, "reason": "Quota optimization - transcript skipped"}
        
        final_videos.append(video)
    
    print(f"Final result: {len(final_videos)} high-credibility crypto videos")
    print(f"🔧 Total API calls made: {api_calls_made}")
    return final_videos

def make_youtube_search_call(query, published_after, max_results):
    """Make a single optimized YouTube search API call"""
    try:
        search_url = "https://www.googleapis.com/youtube/v3/search"
        search_params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "order": "relevance",
            "publishedAfter": published_after,
            "relevanceLanguage": "en",
            "videoDuration": "medium",  # Filter for medium duration videos (4-20 minutes)
            "videoDefinition": "any",
            "key": YOUTUBE_API_KEY
        }
        
        response = requests.get(search_url, params=search_params, timeout=15)
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        print(f"Error in search API call: {e}")
        return None

def get_comprehensive_details_batched(video_ids):
    """Get video details and channel info in minimal batched API calls"""
    try:
        if not video_ids:
            return {"videos": [], "api_calls": []}
        
        # OPTIMIZATION: Process all video IDs in single call (up to 50)
        videos_with_details = []
        api_calls_made = []
        
        # Get video details (1 API call)
        video_details = get_video_details_batch(video_ids)
        api_calls_made.append("video_details")
        
        if not video_details:
            return {"videos": [], "api_calls": api_calls_made}
        
        # Extract unique channel IDs
        channel_ids = list(set([v.get('channel_id') for v in video_details if v.get('channel_id')]))
        
        # Get channel info in batch (1 API call for up to 50 channels)
        channel_info_batch = get_channel_info_batch(channel_ids)
        api_calls_made.append("channel_details")
        
        # Combine video and channel data
        channel_lookup = {ch.get('channel_id'): ch for ch in channel_info_batch}
        
        for video in video_details:
            channel_id = video.get('channel_id')
            channel_info = channel_lookup.get(channel_id, {})
            
            # Combine video and channel data
            combined_video = {
                **video,
                "channel_info": channel_info
            }
            videos_with_details.append(combined_video)
        
        return {
            "videos": videos_with_details,
            "api_calls": api_calls_made
        }
        
    except Exception as e:
        print(f"Error in batched details fetching: {e}")
        return {"videos": [], "api_calls": ["error"]}

def get_video_details_batch(video_ids):
    """Get video details in single batched API call"""
    try:
        # YouTube API allows up to 50 IDs per request
        chunk_size = 50
        all_videos = []
        
        for i in range(0, len(video_ids), chunk_size):
            chunk = video_ids[i:i + chunk_size]
            
            url = "https://www.googleapis.com/youtube/v3/videos"
            params = {
                "part": "snippet,statistics,contentDetails,status",  # Removed less critical parts
                "id": ",".join(chunk),
                "key": YOUTUBE_API_KEY
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("items", []):
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})
                content = item.get("contentDetails", {})
                status = item.get("status", {})
                
                # Parse duration efficiently
                duration_seconds = parse_duration_fast(content.get("duration", "PT0S"))
                
                # Parse publish date for timestamp sorting
                published_timestamp = parse_timestamp_fast(snippet.get("publishedAt", ""))
                
                video_data = {
                    # Core data
                    "video_id": item.get("id"),
                    "title": snippet.get("title"),
                    "description": snippet.get("description", "")[:1000],  # Limit description length
                    "published_at": snippet.get("publishedAt"),
                    "published_timestamp": published_timestamp,
                    "channel_title": snippet.get("channelTitle"),
                    "channel_id": snippet.get("channelId"),
                    
                    # Essential stats
                    "view_count": int(stats.get("viewCount", 0)),
                    "like_count": int(stats.get("likeCount", 0)),
                    "comment_count": int(stats.get("commentCount", 0)),
                    
                    # Content details
                    "duration_seconds": duration_seconds,
                    "duration_formatted": format_duration(duration_seconds),
                    "caption": content.get("caption", "false") == "true",
                    
                    # Status
                    "privacy_status": status.get("privacyStatus"),
                    "embeddable": status.get("embeddable", True),
                    
                    # URLs
                    "url": f"https://www.youtube.com/watch?v={item.get('id')}",
                    "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url")
                }
                all_videos.append(video_data)
        
        return all_videos
        
    except Exception as e:
        print(f"Error fetching video details batch: {e}")
        return []

def get_channel_info_batch(channel_ids):
    """Get channel info in single batched API call"""
    try:
        if not channel_ids:
            return []
        
        # Process up to 50 channels per call
        chunk_size = 50
        all_channels = []
        
        for i in range(0, len(channel_ids), chunk_size):
            chunk = channel_ids[i:i + chunk_size]
            
            url = "https://www.googleapis.com/youtube/v3/channels"
            params = {
                "part": "snippet,statistics,status",  # Removed less critical parts
                "id": ",".join(chunk),
                "key": YOUTUBE_API_KEY
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("items", []):
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})
                status = item.get("status", {})
                
                # Calculate channel age efficiently
                channel_age_days = calculate_channel_age_fast(snippet.get("publishedAt", ""))
                
                channel_data = {
                    "channel_id": item.get("id"),
                    "channel_name": snippet.get("title"),
                    "subscriber_count": int(stats.get("subscriberCount", 0)),
                    "total_view_count": int(stats.get("viewCount", 0)),
                    "video_count": int(stats.get("videoCount", 0)),
                    "channel_description": snippet.get("description", "")[:200],  # Limit length
                    "channel_created": snippet.get("publishedAt"),
                    "channel_age_days": channel_age_days,
                    "country": snippet.get("country"),
                    "subscriber_count_hidden": stats.get("hiddenSubscriberCount", False),
                    "privacy_status": status.get("privacyStatus")
                }
                all_channels.append(channel_data)
        
        return all_channels
        
    except Exception as e:
        print(f"Error fetching channel info batch: {e}")
        return []

def enhance_video_efficiently(video, search_query):
    """Enhance video with credibility analysis - NO additional API calls"""
    try:
        channel_info = video.get('channel_info', {})
        
        # Calculate all metrics without API calls
        engagement_metrics = calculate_engagement_metrics_fast(video, channel_info)
        content_quality = analyze_content_quality_fast(video)
        crypto_relevance = calculate_crypto_relevance_fast(video, search_query)
        credibility_analysis = calculate_trust_score_fast(
            video, channel_info, engagement_metrics, content_quality, crypto_relevance
        )
        
        # Combine all data
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

# Fast calculation functions (no API calls)
def calculate_engagement_metrics_fast(video, channel_info):
    """Calculate engagement metrics quickly without API calls"""
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
    """Analyze content quality quickly"""
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
    if video.get('embeddable'):
        quality_score += 1
    
    # Professional indicators
    if not re.search(r'[!]{3,}|URGENT|BREAKING', title.upper()):
        quality_score += 2
    
    return {
        "content_quality_score": quality_score,
        "duration_category": categorize_video_duration(duration)
    }

def calculate_crypto_relevance_fast(video, search_query):
    """Calculate crypto relevance quickly"""
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
    """Calculate credibility score quickly"""
    
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
        content_quality.get('content_quality_score', 0) * 0.1 +  # Reduced weight
        performance_score * 0.3 +
        crypto_relevance.get('crypto_relevance_score', 0) * 0.2
    )
    
    return {
        "final_trust_score": min(final_score, 10.0),
        "channel_authority_score": channel_score,
        "performance_score": performance_score
    }

# Fast parsing helper functions
def parse_duration_fast(duration_iso):
    """Fast duration parsing"""
    try:
        return int(isodate.parse_duration(duration_iso).total_seconds())
    except:
        return 0

def parse_timestamp_fast(timestamp_str):
    """Fast timestamp parsing"""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).timestamp()
    except:
        return 0

def calculate_channel_age_fast(created_date_str):
    """Fast channel age calculation"""
    try:
        created_date = datetime.fromisoformat(created_date_str.replace('Z', '+00:00'))
        return (timezone.now() - created_date).days
    except:
        return 0

def format_duration(seconds):
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds//60}m {seconds%60}s"
    else:
        return f"{seconds//3600}h {(seconds%3600)//60}m"

def categorize_video_duration(seconds):
    """Categorize video duration for analysis"""
    if seconds < 300:  # 5 minutes
        return "short"
    elif seconds < 1800:  # 30 minutes
        return "medium"
    elif seconds < 3600:  # 1 hour
        return "long"
    else:
        return "very_long"

def get_enhanced_video_transcript(video_id):
    """Get video transcript efficiently (no API quota impact)"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try English first
        try:
            transcript = transcript_list.find_transcript(['en'])
            transcript_data = transcript.fetch()
            full_text = " ".join([entry['text'] for entry in transcript_data])
            
            return {
                "available": True,
                "language": "en",
                "text": full_text[:2000],  # Limit text length for processing
                "word_count": len(full_text.split()),
                "auto_generated": False
            }
        except:
            # Try auto-generated
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
                transcript_data = transcript.fetch()
                full_text = " ".join([entry['text'] for entry in transcript_data])
                
                return {
                    "available": True,
                    "language": "en",
                    "text": full_text[:2000],
                    "word_count": len(full_text.split()),
                    "auto_generated": True
                }
            except:
                pass
        
        return {"available": False, "reason": "No transcript available"}
        
    except Exception as e:
        return {"available": False, "error": str(e)}

# Optimized main function for quota management
def fetch_high_credibility_youtube_videos(max_videos=20, trust_score_threshold=7.0, days_back=3):
    """
    Main function optimized for API quota limits
    
    Args:
        max_videos (int): Maximum videos to return (reduced default)
        trust_score_threshold (float): Minimum credibility score
        days_back (int): How many days back to search (reduced default)
    """
    print(f"🚀 Starting YouTube search with quota optimization")
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
        print(f"\nSUCCESS: Found {len(videos)} high-credibility YouTube videos")
        print(f" Execution time: {execution_time:.1f} seconds")
        
        # Quick analysis
        avg_credibility = sum(v.get('final_trust_score', 0) for v in videos) / len(videos)
        total_views = sum(v.get('view_count', 0) for v in videos)
        channels = len(set(v.get('channel_id') for v in videos))
        
        print(f"📈 Average credibility: {avg_credibility:.2f}/10")
        print(f"👁️  Total views: {total_views:,}")
        print(f"📺 Unique channels: {channels}")
        
        # Show top 3
        print(f"\n🏆 Top 3 Videos:")
        for i, video in enumerate(videos[:3], 1):
            print(f"{i}. {video.get('title', '')[:70]}...")
            print(f"   Score: {video.get('final_trust_score', 0):.2f} | Views: {video.get('view_count', 0):,}")
            print(f"   Channel: {video.get('channel_title', '')}")
    else:
        print("No high-credibility videos found")
    
    return videos

# Test with strict quota management
if __name__ == "__main__":
    print("🔥 YouTube High-Credibility Video Fetcher - API Optimized")
    print("=" * 65)
    
    videos = fetch_high_credibility_youtube_videos(
        max_videos=5,
        trust_score_threshold=7.0,
        days_back=5
    )