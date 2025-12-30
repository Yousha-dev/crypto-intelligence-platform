"""
Mock YouTube Fetcher - Uses local JSON files instead of API calls
RAW DATA ONLY - All analysis happens in services
"""
import json
import time
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


def fetch_youtube_videos(queries=None, max_results=50, days_back=7, min_views=1000):
    """
    MOCK: Fetch cryptocurrency videos from YouTube - RAW DATA ONLY
    
    Args:
        queries (list): Search queries
        max_results (int): Maximum videos to return
        days_back (int): How many days back to search
        min_views (int): Minimum view count filter
    
    Returns:
        list: Raw video data (same structure as real API)
    """
    if queries is None:
        queries = [
            "bitcoin price analysis today",
            "cryptocurrency market analysis",
            "crypto trading analysis"
        ]
    
    print(f"[MOCK] Fetching YouTube videos from {len(queries)} queries (raw data)")
    print(f"Loading from: {MOCK_DATA_DIR / 'youtube.json'}")
    
    time.sleep(0.1)
    
    videos = load_mock_data("youtube.json")
    
    if not videos:
        print("️ No mock YouTube data found")
        return []
    
    print(f"Loaded videos count: {len(videos)}") 
    
    all_videos = []
    
    for query in queries:
        for video in videos[:max_results]:
            try:
                view_count = video.get('view_count', 0)
                
                # Apply minimum views filter
                if view_count < min_views:
                    continue
                
                # Build channel info (raw)
                channel_data = video.get('channel_info', {})
                channel_info = {
                    "channel_id": channel_data.get("channel_id", video.get("channel_id", "")),
                    "channel_name": channel_data.get("channel_name", video.get("channel_title", "")),
                    "channel_description": channel_data.get("channel_description", "")[:500],
                    "channel_created": channel_data.get("channel_created"),
                    "subscriber_count": channel_data.get("subscriber_count", 0),
                    "total_view_count": channel_data.get("total_view_count", 0),
                    "video_count": channel_data.get("video_count", 0),
                    "subscriber_count_hidden": channel_data.get("subscriber_count_hidden", False),
                    "country": channel_data.get("country"),
                }
                
                # Extract ONLY raw API fields
                video_data = {
                    # Raw video fields
                    "video_id": video.get("video_id", video.get("id", "")),
                    "title": video.get("title", ""),
                    "description": video.get("description", "")[:2000],
                    "published_at": video.get("published_at", timezone.now().isoformat()),
                    "channel_id": video.get("channel_id", ""),
                    "channel_title": video.get("channel_title", ""),
                    "thumbnails": video.get("thumbnails", {}),
                    "tags": video.get("tags", []),
                    "category_id": video.get("category_id"),
                    "default_language": video.get("default_language"),
                    "default_audio_language": video.get("default_audio_language"),
                    
                    # Raw statistics (from API)
                    "view_count": view_count,
                    "like_count": video.get("like_count", 0),
                    "comment_count": video.get("comment_count", 0),
                    
                    # Raw content details
                    "duration": video.get("duration"),
                    "duration_seconds": video.get("duration_seconds", 0),
                    "dimension": video.get("dimension"),
                    "definition": video.get("definition"),
                    "caption": video.get("caption", False),
                    "licensed_content": video.get("licensed_content", False),
                    
                    # Raw status
                    "privacy_status": video.get("privacy_status", "public"),
                    "embeddable": video.get("embeddable", True),
                    "made_for_kids": video.get("made_for_kids", False),
                    
                    # Raw channel info
                    "channel_info": channel_info,
                    
                    # Convenience fields
                    "url": f"https://www.youtube.com/watch?v={video.get('video_id', video.get('id', ''))}",
                    "thumbnail": video.get("thumbnail", video.get("thumbnails", {}).get("high", {}).get("url")),
                    
                    # Metadata
                    "platform": "youtube",
                    "query": query,
                    "fetched_at": datetime.now().isoformat()
                }
                
                all_videos.append(video_data)
                
            except Exception as e:
                print(f"Error extracting video: {e}")
                continue
    
    # Remove duplicates by video ID
    unique_videos = {v['video_id']: v for v in all_videos if v.get('video_id')}
    all_videos = list(unique_videos.values())
    
    print(f"[MOCK] Total YouTube videos fetched: {len(all_videos)}")
    
    return all_videos


def get_video_transcript(video_id):
    """
    MOCK: Get video transcript (no API quota impact)
    
    Returns:
        dict: Raw transcript data
    """
    # Return mock transcript data
    return {
        "available": True,
        "language": "en",
        "text": "This is a mock transcript for testing purposes. Bitcoin and cryptocurrency analysis content...",
        "word_count": 150,
        "auto_generated": True
    }


# ============================================================================
# TEST
# ============================================================================
 
if __name__ == "__main__":
    print("🔥 [MOCK] YouTube Fetcher - Raw Data Only")
    print("=" * 65)
    
    videos = fetch_youtube_videos(
        queries=["bitcoin price analysis"],
        max_results=5,
        days_back=7
    )
    
    if videos:
        print(f"\n📺 Sample Video:")
        video = videos[0]
        print(f"   Title: {video.get('title', '')[:60]}...")
        print(f"   Views: {video.get('view_count', 0):,}")
        print(f"   Likes: {video.get('like_count', 0):,}")
        print(f"   Channel: {video.get('channel_title')}")
        print(f"   Fields: {list(video.keys())}")