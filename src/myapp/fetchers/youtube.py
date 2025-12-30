# youtube fetcher - RAW DATA ONLY
"""
YouTube Fetcher - Extract raw data from YouTube Data API v3
All credibility/sentiment analysis happens in services
"""
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
from django.utils import timezone
import time
import requests
import isodate
from youtube_transcript_api import YouTubeTranscriptApi
from pathlib import Path

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Create output directories
SOCIAL_DIR = Path("social")
SOCIAL_DIR.mkdir(exist_ok=True)


def save_to_json(data, filename):
    """Save data to JSON file in social directory"""
    filepath = SOCIAL_DIR / filename
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved to {filepath}")
    except Exception as e:
        print(f"❌ Error saving to {filepath}: {e}")


def fetch_youtube_videos(queries=None, max_results=50, days_back=7, min_views=1000):
    """
    Fetch cryptocurrency videos from YouTube - RAW DATA ONLY
    
    Args:
        queries (list): Search queries
        max_results (int): Maximum videos to return
        days_back (int): How many days back to search
        min_views (int): Minimum view count filter
    
    Returns:
        list: Raw video data with all available API fields
    """
    if not YOUTUBE_API_KEY:
        print("YouTube API key not provided")
        return []
    
    if queries is None:
        queries = [
            "bitcoin price analysis today",
            "cryptocurrency market analysis",
            "crypto trading analysis"
        ]
    
    print(f"Fetching YouTube videos from {len(queries)} queries (raw data)")
    
    all_videos = []
    published_after = (timezone.now() - timedelta(days=days_back)).isoformat()
    api_calls = 0
    
    for query in queries:
        try:
            print(f"  Searching: {query[:50]}...")
            
            # Search for videos
            search_url = "https://www.googleapis.com/youtube/v3/search"
            search_params = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": min(50, max_results),
                "order": "relevance",
                "publishedAfter": published_after,
                "relevanceLanguage": "en",
                "videoDuration": "medium",
                "key": YOUTUBE_API_KEY
            }
            
            search_response = requests.get(search_url, params=search_params, timeout=15)
            search_response.raise_for_status()
            search_data = search_response.json()
            api_calls += 1
            
            if not search_data.get("items"):
                continue
            
            # Get video IDs
            video_ids = [item["id"]["videoId"] for item in search_data["items"]]
            
            # Get detailed video info
            videos_url = "https://www.googleapis.com/youtube/v3/videos"
            videos_params = {
                "part": "snippet,statistics,contentDetails,status",
                "id": ",".join(video_ids),
                "key": YOUTUBE_API_KEY
            }
            
            videos_response = requests.get(videos_url, params=videos_params, timeout=15)
            videos_response.raise_for_status()
            videos_data = videos_response.json()
            api_calls += 1
            
            # Get channel info
            channel_ids = list(set([
                item["snippet"]["channelId"] 
                for item in videos_data.get("items", [])
            ]))
            
            channels_url = "https://www.googleapis.com/youtube/v3/channels"
            channels_params = {
                "part": "snippet,statistics,status",
                "id": ",".join(channel_ids),
                "key": YOUTUBE_API_KEY
            }
            
            channels_response = requests.get(channels_url, params=channels_params, timeout=15)
            channels_response.raise_for_status()
            channels_data = channels_response.json()
            api_calls += 1
            
            # Build channel lookup
            channel_lookup = {}
            for ch in channels_data.get("items", []):
                channel_lookup[ch["id"]] = {
                    "channel_id": ch["id"],
                    "channel_name": ch["snippet"]["title"],
                    "channel_description": ch["snippet"].get("description", "")[:500],
                    "channel_created": ch["snippet"].get("publishedAt"),
                    "subscriber_count": int(ch["statistics"].get("subscriberCount", 0)),
                    "total_view_count": int(ch["statistics"].get("viewCount", 0)),
                    "video_count": int(ch["statistics"].get("videoCount", 0)),
                    "subscriber_count_hidden": ch["statistics"].get("hiddenSubscriberCount", False),
                    "country": ch["snippet"].get("country"),
                }
            
            # Process videos
            for item in videos_data.get("items", []):
                try:
                    snippet = item["snippet"]
                    stats = item["statistics"]
                    content = item["contentDetails"]
                    status = item["status"]
                    
                    view_count = int(stats.get("viewCount", 0))
                    
                    # Apply minimum views filter
                    if view_count < min_views:
                        continue
                    
                    # Parse duration
                    try:
                        duration_seconds = int(isodate.parse_duration(content.get("duration", "PT0S")).total_seconds())
                    except:
                        duration_seconds = 0
                    
                    # Get channel info
                    channel_info = channel_lookup.get(snippet["channelId"], {})
                    
                    transcript = get_video_transcript(item["id"])
                    
                    # Extract ONLY raw API fields
                    video_data = {
                        # === RAW VIDEO FIELDS ===
                        "video_id": item["id"],
                        "title": snippet["title"],
                        "description": snippet.get("description", "")[:2000],
                        "published_at": snippet["publishedAt"],
                        "channel_id": snippet["channelId"],
                        "channel_title": snippet["channelTitle"],
                        "thumbnails": snippet.get("thumbnails", {}),
                        "tags": snippet.get("tags", []),
                        "category_id": snippet.get("categoryId"),
                        "default_language": snippet.get("defaultLanguage"),
                        "default_audio_language": snippet.get("defaultAudioLanguage"),
                        
                        # === RAW STATISTICS (from API) ===
                        "view_count": view_count,
                        "like_count": int(stats.get("likeCount", 0)),
                        "comment_count": int(stats.get("commentCount", 0)),
                        
                        # === RAW CONTENT DETAILS ===
                        "duration": content.get("duration"),
                        "duration_seconds": duration_seconds,
                        "dimension": content.get("dimension"),
                        "definition": content.get("definition"),
                        "caption": content.get("caption") == "true",
                        "licensed_content": content.get("licensedContent", False),
                        
                        # === RAW STATUS ===
                        "privacy_status": status.get("privacyStatus"),
                        "embeddable": status.get("embeddable", True),
                        "made_for_kids": status.get("madeForKids", False),
                        
                        # === RAW CHANNEL INFO ===
                        "channel_info": channel_info,
                        
                        # === CONVENIENCE FIELDS ===
                        "url": f"https://www.youtube.com/watch?v={item['id']}",
                        "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url"),
                        
                        # === METADATA ===
                        "platform": "youtube",
                        "query": query,
                        "fetched_at": datetime.now().isoformat(),
                        "transcript": transcript
                    }
                    
                    all_videos.append(video_data)
                    
                except Exception as e:
                    print(f"    Error extracting video: {e}")
                    continue
            
            print(f"    Fetched {len([v for v in all_videos if v.get('query') == query])} videos")
            
            # Rate limiting
            time.sleep(2)
            
        except Exception as e:
            print(f"    Error searching '{query}': {e}")
            continue
    
    # Remove duplicates
    unique_videos = {v['video_id']: v for v in all_videos}
    all_videos = list(unique_videos.values())
    
    print(f"Total YouTube videos fetched: {len(all_videos)}")
    print(f"API calls made: {api_calls}")
    
    # Save to JSON
    save_to_json(all_videos, "youtube.json")
    
    return all_videos


def get_video_transcript(video_id):     
    """
    Get video transcript (no API quota impact)
    
    Returns:
        dict: Raw transcript data
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try manual English first
        try:
            transcript = transcript_list.find_transcript(['en'])
            transcript_data = transcript.fetch()
            full_text = " ".join([entry['text'] for entry in transcript_data])
            
            return {
                "available": True,
                "language": "en",
                "text": full_text[:5000],
                "word_count": len(full_text.split()),
                "auto_generated": False
            }
        except:
            pass
        
        # Try auto-generated
        try:
            transcript = transcript_list.find_generated_transcript(['en'])
            transcript_data = transcript.fetch()
            full_text = " ".join([entry['text'] for entry in transcript_data])
            
            return {
                "available": True,
                "language": "en",
                "text": full_text[:5000],
                "word_count": len(full_text.split()),
                "auto_generated": True
            }
        except:
            pass
        
        return {"available": False, "reason": "No English transcript"}
        
    except Exception as e:
        return {"available": False, "error": str(e)}