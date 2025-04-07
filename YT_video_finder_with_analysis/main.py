import os
import re
import json
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# ---------- Load Environment Variables ----------
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

if not YOUTUBE_API_KEY:
    raise Exception("YouTube API key not found. Please set it in the .env file.")

# If using Gemini, initialize the LLM from ChatGoogleGenerativeAI.

if not GOOGLE_API_KEY:
    raise Exception("Google API key not found. Please set it in the .env file.")
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

gemini_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    verbose=True,
    temperature=0.75,
    top_p=0.8,
    top_k=45,
    timeout=300,
    max_output_tokens=5000,
    max_retries=5,
    google_api_key=GOOGLE_API_KEY,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }
)

# ---------- HELPER FUNCTIONS ----------

def convert_iso_duration_to_minutes(duration):
    """
    Converts ISO 8601 duration (e.g., PT5M30S) to total minutes.
    """
    pattern = re.compile(
        r'PT(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?'
    )
    match = pattern.match(duration)
    if not match:
        return 0
    parts = match.groupdict()
    hours = int(parts.get('hours') or 0)
    minutes = int(parts.get('minutes') or 0)
    seconds = int(parts.get('seconds') or 0)
    total_minutes = hours * 60 + minutes + seconds / 60.0
    return total_minutes

def call_llm_to_score_title(query, title):
    """
    Scores a video title for relevance given a query.
    Uses Gemini LLM via chat method.
    """
    prompt = (
        f"You are a scoring system. Rate the relevance of this video title for the search query.\n"
        f"Query: '{query}'\n"
        f"Video Title: '{title}'\n"
        f"Respond only with a JSON object in this exact format: {{\"score\": number}}\n"
        f"where number is between 1 and 100 based on relevance."
    )
    try:
        response = gemini_llm.invoke(prompt)
        # Clean and parse the response content
        content = response.content.strip()
        if not content.startswith('{'): # Handle cases where LLM adds extra text
            content = content[content.find('{'):content.find('}')+1]
        result = json.loads(content)
        score = result.get("score", 0)
        return score
    except Exception as exc:
        print("Error with Gemini LLM:", exc)
        return 0

def get_videos_from_youtube(query, max_results=20):
    """
    Search for videos on YouTube using the YouTube Data API.
    Filters videos published in the last 14 days.
    """
    search_url = "https://www.googleapis.com/youtube/v3/search"
    published_after = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat().replace("+00:00", "Z")
    params = {
        "part": "snippet",
        "q": query,
        "maxResults": max_results,
        "type": "video",
        "publishedAfter": published_after,
        "key": YOUTUBE_API_KEY
    }
    
    response = requests.get(search_url, params=params)
    if response.status_code != 200:
        raise Exception(f"YouTube API error: {response.status_code} {response.text}")
    
    results = response.json().get('items', [])
    videos = []
    for item in results:
        video_id = item['id']['videoId']
        snippet = item['snippet']
        videos.append({
            "id": video_id,
            "title": snippet['title'],
            "publishedAt": snippet['publishedAt']
        })
    return videos

def get_video_details(video_ids):
    """
    Retrieve video details such as contentDetails (duration) for a list of video IDs.
    """
    details_url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "contentDetails,snippet",
        "id": ",".join(video_ids),
        "key": YOUTUBE_API_KEY
    }
    response = requests.get(details_url, params=params)
    if response.status_code != 200:
        raise Exception(f"YouTube API error: {response.status_code} {response.text}")
    
    details = response.json().get('items', [])
    video_details = {}
    for video in details:
        video_id = video['id']
        duration = video['contentDetails']['duration']
        video_details[video_id] = {
            "title": video['snippet']['title'],
            "duration": duration,
            "publishedAt": video['snippet']['publishedAt'],
            "channelTitle": video['snippet'].get('channelTitle', '')
        }
    return video_details

def filter_videos_by_duration(video_details, min_minutes=4, max_minutes=20):
    """
    Filters out videos that do not have a duration between min_minutes and max_minutes.
    """
    filtered = {}
    for vid, details in video_details.items():
        duration_minutes = convert_iso_duration_to_minutes(details["duration"])
        if min_minutes <= duration_minutes <= max_minutes:
            filtered[vid] = details
    return filtered

# ---------- MAIN WORKFLOW ----------

def main():
    query = input("Enter your search query (Hindi/English): ").strip()
    if not query:
        print("No query provided.")
        return
    
    print("Searching YouTube for top 20 relevant videos...")
    
    try:
        videos = get_videos_from_youtube(query, max_results=20)
        if not videos:
            print("No videos found.")
            return
    except Exception as e:
        print("Error during YouTube search:", e)
        return

    # Print links to all top 20 videos
    print("\nTop 20 Video Links:")
    for video in videos:
        video_url = f"https://www.youtube.com/watch?v={video['id']}"
        print(video_url)
    
    video_ids = [video["id"] for video in videos]
    
    try:
        details = get_video_details(video_ids)
    except Exception as e:
        print("Error fetching video details:", e)
        return
    
    filtered_videos = filter_videos_by_duration(details)
    if not filtered_videos:
        print("No videos found with duration between 4 and 20 minutes.")
        return
    
    print(f"\nAnalyzing video titles for relevance using {LLM_MODEL}...")
    video_scores = {}
    for vid, info in filtered_videos.items():
        score = call_llm_to_score_title(query, info["title"])
        video_scores[vid] = score
    
    best_video_id = max(video_scores, key=video_scores.get)
    best_video = filtered_videos[best_video_id]
    video_url = f"https://www.youtube.com/watch?v={best_video_id}"
    
    print("\nBest Video Recommendation:")
    print("Title       :", best_video["title"])
    print("Channel     :", best_video["channelTitle"])
    print("Published At:", best_video["publishedAt"])
    print("Duration    :", best_video["duration"])
    print("URL         :", video_url)
    
if __name__ == "__main__":
    main()
