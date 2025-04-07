import os
import re
import json
import requests
from datetime import datetime, timedelta, timezone
from flask import Flask, request, render_template_string
from dotenv import load_dotenv

# ---------- Load Environment Variables ----------
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

if not YOUTUBE_API_KEY:
    raise Exception("YouTube API key not found. Please set it in the .env file.")

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
        content = response.content.strip()
        # Extract the JSON substring using regex
        json_match = re.search(r'\{.*\}', content)
        if not json_match:
            raise ValueError("No JSON found in LLM response")
        json_str = json_match.group()
        result = json.loads(json_str)
        score = float(result.get("score", 0))
        return score
    except Exception as exc:
        print("Error with Gemini LLM:", exc)
        return 0

def get_videos_from_youtube(query, max_results=25, published_after=None, video_duration="any"):
    """
    Search for videos on YouTube using the YouTube Data API.
    Filters videos published after the given published_after datetime.
    Optionally filters by video duration category: any, short, medium, long.
    If published_after is None, defaults to the last 14 days.
    """
    if published_after is None:
        published_after = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat().replace("+00:00", "Z")
    params = {
        "part": "snippet",
        "q": query,
        "maxResults": max_results,
        "type": "video",
        "publishedAfter": published_after,
        "videoDuration": video_duration,
        "key": YOUTUBE_API_KEY
    }
    search_url = "https://www.googleapis.com/youtube/v3/search"
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

def filter_videos_by_duration(video_details, min_minutes, max_minutes):
    """
    Filters out videos that do not have a duration between min_minutes and max_minutes.
    """
    filtered = {}
    for vid, details in video_details.items():
        duration_minutes = convert_iso_duration_to_minutes(details["duration"])
        if min_minutes <= duration_minutes <= max_minutes:
            filtered[vid] = details
    return filtered

def process_query(query, days=14, video_duration="any"):
    """
    Processes the search query and returns details of the best video.
    The video_duration parameter is passed directly to the YouTube API to filter videos.
    The duration bounds are determined based on the video_duration filter:
      - short: less than 4 minutes
      - medium: between 4 and 20 minutes
      - long: more than 20 minutes
      - any: no additional filtering based on duration
    """
    try:
        published_after = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat().replace("+00:00", "Z")
        videos = get_videos_from_youtube(query, max_results=20, published_after=published_after, video_duration=video_duration)
        if not videos:
            return {"error": "No videos found."}
    except Exception as e:
        return {"error": f"Error during YouTube search: {e}"}

    video_ids = [video["id"] for video in videos]
    try:
        details = get_video_details(video_ids)
    except Exception as e:
        return {"error": f"Error fetching video details: {e}"}

    # Determine duration bounds based on video_duration filter
    if video_duration == "short":
        min_duration = 0
        max_duration = 4
    elif video_duration == "medium":
        min_duration = 4
        max_duration = 20
    elif video_duration == "long":
        min_duration = 20
        max_duration = float("inf")
    else:  # any
        min_duration = 0
        max_duration = float("inf")

    filtered_videos = filter_videos_by_duration(details, min_duration, max_duration)
    if not filtered_videos:
        return {"error": f"No videos found with the selected duration filter."}

    video_scores = {}
    for vid, info in filtered_videos.items():
        score = call_llm_to_score_title(query, info["title"])
        video_scores[vid] = score

    best_video_id = max(video_scores, key=video_scores.get)
    best_video = filtered_videos[best_video_id]
    best_video["url"] = f"https://www.youtube.com/watch?v={best_video_id}"

    return best_video

# ---------- FLASK APP WITH IMPROVED FRONTEND ----------

app = Flask(__name__)

home_page = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>YouTube Video Finder</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body>
    <div class="container py-5">
      <h1 class="text-center mb-4">YouTube Video Finder</h1>
      <div class="row justify-content-center">
        <div class="col-md-8">
          <form method="POST" class="mb-4">
            <div class="mb-3">
              <label for="query" class="form-label">Enter your search query (Hindi/English):</label>
              <input type="text" class="form-control" id="query" name="query" required>
            </div>
            <div class="mb-3">
              <label for="days" class="form-label">Previous Days to Search (e.g., 14):</label>
              <input type="number" min="1" class="form-control" id="days" name="days" value="14" required>
            </div>
            <div class="mb-3">
              <label for="video_duration" class="form-label">Video Duration Filter:</label>
              <select class="form-select" id="video_duration" name="video_duration">
                <option value="any">Any</option>
                <option value="short">Short (less than 4 minutes)</option>
                <option value="medium">Medium (4-20 minutes)</option>
                <option value="long">Long (more than 20 minutes)</option>
              </select>
            </div>
            <button type="submit" class="btn btn-primary w-100">Search</button>
          </form>
          {% if error %}
            <div class="alert alert-danger" role="alert">
              {{ error }}
            </div>
          {% endif %}
          {% if video %}
            <div class="card">
              <div class="card-body">
                <h4 class="card-title">Best Video Recommendation</h4>
                <p class="card-text"><strong>Title:</strong> {{ video.title }}</p>
                <p class="card-text"><strong>Channel:</strong> {{ video.channelTitle }}</p>
                <p class="card-text"><strong>Published At:</strong> {{ video.publishedAt }}</p>
                <p class="card-text"><strong>Duration:</strong> {{ video.duration }}</p>
                <a href="{{ video.url }}" target="_blank" class="btn btn-success">Watch Video</a>
              </div>
            </div>
          {% endif %}
        </div>
      </div>
    </div>
    <!-- Bootstrap JS Bundle -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if not query:
            return render_template_string(home_page, error="No query provided.")
        try:
            days = int(request.form.get("days", 14))
            video_duration = request.form.get("video_duration", "any")
        except ValueError:
            return render_template_string(home_page, error="Days value must be a number.")
        result = process_query(query, days, video_duration)
        if "error" in result:
            return render_template_string(home_page, error=result["error"])
        return render_template_string(home_page, video=result)
    return render_template_string(home_page)

if __name__ == "__main__":
    app.run(debug=True)
