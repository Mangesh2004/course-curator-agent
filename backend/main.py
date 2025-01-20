from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain_groq import ChatGroq
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptAvailable
)
from dotenv import load_dotenv
import os
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize YouTube API client
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Request and Response Models
class LearningPathRequest(BaseModel):
    course_name: str
    current_skills: List[str]
    max_results: int = 5

class VideoRecommendation(BaseModel):
    title: str
    url: str
    description: str
    relevance_score: float

class LearningPathResponse(BaseModel):
    status: str
    message: str
    recommendations: List[VideoRecommendation]

def get_english_transcript(video_id: str) -> list:
    """Attempt to fetch English transcript with fallback to regional English variants."""
    logger.info(f"Attempting to fetch transcript for video ID: {video_id}")
    try:
        english_codes = ['en', 'en-IN', 'en-US', 'en-GB']
        for lang_code in english_codes:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang_code])
                logger.info(f"Successfully retrieved transcript in {lang_code}")
                return transcript
            except CouldNotRetrieveTranscript:
                continue
        raise NoTranscriptAvailable()
    except Exception as e:
        logger.error(f"Error fetching transcript: {str(e)}")
        raise

@app.get("/")
async def read_root():
    return {"status": "active", "message": "Learning Pathway Generator API is running"}

@app.post("/api/generate-learning-path", response_model=LearningPathResponse)
async def generate_learning_path(request: LearningPathRequest):
    try:
        logger.info(f"Processing request for course: {request.course_name}")

        # Step 1: Get recommended topics from LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=2000
        )

        system_prompt = "You are an AI that suggests topics to learn next based on existing skills. " \
                        "Return a list of 5 topics to learn in the field of education."

        user_prompt = f"User's current skills: {', '.join(request.current_skills)}\n\n" \
                      "Return a list of 5 skills/concepts that the user should focus on next."

        messages = [("system", system_prompt), ("user", user_prompt)]
        ai_message = llm.invoke(messages)
        response_content = ai_message.content.strip()

        try:
            # Parse the topics from LLM response
            topics = json.loads(response_content)
            if not isinstance(topics, list):
                raise ValueError("LLM response is not a list of topics.")
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            raise HTTPException(status_code=500, detail="Error generating learning topics.")

        # Step 2: Fetch videos for each recommended topic
        all_video_scores = []
        for topic in topics:
            logger.info(f"Processing topic: {topic}")

            # Fetch 15 videos for each topic
            search_response = youtube.search().list(
                q=topic,
                part="snippet",
                type="video",
                maxResults=request.max_results,
                order="relevance"
            ).execute()

            videos = []
            for item in search_response.get("items", []):
                video_id = item["id"]["videoId"]
                snippet = item["snippet"]
                videos.append({
                    "id": video_id,
                    "title": snippet["title"],
                    "description": snippet["description"],
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                })

            logger.info(f"Found {len(videos)} videos for topic: {topic}")

            # Step 3: Evaluate the relevance of each video for the current topic
            for video in videos:
                try:
                    transcript = get_english_transcript(video["id"])
                    transcript_text = " ".join([entry["text"] for entry in transcript])
                    logger.info(f"Transcript length: {len(transcript_text)} characters")

                    # Create prompts for LLaMA analysis
                    system_prompt = (
                        "You are an AI that evaluates YouTube video content. "
                        "Analyze the transcript and return ONLY a JSON object with a 'relevance_score' "
                        "property (float between 0 and 1) indicating how well the content matches "
                        "the learning goals. Return nothing else besides the JSON object."
                    )

                    user_prompt = (
                        f"Learning goal: {topic}\n"
                        f"Video Transcript: {transcript_text[:4000]}\n\n"
                        "Respond with only a JSON object containing the relevance score. Example:\n"
                        '{"relevance_score": 0.85}'
                    )

                    # Generate LLaMA response
                    messages = [("system", system_prompt), ("user", user_prompt)]
                    ai_message = llm.invoke(messages)
                    response_content = ai_message.content.strip()

                    try:
                        response_data = json.loads(response_content)
                        relevance_score = response_data.get("relevance_score", 0)
                        logger.info(f"Relevance score: {relevance_score}")

                        all_video_scores.append({
                            "topic": topic,
                            "video": video,
                            "relevance_score": relevance_score
                        })
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse LLM response as JSON: {e}")
                        continue

                except Exception as e:
                    logger.error(f"Error processing video {video['id']}: {e}")
                    continue

        # Step 4: Sort videos by relevance score and select the best for each topic
        best_video_per_topic = {}
        for video_score in all_video_scores:
            topic = video_score["topic"]
            video = video_score["video"]
            relevance_score = video_score["relevance_score"]

            # Keep only the best video for each topic
            if topic not in best_video_per_topic or best_video_per_topic[topic]["relevance_score"] < relevance_score:
                best_video_per_topic[topic] = {
                    "video": video,
                    "relevance_score": relevance_score
                }

        # Step 5: Prepare recommendations for the response
        recommendations = []
        for topic, best_video in best_video_per_topic.items():
            recommendations.append(VideoRecommendation(
                title=best_video["video"]["title"],
                url=best_video["video"]["url"],
                description=best_video["video"]["description"],
                relevance_score=best_video["relevance_score"]
            ))

        logger.info(f"Final recommendations count: {len(recommendations)}")

        # Return recommendations
        if not recommendations:
            logger.warning("No recommendations generated!")
            return LearningPathResponse(
                status="success",
                message="No suitable videos found matching your criteria. Try adjusting your search terms.",
                recommendations=[]
            )

        return LearningPathResponse(
            status="success",
            message="Generated learning pathway recommendations",
            recommendations=recommendations
        )

    except Exception as e:
        logger.error(f"Fatal error in generate_learning_path: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
