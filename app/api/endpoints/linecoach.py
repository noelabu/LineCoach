import os
import io
import numpy as np
import base64
import wave
from pathlib import Path
import google.generativeai as genai
from google.auth import default
from google.oauth2 import service_account
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from app.core.config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize Gemini API
try:
    # Use API key authentication if available
    if settings.GEMINI_API_KEY:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        logger.info("Successfully initialized Gemini API with API key")
    else:
        # Try to use service account credentials if available
        credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            genai.configure(credentials=credentials)
            logger.info("Successfully initialized Gemini API with service account credentials")
        else:
            logger.error("No Gemini API credentials available")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {e}")


# Pydantic models for request and response
class AudioRequest(BaseModel):
    audio_data: str  # Base64 encoded audio data
    sample_rate: int = 16000
    channels: int = 1
    content_type: str = "audio/wav"  # MIME type of the audio


class TranscriptionResponse(BaseModel):
    transcript: str


class CoachingRequest(BaseModel):
    transcript: str
    conversation_history: List[str] = []


class AnalysisResponse(BaseModel):
    sentiment: float
    empathy: float
    resolution: float
    escalation: float


class RecommendationResponse(BaseModel):
    recommendations: str
    analysis: Optional[AnalysisResponse] = None


# Helper functions
def process_audio(audio_data_base64: str, sample_rate: int, channels: int) -> str:
    """Process audio data and return transcript."""
    try:
        # Decode base64 audio data
        audio_data = base64.b64decode(audio_data_base64)
        
        # Create in-memory WAV file for Gemini API
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit audio = 2 bytes
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        
        # Get the WAV data
        wav_data = wav_buffer.getvalue()
        wav_b64 = base64.b64encode(wav_data).decode('utf-8')
        
        # Create prompt for Gemini
        prompt = "Please transcribe the following audio. Return only the transcribed text without any additional commentary."
        
        # Initialize Gemini model
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        # Generate transcription
        response = model.generate_content([
            prompt,
            {
                "mime_type": "audio/wav",
                "data": wav_b64
            }
        ])
        
        # Get the transcription from the response
        transcript = response.text.strip()
        return transcript
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


def analyze_conversation(transcript: str, conversation_history: List[str]) -> Dict[str, float]:
    """Analyze the conversation for sentiment and other metrics."""
    try:
        # Create prompt for analysis
        prompt = f"""
        Analyze the following customer service conversation transcript:
        
        "{transcript}"
        
        Previous conversation context: "{' '.join(conversation_history[-3:] if conversation_history else [])}"
        
        Please provide the following metrics:
        1. Sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, and 1 is very positive)
        2. Empathy score (0 to 10, where 10 is extremely empathetic)
        3. Resolution progress (0% to 100%)
        4. Escalation risk (0% to 100%)
        
        Format your response as:
        Sentiment: [score]
        Empathy: [score]/10
        Resolution: [score]%
        Escalation Risk: [score]%
        """
        
        # Initialize Gemini model
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        # Generate analysis
        response = model.generate_content(prompt)
        analysis = response.text
        
        # Parse the analysis to extract metrics
        metrics = {
            "sentiment": 0,
            "empathy": 0,
            "resolution": 0,
            "escalation": 0
        }
        
        lines = analysis.strip().split('\n')
        for line in lines:
            if line.startswith('Sentiment:'):
                metrics["sentiment"] = float(line.split(':')[1].strip())
            elif line.startswith('Empathy:'):
                metrics["empathy"] = float(line.split(':')[1].split('/')[0].strip())
            elif line.startswith('Resolution:'):
                metrics["resolution"] = float(line.split(':')[1].strip('%'))
            elif line.startswith('Escalation Risk:'):
                metrics["escalation"] = float(line.split(':')[1].strip('%'))
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error analyzing conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing conversation: {str(e)}")


def get_recommendations(transcript: str, conversation_history: List[str]) -> str:
    """Get recommendations based on the conversation."""
    try:
        # Create prompt for Gemini
        prompt = f"""
        You're an assistant helping a customer service representative during a call.
        Your job is to listen and suggest what they should say next.

        Current transcript: "{transcript}"

        Previous conversation context: "{' '.join(conversation_history[-3:] if conversation_history else [])}"

        Role: Customer Service Representative

        Please provide 2-3 actionable suggestions on what to say next. Keep them concise and relevant.

        Examples:
        1. Ask about specific error messages they're seeing.
        2. Offer to reset their account access.
        3. Confirm their contact information for verification.

        What would you recommend saying next?
        """
        
        # Initialize Gemini model
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        # Generate recommendations
        response = model.generate_content(prompt)
        recommendations = response.text
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


# API Endpoints
@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: AudioRequest):
    """Transcribe audio data."""
    try:
        transcript = process_audio(
            request.audio_data, 
            request.sample_rate, 
            request.channels
        )
        return {"transcript": transcript}
    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: CoachingRequest):
    """Analyze conversation for sentiment and other metrics."""
    try:
        metrics = analyze_conversation(request.transcript, request.conversation_history)
        return AnalysisResponse(**metrics)
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/coach", response_model=RecommendationResponse)
async def coach(request: CoachingRequest, background_tasks: BackgroundTasks):
    """Get coaching recommendations based on the conversation."""
    try:
        # Get recommendations
        recommendations = get_recommendations(request.transcript, request.conversation_history)
        
        # Analyze in the background to avoid blocking
        background_tasks.add_task(
            analyze_conversation, 
            request.transcript, 
            request.conversation_history
        )
        
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error in coach endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/full_coaching", response_model=RecommendationResponse)
async def full_coaching(request: CoachingRequest):
    """Get both recommendations and analysis in a single call."""
    try:
        # Get recommendations
        recommendations = get_recommendations(request.transcript, request.conversation_history)
        
        # Get analysis
        metrics = analyze_conversation(request.transcript, request.conversation_history)
        
        return {
            "recommendations": recommendations,
            "analysis": AnalysisResponse(**metrics)
        }
    except Exception as e:
        logger.error(f"Error in full_coaching endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 