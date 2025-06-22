import os
import io
import numpy as np
import base64
import wave
from pathlib import Path
import google.generativeai as genai
from google.auth import default
from google.oauth2 import service_account
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from app.core.config import settings
import json
import asyncio
from datetime import datetime

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


# WebSocket message models
class WSMessage(BaseModel):
    type: str  # "audio", "config", "ping", "pong"
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


class WSAudioMessage(BaseModel):
    audio_data: str  # Base64 encoded audio chunk
    sample_rate: int = 16000
    channels: int = 1


class WSResponseMessage(BaseModel):
    type: str  # "transcript", "analysis", "recommendation", "error"
    data: Dict[str, Any]
    timestamp: str


# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_states: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_states[client_id] = {
            "conversation_history": [],
            "audio_buffer": b"",
            "last_activity": datetime.now(),
            "audio_detected": False,
            "silence_chunks": 0,
            "metrics": {
                "sentiment": 0,
                "empathy": 5,
                "resolution": 0,
                "escalation": 0
            }
        }
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.connection_states[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: WSResponseMessage):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_json(message.dict())
    
    def get_state(self, client_id: str) -> Dict[str, Any]:
        return self.connection_states.get(client_id, {})
    
    def update_state(self, client_id: str, key: str, value: Any):
        if client_id in self.connection_states:
            self.connection_states[client_id][key] = value


manager = ConnectionManager()


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


# WebSocket helper functions
async def process_audio_chunk(client_id: str, audio_data: str, sample_rate: int, channels: int):
    """Process audio chunk and return transcript if available."""
    try:
        state = manager.get_state(client_id)
        
        # Decode audio chunk
        audio_bytes = base64.b64decode(audio_data)
        
        # Convert to numpy array to check audio level
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_level = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0
        
        # Get or initialize silence tracking
        silence_threshold = 800  # Audio level threshold for silence
        silence_chunks = state.get("silence_chunks", 0)
        audio_detected = state.get("audio_detected", False)
        
        # Append to buffer
        current_buffer = state.get("audio_buffer", b"")
        updated_buffer = current_buffer + audio_bytes
        
        # Update buffer in state
        manager.update_state(client_id, "audio_buffer", updated_buffer)
        
        # Detect speech or silence
        if audio_level > 1500 and not audio_detected:
            # Speech detected
            manager.update_state(client_id, "audio_detected", True)
            manager.update_state(client_id, "silence_chunks", 0)
            logger.debug(f"Audio detected for client {client_id}, level: {audio_level}")
        elif audio_level <= silence_threshold and audio_detected:
            # Count silence chunks
            silence_chunks += 1
            manager.update_state(client_id, "silence_chunks", silence_chunks)
            
            # Check if we have enough silence (approx 1 second of silence)
            # With 100ms chunks, 10 chunks = 1 second
            if silence_chunks >= 10:
                # Silence detected after speech, process the buffer
                buffer_duration = len(updated_buffer) / (sample_rate * channels * 2)  # 16-bit audio
                
                # Only process if we have meaningful audio (at least 0.5 seconds)
                if buffer_duration >= 0.5:
                    # Process the audio
                    transcript = process_audio(
                        base64.b64encode(updated_buffer).decode('utf-8'),
                        sample_rate,
                        channels
                    )
                    
                    # Clear the buffer and reset states
                    manager.update_state(client_id, "audio_buffer", b"")
                    manager.update_state(client_id, "audio_detected", False)
                    manager.update_state(client_id, "silence_chunks", 0)
                    
                    # Update conversation history
                    history = state.get("conversation_history", [])
                    if transcript and len(transcript.strip()) > 0:
                        history.append(transcript)
                        manager.update_state(client_id, "conversation_history", history)
                        
                        return transcript
                else:
                    # Not enough audio, just reset
                    manager.update_state(client_id, "audio_buffer", b"")
                    manager.update_state(client_id, "audio_detected", False)
                    manager.update_state(client_id, "silence_chunks", 0)
        
        # Also check for maximum buffer size (10 seconds) to prevent overflow
        buffer_duration = len(updated_buffer) / (sample_rate * channels * 2)
        if buffer_duration >= 10.0:
            # Force process if buffer is too large
            transcript = process_audio(
                base64.b64encode(updated_buffer).decode('utf-8'),
                sample_rate,
                channels
            )
            
            # Clear the buffer
            manager.update_state(client_id, "audio_buffer", b"")
            manager.update_state(client_id, "audio_detected", False)
            manager.update_state(client_id, "silence_chunks", 0)
            
            # Update conversation history
            history = state.get("conversation_history", [])
            if transcript and len(transcript.strip()) > 0:
                history.append(transcript)
                manager.update_state(client_id, "conversation_history", history)
                
                return transcript
    
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")
        return None
    
    return None


def should_trigger_coaching(metrics: Dict[str, float], conversation_length: int) -> tuple[bool, str]:
    """Determine if coaching should be triggered based on metrics."""
    # Trigger 1: Low empathy score
    if metrics["empathy"] < 5:
        return True, "Low empathy detected"
    
    # Trigger 2: Negative sentiment
    if metrics["sentiment"] < -0.2:
        return True, "Negative sentiment detected"
    
    # Trigger 3: Rising escalation risk
    if metrics["escalation"] > 30:
        return True, "Escalation risk increasing"
    
    # Trigger 4: Stalled resolution progress
    if metrics["resolution"] < 40 and conversation_length > 6:
        return True, "Resolution progress stalled"
    
    return False, ""


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time audio streaming and coaching."""
    await manager.connect(websocket, client_id)
    
    try:
        # Send initial connection confirmation
        await manager.send_message(
            client_id,
            WSResponseMessage(
                type="connected",
                data={"client_id": client_id, "status": "connected"},
                timestamp=datetime.now().isoformat()
            )
        )
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = WSMessage(**data)
            
            # Update last activity
            manager.update_state(client_id, "last_activity", datetime.now())
            
            # Handle different message types
            if message.type == "audio":
                # Process audio data
                audio_msg = WSAudioMessage(**message.data)
                transcript = await process_audio_chunk(
                    client_id,
                    audio_msg.audio_data,
                    audio_msg.sample_rate,
                    audio_msg.channels
                )
                
                if transcript:
                    # Send transcript to client
                    await manager.send_message(
                        client_id,
                        WSResponseMessage(
                            type="transcript",
                            data={"transcript": transcript},
                            timestamp=datetime.now().isoformat()
                        )
                    )
                    
                    # Get current state
                    state = manager.get_state(client_id)
                    history = state.get("conversation_history", [])
                    
                    # Analyze conversation
                    try:
                        metrics = analyze_conversation(transcript, history)
                        manager.update_state(client_id, "metrics", metrics)
                        
                        # Send analysis to client
                        await manager.send_message(
                            client_id,
                            WSResponseMessage(
                                type="analysis",
                                data=metrics,
                                timestamp=datetime.now().isoformat()
                            )
                        )
                        
                        # Check if coaching should be triggered
                        should_coach, reason = should_trigger_coaching(metrics, len(history))
                        
                        if should_coach:
                            # Get recommendations
                            recommendations = get_recommendations(transcript, history)
                            
                            # Send recommendations to client
                            await manager.send_message(
                                client_id,
                                WSResponseMessage(
                                    type="recommendation",
                                    data={
                                        "recommendations": recommendations,
                                        "trigger_reason": reason
                                    },
                                    timestamp=datetime.now().isoformat()
                                )
                            )
                    
                    except Exception as e:
                        logger.error(f"Error in analysis/coaching: {e}")
                        await manager.send_message(
                            client_id,
                            WSResponseMessage(
                                type="error",
                                data={"error": f"Analysis error: {str(e)}"},
                                timestamp=datetime.now().isoformat()
                            )
                        )
            
            elif message.type == "ping":
                # Respond to ping with pong
                await manager.send_message(
                    client_id,
                    WSResponseMessage(
                        type="pong",
                        data={},
                        timestamp=datetime.now().isoformat()
                    )
                )
            
            elif message.type == "config":
                # Handle configuration updates
                if message.data:
                    for key, value in message.data.items():
                        manager.update_state(client_id, key, value)
                    
                    await manager.send_message(
                        client_id,
                        WSResponseMessage(
                            type="config_updated",
                            data=message.data,
                            timestamp=datetime.now().isoformat()
                        )
                    )
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")
    
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await manager.send_message(
            client_id,
            WSResponseMessage(
                type="error",
                data={"error": str(e)},
                timestamp=datetime.now().isoformat()
            )
        )
        manager.disconnect(client_id) 