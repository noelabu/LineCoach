import os
import io
import numpy as np
import base64
import wave
import re
from pathlib import Path
from google.auth import default
from google.oauth2 import service_account
from google.cloud import speech
import google.generativeai as genai
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
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

# Initialize Google Cloud Speech-to-Text client
speech_client = None
try:
    # Try to initialize with credentials
    credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
    if credentials_path:
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        speech_client = speech.SpeechClient(credentials=credentials)
        logger.info("Successfully initialized Google Cloud Speech-to-Text client")
    else:
        # Try default credentials
        speech_client = speech.SpeechClient()
        logger.info("Successfully initialized Google Cloud Speech-to-Text client with default credentials")
except Exception as e:
    logger.error(f"Failed to initialize Google Cloud Speech-to-Text: {e}")

# Initialize Gemini API for analysis and recommendations
try:
    if settings.GEMINI_API_KEY:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        logger.info("Successfully initialized Gemini API with API key")
    else:
        credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            genai.configure(credentials=credentials)
            logger.info("Successfully initialized Gemini API with service account credentials")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {e}")


# WebSocket message models
class WSMessage(BaseModel):
    type: str  # "audio", "config", "ping", "pong", "stop_audio"
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


class WSAudioMessage(BaseModel):
    audio_data: str  # Base64 encoded audio chunk
    sample_rate: int = 16000
    channels: int = 1


class WSResponseMessage(BaseModel):
    type: str  # "transcript", "analysis", "recommendation", "error", "connected", "pong"
    data: Dict[str, Any]
    timestamp: str


# Connection Manager for WebSocket connections
class GCPConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_states: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_states[client_id] = {
            "conversation_history": [],
            "last_activity": datetime.now(),
            "metrics": {
                "sentiment": 0,
                "empathy": 5,
                "resolution": 0,
                "escalation": 0
            }
        }
        logger.info(f"Client {client_id} connected to GCP endpoint")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.connection_states[client_id]
            logger.info(f"Client {client_id} disconnected from GCP endpoint")
    
    async def send_message(self, client_id: str, message: WSResponseMessage):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message.dict())
            except Exception as e:
                logger.warning(f"Failed to send message to client {client_id}: {e}")
                # Remove the disconnected client
                self.disconnect(client_id)
    
    def get_state(self, client_id: str) -> Dict[str, Any]:
        return self.connection_states.get(client_id, {})
    
    def update_state(self, client_id: str, key: str, value: Any):
        if client_id in self.connection_states:
            self.connection_states[client_id][key] = value


manager = GCPConnectionManager()


# Helper functions for analysis and recommendations
def analyze_conversation(transcript: str, conversation_history: List[Any]) -> Dict[str, float]:
    """Analyze the conversation for sentiment and other metrics."""
    try:
        # Format conversation history
        if conversation_history:
            formatted_history = []
            for item in conversation_history[-3:]:
                if isinstance(item, dict):
                    speaker = item.get('speaker', 'unknown')
                    text = item.get('text', '')
                    formatted_history.append(f"{speaker}: {text}")
                else:
                    formatted_history.append(str(item))
            context = " | ".join(formatted_history)
        else:
            context = ""
        
        # Create prompt for analysis
        prompt = f"""
        Analyze the following customer service conversation transcript:
        
        "{transcript}"
        
        Previous conversation context: "{context}"
        
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
            try:
                if line.startswith('Sentiment:'):
                    value = line.split(':')[1].strip()
                    match = re.search(r'-?\d+\.?\d*', value)
                    metrics["sentiment"] = float(match.group()) if match else 0
                elif line.startswith('Empathy:'):
                    value = line.split(':')[1].split('/')[0].strip()
                    match = re.search(r'\d+\.?\d*', value)
                    metrics["empathy"] = float(match.group()) if match else 5
                elif line.startswith('Resolution:'):
                    value = line.split(':')[1].strip()
                    match = re.search(r'\d+\.?\d*', value.replace('%', ''))
                    metrics["resolution"] = float(match.group()) if match else 0
                elif line.startswith('Escalation Risk:'):
                    value = line.split(':')[1].strip()
                    match = re.search(r'\d+\.?\d*', value.replace('%', ''))
                    metrics["escalation"] = float(match.group()) if match else 0
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to parse metric from line '{line}': {e}")
                continue
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error analyzing conversation: {e}")
        raise


def get_recommendations(transcript: str, conversation_history: List[Any]) -> str:
    """Get recommendations based on the conversation."""
    try:
        # Format conversation history
        if conversation_history:
            formatted_history = []
            for item in conversation_history[-3:]:
                if isinstance(item, dict):
                    speaker = item.get('speaker', 'unknown')
                    text = item.get('text', '')
                    formatted_history.append(f"{speaker}: {text}")
                else:
                    formatted_history.append(str(item))
            context = " | ".join(formatted_history)
        else:
            context = ""
        
        # Create prompt for Gemini
        prompt = f"""
        You're an assistant helping a customer service representative during a call.
        Your job is to listen and suggest what they should say next.

        Current transcript: "{transcript}"

        Previous conversation context: "{context}"

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
        raise


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


# WebSocket endpoint for Google Cloud Speech-to-Text
@router.websocket("/ws/{client_id}")
async def websocket_gcp_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time audio streaming using Google Cloud Speech-to-Text."""
    if not speech_client:
        await websocket.close(code=1003, reason="Google Cloud Speech-to-Text not configured")
        return
    
    await manager.connect(websocket, client_id)
    
    try:
        # Send initial connection confirmation
        await manager.send_message(
            client_id,
            WSResponseMessage(
                type="connected",
                data={
                    "client_id": client_id, 
                    "status": "connected",
                    "engine": "google-cloud-speech"
                },
                timestamp=datetime.now().isoformat()
            )
        )
        
        # Configure speech recognition
        diarization_config = speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=1,
            max_speaker_count=2,
        )
        
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            diarization_config=diarization_config,
            enable_automatic_punctuation=True,
            use_enhanced=True,
            model="phone_call",  # Optimized for phone calls
            enable_word_time_offsets=True
        )
        
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
            single_utterance=False
        )
        
        
        # Store audio chunks and speaker state for turn detection
        audio_buffer = []
        current_speaker = None
        last_speech_time = None
        silence_threshold = 1.5  # seconds of silence before considering speaker turn
        
        async def process_audio_batch(force_process=False):
            """Process accumulated audio when speaker turns are detected."""
            nonlocal current_speaker, last_speech_time
            
            if not audio_buffer:
                return
            
            try:
                # Combine all audio chunks
                combined_audio = b''.join(audio_buffer)
                audio_buffer.clear()
                
                # Create recognition request
                audio = speech.RecognitionAudio(content=combined_audio)
                
                # Perform synchronous recognition
                response = speech_client.recognize(config=config, audio=audio)
                
                # Process results
                for result in response.results:
                    if not result.alternatives:
                        continue
                    
                    alternative = result.alternatives[0]
                    transcript = alternative.transcript
                    
                    # Determine speaker from diarization
                    speaker = "unknown"
                    if alternative.words:
                        first_word = alternative.words[0]
                        if hasattr(first_word, 'speaker_tag') and first_word.speaker_tag:
                            speaker = f"speaker_{first_word.speaker_tag}"
                    
                    # Update current speaker for turn detection
                    current_speaker = speaker
                    last_speech_time = datetime.now()
                    
                    # Send transcript
                    await manager.send_message(
                        client_id,
                        WSResponseMessage(
                            type="transcript",
                            data={
                                "transcript": transcript,
                                "speaker": speaker,
                                "is_final": True,
                                "confidence": alternative.confidence
                            },
                            timestamp=datetime.now().isoformat()
                        )
                    )
                    
                    # Update conversation history
                    state = manager.get_state(client_id)
                    history = state.get("conversation_history", [])
                    history.append({
                        "speaker": speaker,
                        "text": transcript,
                        "timestamp": datetime.now().isoformat(),
                        "confidence": alternative.confidence
                    })
                    manager.update_state(client_id, "conversation_history", history)
                    
                    # Analyze and get recommendations
                    try:
                        metrics = analyze_conversation(transcript, history)
                        manager.update_state(client_id, "metrics", metrics)
                        
                        # Send analysis
                        await manager.send_message(
                            client_id,
                            WSResponseMessage(
                                type="analysis",
                                data=metrics,
                                timestamp=datetime.now().isoformat()
                            )
                        )
                        
                        # Check coaching triggers
                        should_coach, reason = should_trigger_coaching(metrics, len(history))
                        
                        if should_coach:
                            recommendations = get_recommendations(transcript, history)
                            
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
                        try:
                            await manager.send_message(
                                client_id,
                                WSResponseMessage(
                                    type="error",
                                    data={"error": f"Analysis error: {str(e)}"},
                                    timestamp=datetime.now().isoformat()
                                )
                            )
                        except Exception:
                            # Client already disconnected, ignore
                            pass
            
            except Exception as e:
                logger.error(f"Error in GCP speech recognition: {e}")
                try:
                    await manager.send_message(
                        client_id,
                        WSResponseMessage(
                            type="error",
                            data={"error": f"Speech recognition error: {str(e)}"},
                            timestamp=datetime.now().isoformat()
                        )
                    )
                except Exception:
                    # Client already disconnected, ignore
                    pass
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
                audio_bytes = base64.b64decode(audio_msg.audio_data)
                
                # Detect if this is silence or speech by checking audio amplitude
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                if len(audio_array) > 0:
                    rms = np.mean(audio_array.astype(np.float64)**2)
                    audio_level = np.sqrt(rms) if rms >= 0 else 0
                else:
                    audio_level = 0
                is_speech = audio_level > 500  # Threshold for speech detection
                
                current_time = datetime.now()
                
                if is_speech:
                    # Add to buffer when speech is detected
                    audio_buffer.append(audio_bytes)
                    last_speech_time = current_time
                else:
                    # Check for speaker turn during silence
                    if last_speech_time and audio_buffer:
                        silence_duration = (current_time - last_speech_time).total_seconds()
                        
                        # Process accumulated speech if silence threshold is reached
                        if silence_duration >= silence_threshold:
                            await process_audio_batch()
                
                # Fallback: Process if buffer gets too large (prevent memory issues)
                total_audio_length = sum(len(chunk) for chunk in audio_buffer)
                audio_duration = total_audio_length / (16000 * 2)  # 16kHz, 16-bit
                
                if audio_duration >= 10.0:  # Maximum 10 seconds before forced processing
                    await process_audio_batch(force_process=True)
            
            elif message.type == "stop_audio":
                # Process any remaining audio
                if audio_buffer:
                    await process_audio_batch()
            
            elif message.type == "ping":
                # Respond to ping
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
        logger.info(f"Client {client_id} disconnected from GCP endpoint")
    
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        try:
            await manager.send_message(
                client_id,
                WSResponseMessage(
                    type="error",
                    data={"error": str(e)},
                    timestamp=datetime.now().isoformat()
                )
            )
        except Exception:
            # Client already disconnected, ignore
            pass
        manager.disconnect(client_id)