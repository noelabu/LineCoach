# LineCoach Usage Guide

This guide explains how to use the LineCoach application for real-time conversation coaching and analysis.

## Available Scripts

LineCoach provides three different scripts with varying levels of functionality:

### 1. `test_streaming.py` - Basic Speech-to-Text with Recommendations

This is the simplest version that:
- Transcribes your speech in real-time using Google Cloud Speech-to-Text
- Provides basic recommendations using Vertex AI

```bash
python test_streaming.py
```

### 2. `gemini_live_coach.py` - Gemini-Powered Coaching

This script uses Google's Gemini API to:
- Stream audio from your microphone
- Process and transcribe the audio
- Generate coaching recommendations based on the conversation

```bash
python gemini_live_coach.py
```

### 3. `advanced_gemini_coach.py` - Advanced Coaching with Analytics

The most sophisticated version that:
- Demonstrates a full conversation coaching system
- Analyzes conversation metrics (sentiment, empathy, resolution progress)
- Provides detailed coaching recommendations
- Alerts when escalation is needed
- Saves session data for later review

```bash
python advanced_gemini_coach.py
```

## Setup Instructions

1. Create a `.env` file by copying `.env-example`:
   ```bash
   cp .env-example .env
   ```

2. Edit the `.env` file with your credentials:
   - Set `GOOGLE_APPLICATION_CREDENTIALS` to the path of your Google Cloud JSON credentials file
   - Set `PROJECT_ID` to your Google Cloud project ID
   - Set `GEMINI_API_KEY` to your Gemini API key (if using API key authentication)

3. Make sure your microphone is properly configured:
   - Not muted
   - Input volume turned up
   - Correct microphone selected in system settings

## Using the Advanced Demo

The `advanced_gemini_coach.py` script includes a demo mode that simulates a conversation between a customer and a service representative. It will:

1. Start audio processing (to demonstrate microphone setup)
2. Play through a simulated conversation
3. Analyze the conversation in real-time
4. Provide coaching recommendations after each exchange
5. Save the session data when complete

To exit any script, press `Ctrl+C`.

## Interpreting the Results

### Conversation Metrics

The advanced script provides four key metrics:

- **Sentiment Score** (-1 to 1): Overall emotional tone of the conversation
- **Empathy Score** (0 to 10): How well the agent is showing empathy
- **Resolution Progress** (0-100%): Progress toward resolving the issue
- **Escalation Risk** (0-100%): Likelihood that supervisor intervention is needed

### Coaching Recommendations

Recommendations are provided as actionable suggestions for the service representative, including:
- Specific aspects of the conversation to improve
- Why the improvement is important
- Suggested phrases to use

## Session Data

The advanced script saves session data to a `sessions` directory in JSON format, including:
- Full conversation transcript
- Metrics over time
- Coaching recommendations

You can review these files to track performance and identify training opportunities.

# LineCoach API Usage Guide

This guide provides examples of how to use the LineCoach API endpoints.

## API Endpoints

The LineCoach API provides the following endpoints:

### 1. Transcribe Audio

**Endpoint:** `POST /api/v1/linecoach/transcribe`

Transcribes audio data and returns the transcribed text.

**Request:**

```json
{
  "audio_data": "base64_encoded_audio_data",
  "sample_rate": 16000,
  "channels": 1,
  "content_type": "audio/wav"
}
```

**Response:**

```json
{
  "transcript": "Hello, I'm having an issue with my account."
}
```

### 2. Analyze Conversation

**Endpoint:** `POST /api/v1/linecoach/analyze`

Analyzes a conversation transcript for sentiment, empathy, resolution progress, and escalation risk.

**Request:**

```json
{
  "transcript": "I understand you're frustrated with the service. Let me help resolve this issue for you.",
  "conversation_history": [
    "Hello, I'm having an issue with my account.",
    "I've been trying to log in for hours and it's not working."
  ]
}
```

**Response:**

```json
{
  "sentiment": 0.2,
  "empathy": 8.5,
  "resolution": 45.0,
  "escalation": 15.0
}
```

### 3. Get Coaching Recommendations

**Endpoint:** `POST /api/v1/linecoach/coach`

Provides coaching recommendations based on the conversation.

**Request:**

```json
{
  "transcript": "I understand you're frustrated with the service. Let me help resolve this issue for you.",
  "conversation_history": [
    "Hello, I'm having an issue with my account.",
    "I've been trying to log in for hours and it's not working."
  ]
}
```

**Response:**

```json
{
  "recommendations": "1. Ask for specific details about the login attempts.\n2. Offer to reset their password.\n3. Reassure them that you'll stay with them until the issue is resolved."
}
```

### 4. Full Coaching Analysis

**Endpoint:** `POST /api/v1/linecoach/full_coaching`

Provides both coaching recommendations and conversation analysis.

**Request:**

```json
{
  "transcript": "I understand you're frustrated with the service. Let me help resolve this issue for you.",
  "conversation_history": [
    "Hello, I'm having an issue with my account.",
    "I've been trying to log in for hours and it's not working."
  ]
}
```

**Response:**

```json
{
  "recommendations": "1. Ask for specific details about the login attempts.\n2. Offer to reset their password.\n3. Reassure them that you'll stay with them until the issue is resolved.",
  "analysis": {
    "sentiment": 0.2,
    "empathy": 8.5,
    "resolution": 45.0,
    "escalation": 15.0
  }
}
```

## Example Usage with Python

Here's an example of how to use the API with Python:

```python
import requests
import base64
import json

# API base URL
BASE_URL = "http://localhost:8000/api/v1/linecoach"

# Example 1: Transcribe audio
def transcribe_audio(audio_file_path):
    # Read audio file and encode as base64
    with open(audio_file_path, "rb") as audio_file:
        audio_data = base64.b64encode(audio_file.read()).decode("utf-8")
    
    # Prepare request
    payload = {
        "audio_data": audio_data,
        "sample_rate": 16000,
        "channels": 1
    }
    
    # Send request
    response = requests.post(f"{BASE_URL}/transcribe", json=payload)
    return response.json()

# Example 2: Get coaching recommendations
def get_coaching(transcript, conversation_history=None):
    if conversation_history is None:
        conversation_history = []
    
    # Prepare request
    payload = {
        "transcript": transcript,
        "conversation_history": conversation_history
    }
    
    # Send request
    response = requests.post(f"{BASE_URL}/coach", json=payload)
    return response.json()

# Example usage
if __name__ == "__main__":
    # Transcribe an audio file
    result = transcribe_audio("path/to/audio.wav")
    print(f"Transcript: {result['transcript']}")
    
    # Get coaching recommendations
    coaching = get_coaching(
        "I understand you're frustrated with the service. Let me help resolve this issue for you.",
        ["Hello, I'm having an issue with my account.", 
         "I've been trying to log in for hours and it's not working."]
    )
    print(f"Recommendations: {coaching['recommendations']}")
```

## Notes

- All endpoints return JSON responses
- Error responses include a detail field with an error message
- Audio data should be base64 encoded
- The API supports WAV audio format 