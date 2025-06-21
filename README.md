# LineCoach

## Overview

LineCoach is a Multi-agent Coaching and Real-time Escalation system designed to enhance customer service interactions. It leverages AI to monitor conversations and provide real-time guidance to service representatives.

### Key Features

* Analyze real-time voice and chat signals for emotion, tone, and frustration cues.
* Apply escalation thresholds and trigger alerts to supervisors or live agents.
* Recommend personalized coaching actions (e.g., tone adjustment, empathy prompts).

## AI Agents

LineCoach operates through a system of specialized AI agents:

1. **Meeting Listener Agent**: Listens and determines the context and flow of the meeting/conversation.
2. **Coaching Agent**: Recommends and coaches agents on the tone and content of the conversation.
3. **Escalation Agent**: Checks for escalation signals in the conversation that may require intervention.
4. **Notification Agent**: If escalation is detected, notifies supervisors via email/chat.

## Setup

### Prerequisites

- Google Cloud account with Speech-to-Text and Vertex AI enabled
- Google Cloud credentials file
- Gemini API access (for gemini_live_coach.py)

## Getting started with Development

### Pre-requisites
Before running the application, make sure you have the following installed:

* Python v3.12^
* uv v0.5.23^

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/noelabu/LineCoach.git
   ```
2. Navigate to the project directory:
   ```bash
   cd LineCoach
   ```
3. Install dependencies using uv:
   ```bash
   # Create the virtual environment within the project directory for code editors to use.
   uv venv --python 3.12.0

   # Activate the virtual environment
   # On macOS/Linux
   source .venv/bin/activate
   # On Windows
   .venv\Scripts\activate

   # Install all the dependencies for the package in a virtual environment
   uv pip install -e .

   # Install pre-commit hooks (if available)
   uv run pre-commit install
   ```

### Configuration

1. Copy the example `.env-example` file to `.env`:
   ```bash
   cp .env-example .env
   ```

2. Update the values in the `.env` file with your own credentials:
   ```
   GOOGLE_APPLICATION_CREDENTIALS='path-to-service-account-credentials'
   PROJECT_ID='gcp-project-id'
   GEMINI_API_KEY='your-gemini-api-key'
   ```

   - Replace `path-to-service-account-credentials` with the path to your Google Cloud credentials JSON file
   - Replace `gcp-project-id` with your Google Cloud project ID
   - Replace `your-gemini-api-key` with your Gemini API key (if using API key authentication)

## Running the Application

### Test Streaming

To run the test streaming application:

```bash
uv run python tests/test_streaming.py
```

This will:
1. Initialize the audio input device (preferring iPhone microphone if available)
2. Start listening for speech
3. Display an audio level indicator to show when your voice is detected
4. Transcribe your speech in real-time using Google Cloud Speech-to-Text
5. Generate coaching recommendations using Vertex AI

### Gemini Live Coach

To run the Gemini-powered coaching application:

```bash
uv run python tests/gemini_live_coach.py
```

This will:
1. Initialize the audio input device and Gemini API
2. Start listening for speech with visual audio level indicators
3. Process audio in chunks and send to Gemini for transcription
4. Generate real-time coaching recommendations based on the conversation
5. Display recommendations to assist the customer service representative

### API Server

To run the LineCoach API server:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

This will start the FastAPI server on http://localhost:8000. 

You can access the API documentation at http://localhost:8000/docs.

#### API Endpoints

The following endpoints are available:

- **POST /api/v1/linecoach/transcribe**: Transcribe audio data
  - Request: Audio data in base64 format
  - Response: Transcribed text

- **POST /api/v1/linecoach/analyze**: Analyze conversation for sentiment and metrics
  - Request: Transcript and conversation history
  - Response: Sentiment, empathy, resolution, and escalation metrics

- **POST /api/v1/linecoach/coach**: Get coaching recommendations
  - Request: Transcript and conversation history
  - Response: Coaching recommendations

- **POST /api/v1/linecoach/full_coaching**: Get both recommendations and analysis
  - Request: Transcript and conversation history
  - Response: Recommendations and analysis metrics

### Important Notes

- Ensure your microphone is properly configured and not muted
- Speak clearly for optimal speech recognition
- The system requires an active internet connection for API access
- Your Google Cloud service account needs appropriate permissions
- For Gemini API access, you need either a service account with proper permissions or an API key
