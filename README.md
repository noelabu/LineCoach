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

- Python 3.7+
- Google Cloud account with Speech-to-Text and Vertex AI enabled
- Google Cloud credentials file
- Gemini API access (for gemini_live_coach.py)

### Installation

1. Clone the repository
2. Create and activate a virtual environment (recommended):

```bash
# Create a virtual environment
python -m venv linecoach-env

# Activate the virtual environment
# On macOS/Linux:
source linecoach-env/bin/activate
# On Windows:
linecoach-env\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
   - Copy the `.env-example` file to `.env`
   - Update the values in the `.env` file with your own credentials:

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
python test_streaming.py
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
python gemini_live_coach.py
```

This will:
1. Initialize the audio input device and Gemini API
2. Start listening for speech with visual audio level indicators
3. Process audio in chunks and send to Gemini for transcription
4. Generate real-time coaching recommendations based on the conversation
5. Display recommendations to assist the customer service representative

### Important Notes

- Ensure your microphone is properly configured and not muted
- Speak clearly for optimal speech recognition
- The system requires an active internet connection for API access
- Your Google Cloud service account needs appropriate permissions
- For Gemini API access, you need either a service account with proper permissions or an API key
