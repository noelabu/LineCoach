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

### Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
   - Copy the `.env-example` file to `.env`
   - Update the values in the `.env` file with your own credentials:

```
GOOGLE_APPLICATION_CREDENTIALS='path-to-service-account-credentials'
PROJECT_ID='gcp-project-id'
```

   - Replace `path-to-service-account-credentials` with the path to your Google Cloud credentials JSON file
   - Replace `gcp-project-id` with your Google Cloud project ID

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

### Important Notes

- Ensure your microphone is properly configured and not muted
- Speak clearly for optimal speech recognition
- The system requires an active internet connection for API access
- Your Google Cloud service account needs the 'Vertex AI User' role
