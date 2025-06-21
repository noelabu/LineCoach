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

## Deployment to Google Cloud Run

### Prerequisites
1. Google Cloud SDK installed and configured
2. Docker installed locally
3. Access to a Google Cloud project with the following APIs enabled:
   - Cloud Run API
   - Cloud Build API
   - Container Registry API
   - Secret Manager API (for managing sensitive environment variables)

### Deployment Steps

1. **Set up your Google Cloud project**
   ```bash
   # Set your project ID
   export PROJECT_ID=your-project-id
   gcloud config set project $PROJECT_ID
   ```

2. **Enable required APIs**
   ```bash
   gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com secretmanager.googleapis.com
   ```

3. **Store sensitive environment variables in Secret Manager**
   ```bash
   # Create a secret for the Gemini API key
   echo -n "your-gemini-api-key" | gcloud secrets create gemini-api-key --data-file=-
   
   # Grant access to the Cloud Run service account
   gcloud secrets add-iam-policy-binding gemini-api-key \
     --member="serviceAccount:$PROJECT_ID-compute@developer.gserviceaccount.com" \
     --role="roles/secretmanager.secretAccessor"
   ```

4. **Deploy using Cloud Build**
   ```bash
   gcloud builds submit --config=cloudbuild.dev.yaml \
     --substitutions=_PROJECT_ID=$PROJECT_ID,_SERVICE_NAME=linecoach,_ENVIRONMENT=dev,_REGION=asia-southeast1
   ```

5. **Access your deployed application**
   ```bash
   gcloud run services describe linecoach --region=asia-southeast1 --format='value(status.url)'
   ```

### Manual Deployment (Alternative)
You can also deploy directly without using Cloud Build:

```bash
# Build the container using the simplified Dockerfile for Cloud Run
docker build -t gcr.io/$PROJECT_ID/linecoach -f Dockerfile.cloud .

# Push to Container Registry
docker push gcr.io/$PROJECT_ID/linecoach

# Deploy to Cloud Run
gcloud run deploy linecoach \
  --image gcr.io/$PROJECT_ID/linecoach \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest
```

### Docker Files
- `Dockerfile`: Full development environment with audio processing capabilities (local development)
- `Dockerfile.cloud`: Simplified container for Cloud Run deployment without audio processing dependencies

### Troubleshooting

If you encounter issues with PyAudio dependencies during deployment:
1. We've created a separate `requirements.cloud.txt` file that excludes PyAudio
2. The `Dockerfile.cloud` uses this file to build a deployment-ready container
3. The Cloud Run deployment doesn't need audio processing capabilities, so PyAudio is not required
