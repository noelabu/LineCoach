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