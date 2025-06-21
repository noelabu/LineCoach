#!/usr/bin/env python3
"""
LineCoach Audio File Client Example

This script demonstrates how to use the LineCoach API with a pre-recorded audio file.
It reads an audio file, sends it to the API, and displays the results.
"""

import os
import base64
import requests
import argparse
from typing import Dict, Any, Optional

# API configuration
API_URL = "http://localhost:8000"
API_ENDPOINT = "/api/v1/linecoach"
HEADERS = {"Content-Type": "application/json"}


def read_audio_file(file_path: str) -> str:
    """Read an audio file and encode it as base64."""
    try:
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return ""


def transcribe_audio(audio_data_base64: str, sample_rate: int = 16000, channels: int = 1) -> Optional[str]:
    """Send audio data to API for transcription."""
    try:
        payload = {
            "audio_data": audio_data_base64,
            "sample_rate": sample_rate,
            "channels": channels,
            "content_type": "audio/wav"
        }
        
        print("Sending audio to API for transcription...")
        response = requests.post(
            f"{API_URL}{API_ENDPOINT}/transcribe",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["transcript"]
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
    
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None


def get_coaching(transcript: str, conversation_history: list = None) -> Optional[Dict[str, Any]]:
    """Get coaching recommendations based on the transcript."""
    if conversation_history is None:
        conversation_history = []
    
    try:
        payload = {
            "transcript": transcript,
            "conversation_history": conversation_history
        }
        
        print("Getting coaching recommendations...")
        response = requests.post(
            f"{API_URL}{API_ENDPOINT}/full_coaching",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
    
    except Exception as e:
        print(f"Error getting coaching: {e}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="LineCoach Audio File Client")
    parser.add_argument("audio_file", help="Path to the audio file (WAV format)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate (default: 16000)")
    parser.add_argument("--channels", type=int, default=1, help="Number of audio channels (default: 1)")
    parser.add_argument("--history", nargs="+", help="Previous conversation history (optional)")
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.isfile(args.audio_file):
        print(f"Error: File '{args.audio_file}' not found")
        return 1
    
    # Read audio file
    print(f"Reading audio file: {args.audio_file}")
    audio_data = read_audio_file(args.audio_file)
    if not audio_data:
        return 1
    
    # Transcribe audio
    transcript = transcribe_audio(audio_data, args.sample_rate, args.channels)
    if not transcript:
        return 1
    
    print(f"\nTranscript: {transcript}")
    
    # Get coaching recommendations
    conversation_history = args.history if args.history else []
    coaching_result = get_coaching(transcript, conversation_history)
    if coaching_result:
        print(f"\nRecommendations:")
        print(coaching_result["recommendations"])
        
        if "analysis" in coaching_result:
            analysis = coaching_result["analysis"]
            print(f"\nConversation Metrics:")
            print(f"Sentiment: {analysis['sentiment']}")
            print(f"Empathy: {analysis['empathy']}/10")
            print(f"Resolution: {analysis['resolution']}%")
            print(f"Escalation Risk: {analysis['escalation']}%")
    
    return 0


if __name__ == "__main__":
    exit(main()) 