#!/usr/bin/env python3
"""
LineCoach API Test Script

This script tests all endpoints of the LineCoach API:
1. Root endpoint (/)
2. Transcribe endpoint (/api/v1/linecoach/transcribe)
3. Analyze endpoint (/api/v1/linecoach/analyze)
4. Coach endpoint (/api/v1/linecoach/coach)
5. Full coaching endpoint (/api/v1/linecoach/full_coaching)

Usage:
    python test_linecoach_api.py --url https://linecoach-api-922304318333.asia-southeast1.run.app
"""

import os
import base64
import requests
import argparse
import json
import wave
import numpy as np
import io
from typing import Dict, Any, Optional

# Default API configuration
DEFAULT_API_URL = "https://linecoach-api-922304318333.asia-southeast1.run.app"
API_ENDPOINT = "/api/v1/linecoach"
HEADERS = {"Content-Type": "application/json"}


def test_root_endpoint(api_url: str) -> bool:
    """Test the root endpoint of the API."""
    print("\n🔍 Testing root endpoint...")
    try:
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Root endpoint test passed: {response.json()}")
            return True
        else:
            print(f"❌ Root endpoint test failed: {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Error testing root endpoint: {e}")
        return False


def generate_test_audio() -> str:
    """Generate a simple test audio file with a sine wave and encode it as base64."""
    print("\n🔊 Generating test audio...")
    try:
        # Generate a simple sine wave
        sample_rate = 16000
        duration = 3  # seconds
        frequency = 440  # Hz (A4 note)
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t) * 32767 / 2  # Half amplitude
        audio = audio.astype(np.int16)
        
        # Create in-memory WAV file
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit audio = 2 bytes
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())
        
        # Get the WAV data and encode as base64
        wav_data = wav_buffer.getvalue()
        wav_b64 = base64.b64encode(wav_data).decode("utf-8")
        
        print(f"✅ Test audio generated ({len(wav_data)} bytes)")
        return wav_b64
    
    except Exception as e:
        print(f"❌ Error generating test audio: {e}")
        return ""


def test_transcribe_endpoint(api_url: str) -> Optional[str]:
    """Test the transcribe endpoint of the API."""
    print("\n🔍 Testing transcribe endpoint...")
    
    # Generate test audio
    audio_data = generate_test_audio()
    if not audio_data:
        return None
    
    try:
        payload = {
            "audio_data": audio_data,
            "sample_rate": 16000,
            "channels": 1,
            "content_type": "audio/wav"
        }
        
        response = requests.post(
            f"{api_url}{API_ENDPOINT}/transcribe",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            transcript = result.get("transcript", "")
            print(f"✅ Transcribe endpoint test passed: {transcript}")
            return transcript
        else:
            print(f"❌ Transcribe endpoint test failed: {response.status_code} - {response.text}")
            return None
    
    except Exception as e:
        print(f"❌ Error testing transcribe endpoint: {e}")
        return None


def test_analyze_endpoint(api_url: str, transcript: Optional[str] = None) -> bool:
    """Test the analyze endpoint of the API."""
    print("\n🔍 Testing analyze endpoint...")
    
    # Use provided transcript or a default one
    if not transcript:
        transcript = "Hello, I'm having an issue with my account. Can you help me?"
    
    try:
        payload = {
            "transcript": transcript,
            "conversation_history": ["Hi, how can I help you today?"]
        }
        
        response = requests.post(
            f"{api_url}{API_ENDPOINT}/analyze",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analyze endpoint test passed:")
            print(f"  - Sentiment: {result.get('sentiment', 'N/A')}")
            print(f"  - Empathy: {result.get('empathy', 'N/A')}/10")
            print(f"  - Resolution: {result.get('resolution', 'N/A')}%")
            print(f"  - Escalation Risk: {result.get('escalation', 'N/A')}%")
            return True
        else:
            print(f"❌ Analyze endpoint test failed: {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Error testing analyze endpoint: {e}")
        return False


def test_coach_endpoint(api_url: str, transcript: Optional[str] = None) -> bool:
    """Test the coach endpoint of the API."""
    print("\n🔍 Testing coach endpoint...")
    
    # Use provided transcript or a default one
    if not transcript:
        transcript = "Hello, I'm having an issue with my account. Can you help me?"
    
    try:
        payload = {
            "transcript": transcript,
            "conversation_history": ["Hi, how can I help you today?"]
        }
        
        response = requests.post(
            f"{api_url}{API_ENDPOINT}/coach",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Coach endpoint test passed:")
            print(f"  - Recommendations: {result.get('recommendations', 'N/A')[:100]}...")
            return True
        else:
            print(f"❌ Coach endpoint test failed: {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Error testing coach endpoint: {e}")
        return False


def test_full_coaching_endpoint(api_url: str, transcript: Optional[str] = None) -> bool:
    """Test the full_coaching endpoint of the API."""
    print("\n🔍 Testing full_coaching endpoint...")
    
    # Use provided transcript or a default one
    if not transcript:
        transcript = "Hello, I'm having an issue with my account. Can you help me?"
    
    try:
        payload = {
            "transcript": transcript,
            "conversation_history": ["Hi, how can I help you today?"]
        }
        
        response = requests.post(
            f"{api_url}{API_ENDPOINT}/full_coaching",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Full coaching endpoint test passed:")
            print(f"  - Recommendations: {result.get('recommendations', 'N/A')[:100]}...")
            
            if "analysis" in result:
                analysis = result["analysis"]
                print(f"  - Sentiment: {analysis.get('sentiment', 'N/A')}")
                print(f"  - Empathy: {analysis.get('empathy', 'N/A')}/10")
                print(f"  - Resolution: {analysis.get('resolution', 'N/A')}%")
                print(f"  - Escalation Risk: {analysis.get('escalation', 'N/A')}%")
            
            return True
        else:
            print(f"❌ Full coaching endpoint test failed: {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Error testing full_coaching endpoint: {e}")
        return False


def run_all_tests(api_url: str):
    """Run all tests for the LineCoach API."""
    print(f"\n🚀 Starting LineCoach API tests against {api_url}")
    
    # Test results
    results = {
        "root": False,
        "transcribe": False,
        "analyze": False,
        "coach": False,
        "full_coaching": False
    }
    
    # Test root endpoint
    results["root"] = test_root_endpoint(api_url)
    
    # Test transcribe endpoint
    transcript = test_transcribe_endpoint(api_url)
    results["transcribe"] = transcript is not None
    
    # Test analyze endpoint
    results["analyze"] = test_analyze_endpoint(api_url, transcript)
    
    # Test coach endpoint
    results["coach"] = test_coach_endpoint(api_url, transcript)
    
    # Test full_coaching endpoint
    results["full_coaching"] = test_full_coaching_endpoint(api_url, transcript)
    
    # Print summary
    print("\n📊 Test Results Summary:")
    for endpoint, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  - {endpoint.upper()}: {status}")
    
    # Overall result
    all_passed = all(results.values())
    print(f"\n{'🎉 All tests passed!' if all_passed else '❌ Some tests failed.'}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="LineCoach API Test Script")
    parser.add_argument("--url", default=DEFAULT_API_URL, help=f"API URL (default: {DEFAULT_API_URL})")
    args = parser.parse_args()
    
    run_all_tests(args.url)


if __name__ == "__main__":
    main() 