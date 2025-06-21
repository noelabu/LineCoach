#!/usr/bin/env python3
"""
Simple test script to check if the LineCoach API is working.
"""

import requests
import json

def test_api():
    """Test the LineCoach API."""
    base_url = "http://localhost:8000"
    
    # Test the root endpoint
    print("Testing root endpoint...")
    try:
        response = requests.get(base_url)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test the API version endpoint
    print("\nTesting API version endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test the linecoach endpoint
    print("\nTesting linecoach endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1/linecoach")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test the transcribe endpoint
    print("\nTesting transcribe endpoint...")
    try:
        payload = {
            "audio_data": "test",
            "sample_rate": 16000,
            "channels": 1
        }
        response = requests.post(
            f"{base_url}/api/v1/linecoach/transcribe",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test the analyze endpoint
    print("\nTesting analyze endpoint...")
    try:
        payload = {
            "transcript": "Hello, how can I help you?",
            "conversation_history": []
        }
        response = requests.post(
            f"{base_url}/api/v1/linecoach/analyze",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api() 