#!/usr/bin/env python3
"""
LineCoach Real-time Audio Client Example

This script demonstrates how to use the LineCoach API with real-time audio.
It captures audio from the microphone, sends it to the API, and displays the results.
"""

import os
import io
import base64
import time
import json
import threading
import queue
import pyaudio
import numpy as np
import requests
import wave
from typing import List, Dict, Any, Optional

# API configuration
API_URL = "http://localhost:8000"
API_ENDPOINT = "/api/v1/linecoach"
HEADERS = {"Content-Type": "application/json"}

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms chunks
GAIN = 2.0  # Audio gain to amplify signal


class LineCoachClient:
    """Client for the LineCoach API with real-time audio processing."""
    
    def __init__(self):
        """Initialize the LineCoach client."""
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.device_id = self._select_input_device()
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.conversation_history = []
        
        # For display purposes
        self.max_level_seen = 0
        self.last_update = time.time()
        self.audio_detected = False
        self.silent_chunks = 0
    
    def _select_input_device(self) -> int:
        """Select the best available input device."""
        print("\n🎧 Available audio devices:")
        input_devices = []
        
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            print(f"Device {i}: {device_info['name']}")
            print(f"  - Input channels: {device_info['maxInputChannels']}")
            print(f"  - Default sample rate: {device_info['defaultSampleRate']}")
            
            # Add to list of input devices if it has input channels
            if device_info['maxInputChannels'] > 0:
                input_devices.append((i, device_info['name']))
        
        # Try to use iPhone microphone first if available
        selected_device = None
        for device_id, device_name in input_devices:
            if 'iPhone' in device_name:
                selected_device = (device_id, device_name)
                break
        
        # Fall back to default if iPhone mic not found
        if not selected_device and input_devices:
            selected_device = input_devices[0]
        
        if not selected_device:
            raise ValueError("No input devices found!")
            
        device_id, device_name = selected_device
        print(f"\n🎤 Using microphone: {device_name} (ID: {device_id})")
        return device_id
    
    def start_streaming(self):
        """Start streaming audio from the microphone."""
        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=self.device_id
            )
            print("✅ Audio stream opened successfully")
        except Exception as e:
            print(f"❌ Error opening audio stream: {e}")
            raise
        
        self.is_running = True
        threading.Thread(target=self._audio_processing_thread, daemon=True).start()
        print("🎤 Listening...")
        print("\n💡 Speak clearly into your microphone...")
        print("   The audio level indicator will show when your voice is detected")
        print("   Audio Level: [                    ] 0")
    
    def _audio_processing_thread(self):
        """Process audio in a separate thread."""
        audio_buffer = b''
        silence_counter = 0
        
        while self.is_running:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                
                # Convert bytes to numpy array for processing
                audio_array = np.frombuffer(data, dtype=np.int16)
                
                # Apply gain to amplify the signal
                amplified_array = np.clip(audio_array * GAIN, -32768, 32767).astype(np.int16)
                
                # Convert back to bytes
                amplified_data = amplified_array.tobytes()
                
                # Calculate audio level for display
                audio_level = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0
                self.max_level_seen = max(self.max_level_seen, audio_level)
                
                # Update audio level display every 0.5 seconds
                current_time = time.time()
                if current_time - self.last_update > 0.5:
                    # Create a visual bar representing audio level
                    bar_length = 20
                    filled_length = int(audio_level / 5000 * bar_length) if self.max_level_seen > 0 else 0
                    filled_length = min(filled_length, bar_length)  # Cap at bar_length
                    bar = '█' * filled_length + ' ' * (bar_length - filled_length)
                    
                    # Clear previous line and print new level
                    print(f"\r   Audio Level: [{bar}] {audio_level}    ", end='')
                    self.last_update = current_time
                
                # Detect speech based on audio level with higher threshold
                if audio_level > 1500 and not self.audio_detected:  # Increased threshold from 500 to 1500
                    print(f"\n\n🔊 Audio detected! Level: {audio_level}")
                    self.audio_detected = True
                    silence_counter = 0
                elif audio_level <= 800 and self.audio_detected:  # Increased threshold from 300 to 800
                    silence_counter += 1
                    if silence_counter > 30:  # About 3.0 seconds of silence
                        print(f"\n🔇 Silence detected after speech")
                        self.audio_detected = False
                
                # Add audio to buffer
                audio_buffer += amplified_data
                
                # If we have enough audio data or there's been silence, process it
                if len(audio_buffer) > 32000 or (self.audio_detected and silence_counter > 30):  # ~2 seconds of audio or 3 seconds of silence
                    if len(audio_buffer) > 0 and self.audio_detected:
                        # Process the audio buffer
                        self._process_audio_buffer(audio_buffer)
                        
                        # Clear the buffer after processing
                        audio_buffer = b''
                        silence_counter = 0
                
            except Exception as e:
                print(f"\n⚠️ Error reading audio: {e}")
    
    def _process_audio_buffer(self, audio_buffer: bytes):
        """Process audio buffer and send to API."""
        try:
            # Create in-memory WAV file
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(2)  # 16-bit audio = 2 bytes
                wav_file.setframerate(RATE)
                wav_file.writeframes(audio_buffer)
            
            # Get the WAV data and encode as base64
            wav_data = wav_buffer.getvalue()
            wav_b64 = base64.b64encode(wav_data).decode('utf-8')
            
            # Send to API for transcription
            print("\n📤 Sending audio to API for transcription...")
            transcript = self._transcribe_audio(wav_b64)
            
            if transcript and len(transcript) > 0:
                print(f"\n📝 TRANSCRIPT: {transcript}")
                
                # Add to conversation history
                self.conversation_history.append(transcript)
                
                # Get coaching recommendations
                self._get_coaching(transcript)
        
        except Exception as e:
            print(f"\n❌ Error processing audio buffer: {e}")
    
    def _transcribe_audio(self, audio_data_base64: str) -> str:
        """Send audio data to API for transcription."""
        try:
            payload = {
                "audio_data": audio_data_base64,
                "sample_rate": RATE,
                "channels": CHANNELS,
                "content_type": "audio/wav"
            }
            
            response = requests.post(
                f"{API_URL}{API_ENDPOINT}/transcribe",
                headers=HEADERS,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()["transcript"]
            else:
                print(f"\n❌ API Error: {response.status_code} - {response.text}")
                return ""
        
        except Exception as e:
            print(f"\n❌ Error transcribing audio: {e}")
            return ""
    
    def _get_coaching(self, transcript: str):
        """Get coaching recommendations based on the transcript."""
        try:
            # Keep only the last 5 items in conversation history
            recent_history = self.conversation_history[-5:] if len(self.conversation_history) > 5 else self.conversation_history
            
            payload = {
                "transcript": transcript,
                "conversation_history": recent_history[:-1]  # Exclude the current transcript
            }
            
            print("\n⏳ Getting coaching recommendations...")
            response = requests.post(
                f"{API_URL}{API_ENDPOINT}/full_coaching",
                headers=HEADERS,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display recommendations
                print(f"\n💡 RECOMMENDATIONS:\n{result['recommendations']}")
                
                # Display analysis if available
                if "analysis" in result:
                    analysis = result["analysis"]
                    print(f"\n📊 Conversation Metrics:")
                    print(f"   Sentiment: {analysis['sentiment']}")
                    print(f"   Empathy: {analysis['empathy']}/10")
                    print(f"   Resolution: {analysis['resolution']}%")
                    print(f"   Escalation Risk: {analysis['escalation']}%")
            else:
                print(f"\n❌ API Error: {response.status_code} - {response.text}")
        
        except Exception as e:
            print(f"\n❌ Error getting coaching: {e}")
    
    def stop_streaming(self):
        """Stop streaming audio."""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("\n🛑 Stopped audio streaming.")


def main():
    """Main function to run the LineCoach client."""
    print("🎙️ Starting LineCoach Real-time Audio Client...")
    
    try:
        # Initialize client
        client = LineCoachClient()
        client.start_streaming()
        
        # Keep the main thread running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n👋 Stopping LineCoach client...")
        finally:
            client.stop_streaming()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 