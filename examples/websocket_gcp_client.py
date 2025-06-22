import asyncio
import websockets
import json
import pyaudio
import numpy as np
import base64
import threading
import queue
import sys
import os
from datetime import datetime
from pathlib import Path
import dotenv
import uuid

# Load environment variables
dotenv.load_dotenv(Path(__file__).parent.parent / '.env')


class WebSocketGCPClient:
    """WebSocket client for Google Cloud Speech-to-Text endpoint."""
    
    def __init__(self, server_url: str = "ws://localhost:8080/api/v1/linecoach-gcp/ws"):
        self.server_url = server_url
        self.client_id = str(uuid.uuid4())
        self.websocket = None
        self.is_running = False
        
        # Audio configuration for GCP Speech-to-Text
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # GCP requires 16kHz for phone_call model
        self.CHUNK = int(self.RATE / 20)  # 50ms chunks for better streaming
        self.GAIN = 1.5  # Lower gain for cleaner audio
        
        # Audio components
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        
        # For display purposes
        self.conversation_history = []
        self.last_recommendations = None
        self.last_metrics = None
        self.interim_transcript = ""
    
    def _select_input_device(self):
        """Select the best available input device."""
        print("\n🎧 Available audio devices:")
        input_devices = []
        
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"Device {i}: {device_info['name']}")
                input_devices.append((i, device_info['name']))
        
        # Try to find iPhone microphone or use first available
        selected_device = None
        for device_id, device_name in input_devices:
            if 'iPhone' in device_name:
                selected_device = device_id
                break
        
        if selected_device is None and input_devices:
            selected_device = input_devices[0][0]
        
        if selected_device is None:
            raise ValueError("No input devices found!")
        
        print(f"\n🎤 Using microphone: Device {selected_device}")
        return selected_device
    
    def start_audio_capture(self):
        """Start capturing audio from microphone."""
        device_id = self._select_input_device()
        
        try:
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                input_device_index=device_id
            )
            print("✅ Audio stream opened successfully")
            print("🔊 Using Google Cloud Speech-to-Text with speaker diarization")
        except Exception as e:
            print(f"❌ Error opening audio stream: {e}")
            raise
        
        self.is_running = True
        threading.Thread(target=self._audio_capture_thread, daemon=True).start()
    
    def _audio_capture_thread(self):
        """Capture audio in a separate thread."""
        silence_threshold = 500
        
        while self.is_running:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                
                # Convert to numpy array
                audio_array = np.frombuffer(data, dtype=np.int16)
                
                # Apply gain
                amplified_array = np.clip(audio_array * self.GAIN, -32768, 32767).astype(np.int16)
                amplified_data = amplified_array.tobytes()
                
                # Calculate audio level for display
                audio_level = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0
                
                # Display audio level
                bar_length = 20
                filled_length = int(audio_level / 5000 * bar_length) if audio_level > 0 else 0
                filled_length = min(filled_length, bar_length)
                bar = '█' * filled_length + ' ' * (bar_length - filled_length)
                
                # Show interim transcript if available
                if self.interim_transcript:
                    print(f"\r   Audio: [{bar}] {audio_level:4d} | Interim: {self.interim_transcript[:50]}{'...' if len(self.interim_transcript) > 50 else ''}    ", end='')
                else:
                    print(f"\r   Audio: [{bar}] {audio_level:4d}    ", end='')
                
                # Always send audio data for continuous streaming
                self.audio_queue.put(amplified_data)
                
            except Exception as e:
                print(f"\n⚠️ Error reading audio: {e}")
    
    async def send_audio_data(self):
        """Send audio data to the server via WebSocket."""
        while self.is_running:
            try:
                # Get audio from queue (smaller chunks for GCP streaming)
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get(timeout=0.05)
                    
                    # Encode audio data to base64
                    audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                    
                    # Create message
                    message = {
                        "type": "audio",
                        "data": {
                            "audio_data": audio_b64,
                            "sample_rate": self.RATE,
                            "channels": self.CHANNELS
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Send to server
                    await self.websocket.send(json.dumps(message))
                
                await asyncio.sleep(0.01)  # Small delay for smooth streaming
                
            except Exception as e:
                print(f"\n⚠️ Error sending audio: {e}")
    
    async def receive_messages(self):
        """Receive messages from the server."""
        while self.is_running:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                # Handle different message types
                if data["type"] == "connected":
                    print(f"\n✅ Connected to server with client ID: {data['data']['client_id']}")
                    print(f"🚀 Using engine: {data['data'].get('engine', 'unknown')}")
                
                elif data["type"] == "transcript":
                    transcript_data = data["data"]
                    
                    if transcript_data.get("is_final", False):
                        # Final transcript
                        transcript = transcript_data["transcript"]
                        speaker = transcript_data.get("speaker", "unknown")
                        confidence = transcript_data.get("confidence", 0)
                        
                        print(f"\n\n📝 [{speaker}] {transcript} (confidence: {confidence:.2%})")
                        
                        self.conversation_history.append({
                            "speaker": speaker,
                            "text": transcript,
                            "confidence": confidence
                        })
                        self.interim_transcript = ""
                    else:
                        # Interim transcript
                        self.interim_transcript = transcript_data["transcript"]
                
                elif data["type"] == "analysis":
                    self.last_metrics = data["data"]
                    print(f"\n📊 Conversation Metrics:")
                    print(f"   Sentiment: {self.last_metrics['sentiment']:.2f}")
                    print(f"   Empathy: {self.last_metrics['empathy']:.1f}/10")
                    print(f"   Resolution: {self.last_metrics['resolution']:.0f}%")
                    print(f"   Escalation Risk: {self.last_metrics['escalation']:.0f}%")
                
                elif data["type"] == "recommendation":
                    self.last_recommendations = data["data"]["recommendations"]
                    trigger = data["data"]["trigger_reason"]
                    print(f"\n🔔 COACHING TRIGGERED: {trigger}")
                    print(f"💡 RECOMMENDATIONS:\n{self.last_recommendations}")
                
                elif data["type"] == "error":
                    print(f"\n❌ Server error: {data['data']['error']}")
                
            except websockets.exceptions.ConnectionClosed:
                print("\n❌ Connection closed by server")
                self.is_running = False
                break
            except Exception as e:
                print(f"\n⚠️ Error receiving message: {e}")
    
    async def send_ping(self):
        """Send periodic ping messages to keep connection alive."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Ping every 30 seconds
                if self.websocket and not self.websocket.closed:
                    message = {
                        "type": "ping",
                        "timestamp": datetime.now().isoformat()
                    }
                    await self.websocket.send(json.dumps(message))
            except Exception as e:
                print(f"\n⚠️ Error sending ping: {e}")
    
    async def connect_and_stream(self):
        """Connect to WebSocket and start streaming."""
        full_url = f"{self.server_url}/{self.client_id}"
        print(f"🔌 Connecting to {full_url}...")
        
        try:
            async with websockets.connect(full_url) as websocket:
                self.websocket = websocket
                print("✅ WebSocket connected")
                
                # Start audio capture
                self.start_audio_capture()
                
                # Create tasks for sending and receiving
                send_task = asyncio.create_task(self.send_audio_data())
                receive_task = asyncio.create_task(self.receive_messages())
                ping_task = asyncio.create_task(self.send_ping())
                
                # Wait for tasks
                await asyncio.gather(send_task, receive_task, ping_task)
                
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the client."""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("\n🛑 Client stopped.")


async def main():
    print("🎙️ Starting LineCoach WebSocket Client with Google Cloud Speech-to-Text...")
    print("\n⚠️  IMPORTANT:")
    print("   • Make sure your microphone is not muted")
    print("   • Speak clearly and naturally")
    print("   • GCP Speech-to-Text will automatically detect speakers")
    print("   • Interim results will be shown as you speak")
    
    # Check for GCP credentials
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        print("\n❌ WARNING: GOOGLE_APPLICATION_CREDENTIALS not set")
        print("   The server may not be able to use Google Cloud Speech-to-Text")
    
    # Get server URL from environment or use default
    server_url = os.environ.get('LINECOACH_GCP_WS_URL', 'ws://localhost:8080/api/v1/linecoach-gcp/ws')
    
    client = WebSocketGCPClient(server_url)
    
    try:
        await client.connect_and_stream()
    except KeyboardInterrupt:
        print("\n\n👋 Stopping client...")
        client.stop()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())