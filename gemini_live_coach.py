import os
import pyaudio
import numpy as np
import time
import dotenv
from pathlib import Path
import google.generativeai as genai
from google.auth import default
from google.oauth2 import service_account
import threading
import queue
import sys

# Load environment variables from .env file
dotenv.load_dotenv(Path(__file__).parent / '.env')

# Set the path to your Google Cloud credentials file
credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
if not credentials_path:
    print("❌ GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    print("Please create a .env file with your Google Cloud credentials path")
    sys.exit(1)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

# Print a warning about microphone settings
print("\n⚠️  IMPORTANT: Please check your microphone settings!")
print("   1. Make sure your microphone is not muted")
print("   2. Check that the input volume is turned up")
print("   3. Speak clearly and close to the microphone")
print("   4. If using System Preferences > Sound > Input, ensure the correct microphone is selected\n")

# Initialize Gemini API
try:
    # Choose authentication method based on what's available
    if os.environ.get('GEMINI_API_KEY'):
        # Use API key authentication
        genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
        print("✅ Successfully initialized Gemini API with API key")
    else:
        # Use service account credentials
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        genai.configure(credentials=credentials)
        print("✅ Successfully initialized Gemini API with service account credentials")
    
    print("✅ Successfully initialized Gemini API")
except Exception as e:
    print(f"❌ Failed to initialize Gemini API: {e}")
    print("Please check your credentials and API key.")
    sys.exit(1)

class AudioStreamer:
    def __init__(self):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = int(self.RATE / 10)  # 100ms chunks
        self.GAIN = 5.0  # Amplify the audio signal
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.device_id = self._select_input_device()
        self.audio_queue = queue.Queue()
        self.is_running = False
        
        # For display purposes
        self.max_level_seen = 0
        self.last_update = time.time()
        self.audio_detected = False
        self.silent_chunks = 0
        
        # For transcription
        self.conversation_history = []
        
    def _select_input_device(self):
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
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
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
        while self.is_running:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                
                # Convert bytes to numpy array for processing
                audio_array = np.frombuffer(data, dtype=np.int16)
                
                # Apply gain to amplify the signal
                amplified_array = np.clip(audio_array * self.GAIN, -32768, 32767).astype(np.int16)
                
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
                
                # Detect speech based on audio level
                if audio_level > 500 and not self.audio_detected:
                    print(f"\n\n🔊 Audio detected! Level: {audio_level}")
                    self.audio_detected = True
                    self.silent_chunks = 0
                elif audio_level <= 300 and self.audio_detected:
                    self.silent_chunks += 1
                    if self.silent_chunks > 30:  # About 3.0 seconds of silence
                        print(f"\n🔇 Silence detected after speech")
                        self.audio_detected = False
                
                # Add the audio data to the queue for processing
                self.audio_queue.put(amplified_data)
                
            except Exception as e:
                print(f"\n⚠️ Error reading audio: {e}")
    
    def stop_streaming(self):
        """Stop streaming audio."""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("\n🛑 Stopped audio streaming.")

class GeminiTranscriber:
    def __init__(self, audio_streamer):
        self.audio_streamer = audio_streamer
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.is_running = False
        self.current_transcript = ""
        self.full_conversation = []
        
    def start_transcription(self):
        """Start transcribing audio."""
        self.is_running = True
        threading.Thread(target=self._transcription_thread, daemon=True).start()
        print("📝 Started transcription service.")
        
    def _transcription_thread(self):
        """Process audio and transcribe in a separate thread."""
        # This would normally connect to Gemini Live API for streaming
        # Since direct streaming audio to Gemini isn't available in the same way as Google Speech API,
        # we'll simulate it by collecting audio chunks and sending batches
        
        audio_buffer = b''
        silence_counter = 0
        
        while self.is_running:
            try:
                # Get audio data from the queue
                if not self.audio_streamer.audio_queue.empty():
                    audio_data = self.audio_streamer.audio_queue.get(timeout=0.1)
                    audio_buffer += audio_data
                    
                    # Reset silence counter when we get audio
                    silence_counter = 0
                else:
                    silence_counter += 1
                
                # If we have enough audio data or there's been silence, process it
                if len(audio_buffer) > 32000 or silence_counter > 30:  # ~2 seconds of audio or 3 seconds of silence
                    if len(audio_buffer) > 0:
                        # Here we would send audio to Gemini for transcription
                        # For simulation, we'll use a placeholder
                        # In a real implementation, you would use Gemini's audio transcription capabilities
                        
                        # Simulate transcription delay
                        time.sleep(0.5)
                        
                        # For demo purposes, we'll just print a message
                        # In a real implementation, you would get the transcript from Gemini
                        print("\n📝 TRANSCRIPT: [Audio processed for transcription]")
                        
                        # Clear the buffer
                        audio_buffer = b''
                        
                        # In a real implementation, we would update the transcript here
                        # For demo purposes, we'll just simulate getting recommendations
                        self._get_recommendations()
            
            except queue.Empty:
                # No audio data available, continue
                pass
            except Exception as e:
                print(f"\n⚠️ Error in transcription: {e}")
    
    def _get_recommendations(self):
        """Get recommendations based on the conversation."""
        # For demo purposes, we'll use a simulated transcript
        simulated_transcript = "Hi there, I'm having an issue with my account. I've been trying to access it for days but keep getting an error message."
        
        # Add to conversation history
        self.full_conversation.append(simulated_transcript)
        
        print(f"\n📝 TRANSCRIPT: {simulated_transcript}")
        print("\n⏳ Generating recommendations...")
        
        try:
            # Create prompt for Gemini
            prompt = f"""
            You're an assistant helping a customer service representative during a call.
            Your job is to listen and suggest what they should say next.

            Current transcript: "{simulated_transcript}"

            Role: Customer Service Representative

            Please provide 2-3 actionable suggestions on what to say next. Keep them concise and relevant.

            Examples:
            1. Ask about specific error messages they're seeing.
            2. Offer to reset their account access.
            3. Confirm their contact information for verification.

            What would you recommend saying next?
            """
            
            # Generate recommendations using Gemini
            response = self.model.generate_content(prompt)
            recommendations = response.text
            
            print(f"💡 RECOMMENDATIONS:\n{recommendations}\n")
            
        except Exception as e:
            print(f"❌ Error generating recommendation: {e}")
            print("This may be due to insufficient permissions or API issues.")
    
    def stop_transcription(self):
        """Stop transcribing audio."""
        self.is_running = False
        print("\n🛑 Stopped transcription service.")

def main():
    print("🎙️ Starting LineCoach with Gemini Live API...")
    
    try:
        # Initialize audio streaming
        audio_streamer = AudioStreamer()
        audio_streamer.start_streaming()
        
        # Initialize transcription
        transcriber = GeminiTranscriber(audio_streamer)
        transcriber.start_transcription()
        
        # Keep the main thread running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n👋 Stopping LineCoach...")
        finally:
            transcriber.stop_transcription()
            audio_streamer.stop_streaming()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 