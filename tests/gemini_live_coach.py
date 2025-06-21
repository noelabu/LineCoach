import os
import io
import pyaudio
import numpy as np
import time
import dotenv
import base64
import wave
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
        self.GAIN = 2.0  # Reduced gain to make it less sensitive to background noise
        
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
                
                # Detect speech based on audio level with higher threshold
                if audio_level > 1500 and not self.audio_detected:  # Increased threshold from 500 to 1500
                    print(f"\n\n🔊 Audio detected! Level: {audio_level}")
                    self.audio_detected = True
                    self.silent_chunks = 0
                elif audio_level <= 800 and self.audio_detected:  # Increased threshold from 300 to 800
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
        """Process audio and transcribe in a separate thread using Gemini."""
        # Process audio in batches for transcription
        
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
                        # Convert audio buffer to numpy array for processing
                        audio_array = np.frombuffer(audio_buffer, dtype=np.int16)
                        
                        # Check if audio has enough signal to process with higher threshold
                        if np.max(np.abs(audio_array)) > 1500:  # Increased threshold from 500 to 1500
                            try:
                                # Create a prompt for Gemini that includes the audio transcription request
                                prompt = "Please transcribe the following audio. Return only the transcribed text without any additional commentary."
                                
                                # Convert audio to base64 for API transmission
                                import base64
                                audio_b64 = base64.b64encode(audio_buffer).decode('utf-8')
                                
                                # Create a multipart request with text and audio
                                # Gemini expects specific audio formats - let's use MP3
                                # First, convert the raw PCM audio to proper WAV format
                                import io
                                import wave
                                
                                # Create in-memory WAV file
                                wav_buffer = io.BytesIO()
                                with wave.open(wav_buffer, 'wb') as wav_file:
                                    wav_file.setnchannels(self.audio_streamer.CHANNELS)
                                    wav_file.setsampwidth(2)  # 16-bit audio = 2 bytes
                                    wav_file.setframerate(self.audio_streamer.RATE)
                                    wav_file.writeframes(audio_buffer)
                                
                                # Get the WAV data
                                wav_data = wav_buffer.getvalue()
                                wav_b64 = base64.b64encode(wav_data).decode('utf-8')
                                
                                response = self.model.generate_content([
                                    prompt,
                                    {
                                        "mime_type": "audio/wav",
                                        "data": wav_b64
                                    }
                                ])
                                
                                # Get the transcription from the response
                                transcript = response.text.strip()
                                
                                # Only process non-empty transcripts
                                if transcript and len(transcript) > 0:
                                    print(f"\n📝 TRANSCRIPT: {transcript}")
                                    
                                    # Add to conversation history
                                    self.current_transcript = transcript
                                    self.full_conversation.append(transcript)
                                    
                                    # First analyze the conversation
                                    self._analyze_conversation(transcript)
                                    
                                    # Determine if coaching is needed based on specific triggers
                                    should_coach = False
                                    coaching_reason = ""
                                    
                                    # Get the last analysis results if available
                                    sentiment = getattr(self, 'last_sentiment', 0)
                                    empathy = getattr(self, 'last_empathy', 0)
                                    resolution = getattr(self, 'last_resolution', 0)
                                    escalation = getattr(self, 'last_escalation', 0)
                                    
                                    # Trigger 1: Low empathy score
                                    if empathy < 5:
                                        should_coach = True
                                        coaching_reason = "Low empathy detected"
                                    
                                    # Trigger 2: Negative sentiment
                                    if sentiment < -0.2:
                                        should_coach = True
                                        coaching_reason = "Negative sentiment detected"
                                    
                                    # Trigger 3: Rising escalation risk
                                    if escalation > 30:
                                        should_coach = True
                                        coaching_reason = "Escalation risk increasing"
                                    
                                    # Trigger 4: Stalled resolution progress
                                    if resolution < 40 and len(self.full_conversation) > 6:
                                        should_coach = True
                                        coaching_reason = "Resolution progress stalled"
                                    
                                    # Trigger 5: Regular check-in (every 3rd message)
                                    if len(self.full_conversation) % 3 == 0:
                                        should_coach = True
                                        coaching_reason = "Regular coaching check-in"
                                    
                                    # Get recommendations if triggers activated
                                    if should_coach:
                                        print(f"\n🔔 COACHING TRIGGERED: {coaching_reason}")
                                        self._get_recommendations(transcript)
                            except Exception as e:
                                print(f"\n❌ Error transcribing audio: {e}")
                        
                        # Clear the buffer after processing
                        audio_buffer = b''
                        
            except queue.Empty:
                # No audio data available, continue
                pass
            except Exception as e:
                print(f"\n⚠️ Error in transcription: {e}")
    
    def _get_recommendations(self, transcript):
        """Get recommendations based on the conversation."""
        print("\n⏳ Generating recommendations...")
        
        try:
            # Create prompt for Gemini
            prompt = f"""
            You're an assistant helping a customer service representative during a call.
            Your job is to listen and suggest what they should say next.

            Current transcript: "{transcript}"

            Previous conversation context: "{' '.join(self.full_conversation[-3:])}"

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
            
            # Also analyze sentiment and escalation risk
            self._analyze_conversation(transcript)
            
        except Exception as e:
            print(f"❌ Error generating recommendation: {e}")
            print("This may be due to insufficient permissions or API issues.")
            
    def _analyze_conversation(self, transcript):
        """Analyze the conversation for sentiment and escalation risk."""
        try:
            # Create prompt for analysis
            prompt = f"""
            Analyze the following customer service conversation transcript:
            
            "{transcript}"
            
            Previous conversation context: "{' '.join(self.full_conversation[-3:])}"
            
            Please provide the following metrics:
            1. Sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, and 1 is very positive)
            2. Empathy score (0 to 10, where 10 is extremely empathetic)
            3. Resolution progress (0% to 100%)
            4. Escalation risk (0% to 100%)
            
            Format your response as:
            Sentiment: [score]
            Empathy: [score]/10
            Resolution: [score]%
            Escalation Risk: [score]%
            """
            
            # Generate analysis using Gemini
            response = self.model.generate_content(prompt)
            analysis = response.text
            
            print(f"\n📊 Conversation Metrics:\n   {analysis}")
            
            # Parse the analysis to extract metrics
            try:
                lines = analysis.strip().split('\n')
                for line in lines:
                    if line.startswith('Sentiment:'):
                        self.last_sentiment = float(line.split(':')[1].strip())
                    elif line.startswith('Empathy:'):
                        self.last_empathy = float(line.split(':')[1].split('/')[0].strip())
                    elif line.startswith('Resolution:'):
                        self.last_resolution = float(line.split(':')[1].strip('%'))
                    elif line.startswith('Escalation Risk:'):
                        self.last_escalation = float(line.split(':')[1].strip('%'))
            except Exception as parse_error:
                print(f"⚠️ Error parsing metrics: {parse_error}")
            
        except Exception as e:
            print(f"❌ Error analyzing conversation: {e}")
    
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