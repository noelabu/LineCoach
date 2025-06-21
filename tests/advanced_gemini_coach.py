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
from google.oauth2 import service_account
import threading
import queue
import sys
import json
from datetime import datetime

# Load environment variables from .env file
dotenv.load_dotenv(Path(__file__).parent / '.env')

# Set the path to your Google Cloud credentials file
credentials_path = 'hacker2025-team-175-dev-cec04ece595a.json'
if not credentials_path:
    print("❌ GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    print("Please create a .env file with your Google Cloud credentials path")
    sys.exit(1)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

# Print a welcome message
print("\n🎙️ LineCoach Advanced - Powered by Gemini\n")
print("This application provides real-time coaching for customer service representatives")
print("by analyzing conversations and offering suggestions to improve communication.\n")

# Print a warning about microphone settings
print("⚠️  IMPORTANT: Please check your microphone settings!")
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
        print("✅ Using Gemini API with API key")
    else:
        # Use service account credentials
        credentials = service_account.Credentials.from_service_account_file('hacker2025-team-175-dev-cec04ece595a.json')
        genai.configure(credentials=credentials)
        print("✅ Using Gemini API with service account credentials")
    
    # Use the model specified in .env or default to gemini-1.5-flash
    model_name = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')
    
    print(f"✅ Successfully initialized Gemini API with model: {model_name}")
except Exception as e:
    print(f"❌ Failed to initialize Gemini API: {e}")
    print("Please check your credentials and API key.")
    sys.exit(1)

class AudioProcessor:
    """Handles audio input from microphone and processes it for transcription."""
    
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

class ConversationAnalyzer:
    """Analyzes conversation transcripts and provides coaching recommendations."""
    
    def __init__(self):
        # Use the model specified in .env or default to gemini-1.5-flash
        model_name = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')
        self.model = genai.GenerativeModel(model_name)
        
        # Track conversation history
        self.conversation_history = []
        self.speaker_roles = {
            "agent": "Customer Service Representative",
            "customer": "Customer"
        }
        self.current_speaker = "customer"  # Default to customer speaking first
        
        # Track coaching history to avoid repetition
        self.coaching_history = []
        
        # Conversation metrics
        self.metrics = {
            "sentiment_score": 0,  # -1 to 1
            "empathy_score": 0,    # 0 to 10
            "resolution_progress": 0,  # 0 to 100%
            "escalation_risk": 0,  # 0 to 100%
        }
        
        # Session data
        self.session_start = datetime.now()
        self.session_id = f"session_{self.session_start.strftime('%Y%m%d_%H%M%S')}"
    
    def add_transcript(self, transcript, speaker=None):
        """Add a new transcript to the conversation history."""
        if speaker:
            self.current_speaker = speaker
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "speaker": self.current_speaker,
            "role": self.speaker_roles.get(self.current_speaker, "Unknown"),
            "text": transcript
        }
        
        self.conversation_history.append(entry)
        
        # Toggle speaker for next entry (assuming alternating conversation)
        self.current_speaker = "agent" if self.current_speaker == "customer" else "customer"
        
        return entry
    
    def format_conversation_history(self):
        """Format the conversation history for the Gemini prompt."""
        formatted_history = ""
        for entry in self.conversation_history:
            formatted_history += f"[{entry['timestamp']}] {entry['role']}: {entry['text']}\n"
        return formatted_history
    
    def analyze_conversation(self):
        """Analyze the conversation and update metrics."""
        if len(self.conversation_history) < 2:
            # Not enough conversation to analyze yet
            return self.metrics
        
        try:
            # Create prompt for Gemini to analyze conversation
            conversation_text = self.format_conversation_history()
            
            prompt = f"""
            You are an expert conversation analyst for customer service interactions.
            Analyze the following conversation between a customer service representative and a customer.
            
            CONVERSATION:
            {conversation_text}
            
            Provide an analysis in JSON format with the following metrics:
            1. sentiment_score: A number from -1 (very negative) to 1 (very positive) representing the overall sentiment
            2. empathy_score: A number from 0 to 10 representing how empathetic the agent is being
            3. resolution_progress: A percentage (0-100) indicating progress toward resolving the customer's issue
            4. escalation_risk: A percentage (0-100) indicating the risk that this conversation needs escalation to a supervisor
            
            Return ONLY valid JSON with these four metrics and no other text.
            """
            
            response = self.model.generate_content(prompt)
            
            # Parse the JSON response
            try:
                # Extract JSON from the response
                response_text = response.text.strip()
                
                # If the response is wrapped in code blocks, extract just the JSON
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].strip()
                
                metrics = json.loads(response_text)
                
                # Update metrics with new values
                for key in self.metrics:
                    if key in metrics:
                        self.metrics[key] = metrics[key]
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Error parsing metrics JSON: {e}")
                print(f"Response was: {response.text}")
        
        except Exception as e:
            print(f"❌ Error analyzing conversation: {e}")
        
        return self.metrics
    
    def get_coaching_recommendations(self):
        """Generate coaching recommendations based on the conversation history."""
        if not self.conversation_history:
            return "Waiting for conversation to begin..."
        
        try:
            # Create prompt for Gemini
            conversation_text = self.format_conversation_history()
            metrics = self.analyze_conversation()
            
            prompt = f"""
            You are an expert coach for customer service representatives.
            
            CONVERSATION HISTORY:
            {conversation_text}
            
            CURRENT METRICS:
            - Sentiment Score: {metrics['sentiment_score']} (-1 to 1)
            - Empathy Score: {metrics['empathy_score']} (0 to 10)
            - Resolution Progress: {metrics['resolution_progress']}%
            - Escalation Risk: {metrics['escalation_risk']}%
            
            Based on this conversation and these metrics, provide 3 specific coaching recommendations for the customer service representative.
            Each recommendation should:
            1. Identify a specific aspect of the conversation to improve
            2. Explain why it's important
            3. Suggest 1-2 specific phrases the representative could use next
            
            Format your response as a bulleted list with clear, actionable advice.
            """
            
            response = self.model.generate_content(prompt)
            recommendations = response.text
            
            # Store in coaching history
            self.coaching_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "recommendations": recommendations
            })
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return "Error generating recommendations. Please check your Gemini API access."
    
    def get_escalation_alert(self):
        """Check if the conversation needs escalation to a supervisor."""
        if self.metrics['escalation_risk'] > 70:
            return {
                "escalate": True,
                "reason": "High risk of customer dissatisfaction",
                "urgency": "High" if self.metrics['escalation_risk'] > 85 else "Medium"
            }
        return {"escalate": False}
    
    def save_session(self, filepath=None):
        """Save the current session data to a JSON file."""
        if not filepath:
            filepath = f"sessions/{self.session_id}.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        session_data = {
            "session_id": self.session_id,
            "start_time": self.session_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "conversation": self.conversation_history,
            "final_metrics": self.metrics,
            "coaching_history": self.coaching_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"\n✅ Session saved to {filepath}")
        return filepath

class LineCoachApp:
    """Main application class for LineCoach."""
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.analyzer = ConversationAnalyzer()
        self.is_running = False
        
        # For audio transcription
        self.audio_buffer = b''
        self.silence_counter = 0
        self.current_speaker = "customer"  # Default to customer speaking first
        self.last_transcript_time = time.time()
    
    def start(self):
        """Start the LineCoach application."""
        print("\n🚀 Starting LineCoach Advanced...")
        
        # Create sessions directory if it doesn't exist
        os.makedirs("sessions", exist_ok=True)
        
        try:
            # Start audio processing
            self.audio_processor.start_streaming()
            
            # Start the main application loop
            self.is_running = True
            self._main_loop()
            
        except KeyboardInterrupt:
            print("\n\n👋 Stopping LineCoach...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.stop()
    
    def _main_loop(self):
        """Main application loop."""
        print("\n📊 Conversation Metrics:")
        print("   Waiting for conversation data...")
        
        while self.is_running:
            try:
                # Process audio from the queue
                if not self.audio_processor.audio_queue.empty():
                    audio_data = self.audio_processor.audio_queue.get(timeout=0.1)
                    self.audio_buffer += audio_data
                    
                    # Reset silence counter when we get audio
                    self.silence_counter = 0
                else:
                    self.silence_counter += 1
                
                # If we have enough audio data or there's been silence, process it
                current_time = time.time()
                if (len(self.audio_buffer) > 32000 or self.silence_counter > 30) and \
                   (current_time - self.last_transcript_time > 2):  # Minimum 2 seconds between transcriptions
                    
                    if len(self.audio_buffer) > 0:
                        # Convert audio buffer to numpy array for processing
                        audio_array = np.frombuffer(self.audio_buffer, dtype=np.int16)
                        
                        # Check if audio has enough signal to process with higher threshold
                        if np.max(np.abs(audio_array)) > 1500:  # Increased threshold from 500 to 1500
                            try:
                                # Create a prompt for Gemini that includes the audio transcription request
                                prompt = "Please transcribe the following audio. Return only the transcribed text without any additional commentary."
                                
                                # Create a properly formatted WAV file
                                import io
                                import wave
                                
                                # Create in-memory WAV file
                                wav_buffer = io.BytesIO()
                                with wave.open(wav_buffer, 'wb') as wav_file:
                                    wav_file.setnchannels(self.audio_processor.CHANNELS)
                                    wav_file.setsampwidth(2)  # 16-bit audio = 2 bytes
                                    wav_file.setframerate(self.audio_processor.RATE)
                                    wav_file.writeframes(self.audio_buffer)
                                
                                # Get the WAV data and convert to base64
                                wav_data = wav_buffer.getvalue()
                                wav_b64 = base64.b64encode(wav_data).decode('utf-8')
                                
                                # Create a multipart request with text and audio
                                model_name = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')
                                model = genai.GenerativeModel(model_name)
                                response = model.generate_content([
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
                                    # Toggle speaker for each transcription
                                    self.current_speaker = "customer" if self.current_speaker == "agent" else "agent"
                                    
                                    # Add to conversation history
                                    self.analyzer.add_transcript(transcript, self.current_speaker)
                                    
                                    # Print the transcript
                                    print(f"\n📝 [{datetime.now().strftime('%H:%M:%S')}] {self.analyzer.speaker_roles[self.current_speaker]}: {transcript}")
                                    
                                    # Update last transcript time
                                    self.last_transcript_time = current_time
                                    
                                    # Get updated metrics
                                    metrics = self.analyzer.analyze_conversation()
                                    
                                    # Print metrics
                                    print("\n📊 Conversation Metrics:")
                                    print(f"   Sentiment: {metrics['sentiment_score']:.2f} | Empathy: {metrics['empathy_score']:.1f}/10 | Resolution: {metrics['resolution_progress']}% | Escalation Risk: {metrics['escalation_risk']}%")
                                    
                                    # Determine when to provide coaching recommendations based on specific triggers
                                    should_coach = False
                                    coaching_reason = ""
                                    
                                    # Trigger 1: Low empathy score
                                    if metrics['empathy_score'] < 5:
                                        should_coach = True
                                        coaching_reason = "Low empathy detected"
                                    
                                    # Trigger 2: Negative sentiment
                                    if metrics['sentiment_score'] < -0.2:
                                        should_coach = True
                                        coaching_reason = "Negative sentiment detected"
                                    
                                    # Trigger 3: Rising escalation risk
                                    if metrics['escalation_risk'] > 30:
                                        should_coach = True
                                        coaching_reason = "Escalation risk increasing"
                                    
                                    # Trigger 4: Stalled resolution progress
                                    if metrics['resolution_progress'] < 40 and len(self.analyzer.conversation_history) > 6:
                                        should_coach = True
                                        coaching_reason = "Resolution progress stalled"
                                    
                                    # Trigger 5: Every 3rd customer message (regular check-in)
                                    if self.current_speaker == "customer" and len(self.analyzer.conversation_history) % 6 == 0:
                                        should_coach = True
                                        coaching_reason = "Regular coaching check-in"
                                    
                                    # Get coaching recommendations if triggers activated
                                    if should_coach:
                                        print(f"\n🔔 COACHING TRIGGERED: {coaching_reason}")
                                        recommendations = self.analyzer.get_coaching_recommendations()
                                        print(f"\n💡 COACHING RECOMMENDATIONS:\n{recommendations}\n")
                                    
                                    # Check for escalation
                                    escalation = self.analyzer.get_escalation_alert()
                                    if escalation.get("escalate", False):
                                        print(f"\n⚠️ ESCALATION ALERT - {escalation.get('urgency', 'MEDIUM')} URGENCY")
                                        print(f"   Reason: {escalation.get('reason', 'High risk of customer dissatisfaction')}")
                                        print("   Recommended action: Transfer to supervisor\n")
                            except Exception as e:
                                print(f"\n❌ Error transcribing audio: {e}")
                        
                        # Clear the buffer after processing
                        self.audio_buffer = b''
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\n⚠️ Error in main loop: {e}")
    
    def stop(self):
        """Stop the LineCoach application."""
        self.is_running = False
        self.audio_processor.stop_streaming()
        print("\n👋 LineCoach session ended.")

if __name__ == "__main__":
    app = LineCoachApp()
    app.start()