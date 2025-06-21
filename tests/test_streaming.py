# --- Part 1: Real-Time Speech-to-Text ---
import os
import pyaudio
import numpy as np
from google.cloud import speech_v1p1beta1 as speech
import time
import dotenv
from pathlib import Path

# Load environment variables from .env file
dotenv.load_dotenv(Path(__file__).parent / '.env')

# Set the path to your Google Cloud credentials file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')

# Print a warning about microphone settings
print("\n⚠️  IMPORTANT: Please check your microphone settings!")
print("   1. Make sure your microphone is not muted")
print("   2. Check that the input volume is turned up")
print("   3. Speak clearly and close to the microphone")
print("   4. If using System Preferences > Sound > Input, ensure the correct microphone is selected\n")

def transcribe_streaming():
    client = speech.SpeechClient()

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = int(RATE / 10)
    GAIN = 5.0  # Amplify the audio signal

    # Print available audio devices for debugging
    audio = pyaudio.PyAudio()
    print("\n🎧 Available audio devices:")
    input_devices = []
    
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
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
    
    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=device_id
        )
        print("✅ Audio stream opened successfully")
    except Exception as e:
        print(f"❌ Error opening audio stream: {e}")
        raise

    print("🎤 Listening...")

    def request_generator():
        audio_detected = False
        silent_chunks = 0
        print("\n💡 Speak clearly into your microphone...")
        print("   The audio level indicator will show when your voice is detected")
        print("   Audio Level: [                    ] 0")
        
        # Track max level for display purposes
        max_level_seen = 0
        last_update = time.time()
        
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                # Convert bytes to numpy array for processing
                audio_array = np.frombuffer(data, dtype=np.int16)
                
                # Apply gain to amplify the signal
                amplified_array = np.clip(audio_array * GAIN, -32768, 32767).astype(np.int16)
                
                # Convert back to bytes
                amplified_data = amplified_array.tobytes()
                
                # Calculate audio level for display
                audio_level = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0
                max_level_seen = max(max_level_seen, audio_level)
                
                # Update audio level display every 0.5 seconds
                current_time = time.time()
                if current_time - last_update > 0.5:
                    # Create a visual bar representing audio level
                    bar_length = 20
                    filled_length = int(audio_level / 5000 * bar_length) if max_level_seen > 0 else 0
                    filled_length = min(filled_length, bar_length)  # Cap at bar_length
                    bar = '█' * filled_length + ' ' * (bar_length - filled_length)
                    
                    # Clear previous line and print new level
                    print(f"\r   Audio Level: [{bar}] {audio_level}    ", end='')
                    last_update = current_time
                
                # Detect speech based on audio level
                if audio_level > 500 and not audio_detected:
                    print(f"\n\n🔊 Audio detected! Level: {audio_level}")
                    audio_detected = True
                    silent_chunks = 0
                elif audio_level <= 300 and audio_detected:
                    silent_chunks += 1
                    if silent_chunks > 30:  # About 3.0 seconds of silence
                        print(f"\n🔇 Silence detected after speech")
                        audio_detected = False
                
                # Send the amplified audio to the Speech-to-Text API
                yield speech.StreamingRecognizeRequest(audio_content=amplified_data)
                
            except Exception as e:
                print(f"\n⚠️ Error reading audio: {e}")
                yield speech.StreamingRecognizeRequest(audio_content=b'')

    recognition_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US"
    )
    
    streaming_config = speech.StreamingRecognitionConfig(
        config=recognition_config,
        interim_results=True
    )

    requests = request_generator()
    responses = client.streaming_recognize(streaming_config, requests)

    try:
        for response in responses:
            for result in response.results:
                if result.is_final:
                    transcript = result.alternatives[0].transcript
                    print(f"\n📝 TRANSCRIPT: {transcript}")
                    yield transcript
    except KeyboardInterrupt:
        print("🛑 Stopped listening.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

# --- Part 2: Use Vertex AI to Get Recommendations ---
import vertexai
from vertexai.language_models import TextGenerationModel
import google.auth
from google.oauth2 import service_account
import sys

# Get project ID from environment variables
PROJECT_ID = os.environ.get('PROJECT_ID', '')
LOCATION = "us-central1"

# Load credentials explicitly
credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
credentials = service_account.Credentials.from_service_account_file(credentials_path)

try:
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
    
    # Try a more recent model version - gemini-2.0-flash-001 instead of text-bison@001
    # Also add a fallback to gemini-1.0-pro if text-bison is not accessible
    try:
        model = TextGenerationModel.from_pretrained("gemini-2.0-flash-lite-001")
        print("✅ Successfully loaded gemini-2.0-flash-001 model")
    except Exception as e:
        print(f"⚠️ Could not load text-bison model: {e}")
        print("Trying to use Gemini model instead...")
        try:
            from vertexai.generative_models import GenerativeModel
            model = GenerativeModel("gemini-2.0-flash-001")
            print("✅ Successfully loaded Gemini model")
        except Exception as e2:
            print(f"❌ Failed to load Gemini model: {e2}")
            print("Please check your service account permissions in the Google Cloud Console.")
            print("The service account needs the 'Vertex AI User' role.")
            raise
except Exception as e:
    print(f"❌ Failed to initialize Vertex AI: {e}")
    print("Please check your credentials and project settings.")
    sys.exit(1)

def get_recommendation(transcript):
    prompt = f"""
You're an assistant helping someone during a business meeting.
Your job is to listen and suggest what they should say next.

Current transcript: "{transcript}"

Role: Sales representative

Please provide 2-3 actionable suggestions on what to say next. Keep them concise and relevant.

Examples:
1. Ask about budget constraints.
2. Offer a phased delivery plan.
3. Confirm stakeholder alignment.

What would you recommend saying next?
"""

    try:
        # Check if we're using Gemini or text-bison
        if hasattr(model, 'predict'):
            # Text-bison model
            response = model.predict(prompt, temperature=0.5, max_output_tokens=256)
            return response.text
        else:
            # Gemini model
            response = model.generate_content(prompt, generation_config={"temperature": 0.5, "max_output_tokens": 256})
            return response.text
    except Exception as e:
        print(f"❌ Error generating recommendation: {e}")
        print("This may be due to insufficient permissions on your service account.")
        print("Please ensure your service account has the 'Vertex AI User' role.")
        return "Error generating recommendations. Please check your service account permissions."

# --- Main Loop ---
if __name__ == "__main__":
    print("🎙️ Starting meeting assistant...")
    for transcript in transcribe_streaming():
        print("\n⏳ Generating recommendations...")
        recommendation = get_recommendation(transcript)
        print(f"💡 RECOMMENDATIONS:\n{recommendation}\n")