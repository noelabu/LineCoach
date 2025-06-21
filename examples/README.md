# LineCoach API Examples

This directory contains example clients that demonstrate how to use the LineCoach API.

## Prerequisites

Before running these examples, make sure you have:

1. Started the LineCoach API server:
   ```bash
   .venv/bin/python main.py
   ```

2. Installed the required dependencies:
   ```bash
   pip install requests pyaudio numpy
   ```

## Examples

### 1. Real-time Audio Client

The `real_time_audio_client.py` example demonstrates how to use the LineCoach API with real-time audio from a microphone.

**Usage:**
```bash
python examples/real_time_audio_client.py
```

This will:
1. List available audio input devices
2. Select an appropriate microphone
3. Start listening for speech
4. Send audio to the API for transcription
5. Get coaching recommendations based on the transcription
6. Display the results

### 2. Audio File Client

The `audio_file_client.py` example demonstrates how to use the LineCoach API with a pre-recorded audio file.

**Usage:**
```bash
python examples/audio_file_client.py path/to/audio.wav
```

**Options:**
- `--sample-rate`: Audio sample rate (default: 16000)
- `--channels`: Number of audio channels (default: 1)
- `--history`: Previous conversation history (optional)

**Example with options:**
```bash
python examples/audio_file_client.py path/to/audio.wav --sample-rate 44100 --channels 2 --history "Hello, how can I help you?" "I'm having an issue with my account."
```

## API Endpoints Used

These examples use the following LineCoach API endpoints:

1. **POST /api/v1/linecoach/transcribe**: Transcribe audio data
2. **POST /api/v1/linecoach/full_coaching**: Get coaching recommendations and analysis

## Notes

- Audio data should be in WAV format
- The API server must be running on http://localhost:8000
- The API endpoints are available at http://localhost:8000/api/v1/linecoach
- For real-time audio, make sure your microphone is properly configured 