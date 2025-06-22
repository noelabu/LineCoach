# LineCoach API Test Script

This script tests all endpoints of the LineCoach API deployed at https://linecoach-api-922304318333.asia-southeast1.run.app

## Features

- Tests all available LineCoach API endpoints:
  - Root endpoint (`/`)
  - Transcribe endpoint (`/api/v1/linecoach/transcribe`)
  - Analyze endpoint (`/api/v1/linecoach/analyze`)
  - Coach endpoint (`/api/v1/linecoach/coach`)
  - Full coaching endpoint (`/api/v1/linecoach/full_coaching`)
- Generates test audio for the transcription endpoint
- Provides detailed test results and error messages
- Displays a summary of all test results

## Requirements

- Python 3.7+
- Required Python packages:
  - requests
  - numpy
  - pyaudio (optional, only if you want to test with real audio)

## Installation

1. Clone the repository or download the test script
2. Install required packages:

```bash
pip install requests numpy
```

## Usage

Run the script with default settings (tests against the production API):

```bash
python test_linecoach_api.py
```

Specify a different API URL:

```bash
python test_linecoach_api.py --url http://localhost:8000
```

## Example Output

```
🚀 Starting LineCoach API tests against https://linecoach-api-922304318333.asia-southeast1.run.app

🔍 Testing root endpoint...
✅ Root endpoint test passed: {'message': 'LineCoach API is running', 'docs': '/docs'}

🔊 Generating test audio...
✅ Test audio generated (96044 bytes)

🔍 Testing transcribe endpoint...
✅ Transcribe endpoint test passed: The audio appears to contain a continuous tone or note.

🔍 Testing analyze endpoint...
✅ Analyze endpoint test passed:
  - Sentiment: 0.0
  - Empathy: 0.0/10
  - Resolution: 0.0%
  - Escalation Risk: 0.0%

🔍 Testing coach endpoint...
✅ Coach endpoint test passed:
  - Recommendations: 1. Ask the customer to describe what they're hearing.
2. Inquire if they've noticed any pattern...

🔍 Testing full_coaching endpoint...
✅ Full coaching endpoint test passed:
  - Recommendations: 1. Ask the customer to describe what they're hearing.
2. Inquire if they've noticed any pattern...
  - Sentiment: 0.0
  - Empathy: 0.0/10
  - Resolution: 0.0%
  - Escalation Risk: 0.0%

📊 Test Results Summary:
  - ROOT: ✅ PASSED
  - TRANSCRIBE: ✅ PASSED
  - ANALYZE: ✅ PASSED
  - COACH: ✅ PASSED
  - FULL_COACHING: ✅ PASSED

🎉 All tests passed!
```

## Troubleshooting

If you encounter errors:

1. Check that the API is running and accessible
2. Verify your internet connection
3. Check if the API endpoints have changed
4. Ensure you have the required Python packages installed

### Authentication Errors

If you see errors like `403 Request had insufficient authentication scopes` or `ACCESS_TOKEN_SCOPE_INSUFFICIENT`, this indicates a problem with the Google Generative AI API authentication:

1. **API Key Not Set**: Ensure the GEMINI_API_KEY environment variable is properly set on the server
2. **Invalid API Key**: Verify that the API key is valid and has the necessary permissions
3. **Service Account Issues**: If using a service account, check that it has the correct roles and permissions
4. **Deployment Configuration**: For Cloud Run deployments, verify that the Secret Manager is properly configured as described in DEPLOYMENT.md

To fix these issues:
- For local development: Set the GEMINI_API_KEY environment variable in a .env file
- For Cloud Run deployment: Update the secret in Secret Manager and redeploy the service

```bash
# Local development
echo "GEMINI_API_KEY=your-api-key" > .env

# Cloud Run deployment
echo -n "your-gemini-api-key" | gcloud secrets create gemini-api-key --data-file=-
```

## License

This test script is provided under the same license as the LineCoach project. 