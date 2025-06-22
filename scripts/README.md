# LineCoach Helper Scripts

This directory contains helper scripts for the LineCoach project.

## Authentication Scripts

### `check_gemini_auth.py`

This script tests the Gemini API authentication to help diagnose issues with the LineCoach API.

**Usage:**
```bash
# Test with environment variables
python tests/check_gemini_auth.py

# Test with a specific API key
python tests/check_gemini_auth.py --api-key YOUR_API_KEY

# Test with a specific service account credentials file
python tests/check_gemini_auth.py --credentials-file path/to/credentials.json
```

### `fix_cloud_run_auth.sh`

This script helps fix Gemini API authentication issues in Cloud Run deployments.

**Usage:**
```bash
# Make the script executable
chmod +x scripts/fix_cloud_run_auth.sh

# Run the script
./scripts/fix_cloud_run_auth.sh
```

The script will:
1. Check if the Secret Manager API is enabled
2. Create or update the Gemini API key secret
3. Grant the service account access to the secret
4. Update the Cloud Run service to use the secret
5. Verify the deployment

## Requirements

- For `check_gemini_auth.py`:
  - Python 3.7+
  - `google-generativeai` package
  - `google-auth` package

- For `fix_cloud_run_auth.sh`:
  - Google Cloud SDK (`gcloud` CLI)
  - Bash shell
  - Active Google Cloud account with appropriate permissions 