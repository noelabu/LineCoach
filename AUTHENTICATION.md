# LineCoach Authentication Guide

This document provides detailed guidance on setting up and troubleshooting authentication for the LineCoach application, particularly for the Gemini API integration.

## Authentication Methods

LineCoach supports two authentication methods for the Gemini API:

1. **API Key Authentication** (recommended for development)
2. **Service Account Authentication** (recommended for production)

## API Key Authentication

### Setting Up API Key Authentication

1. **Get a Gemini API key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key or use an existing one

2. **Configure the API key**:
   - For local development:
     ```
     # In your .env file
     GEMINI_API_KEY='your-gemini-api-key'
     ```
   - For Cloud Run deployment:
     ```bash
     # Create a secret
     echo -n "your-gemini-api-key" | gcloud secrets create gemini-api-key --data-file=-
     
     # Deploy with the secret
     gcloud run deploy linecoach \
       --set-secrets GEMINI_API_KEY=gemini-api-key:latest \
       [other deployment options]
     ```

## Service Account Authentication

### Setting Up Service Account Authentication

1. **Create a service account**:
   ```bash
   gcloud iam service-accounts create linecoach-sa \
     --display-name="LineCoach Service Account"
   ```

2. **Grant necessary permissions**:
   ```bash
   # Grant Vertex AI User role
   gcloud projects add-iam-policy-binding your-project-id \
     --member="serviceAccount:linecoach-sa@your-project-id.iam.gserviceaccount.com" \
     --role="roles/aiplatform.user"
   ```

3. **Create and download credentials**:
   ```bash
   gcloud iam service-accounts keys create credentials.json \
     --iam-account=linecoach-sa@your-project-id.iam.gserviceaccount.com
   ```

4. **Configure credentials**:
   - For local development:
     ```
     # In your .env file
     GOOGLE_APPLICATION_CREDENTIALS='path/to/credentials.json'
     ```
   - For Cloud Run deployment:
     ```bash
     # Create a secret
     gcloud secrets create sa-credentials --data-file=credentials.json
     
     # Deploy with the secret
     gcloud run deploy linecoach \
       --set-secrets GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa-credentials:latest \
       [other deployment options]
     ```

## Common Authentication Errors

### Error: 403 Request had insufficient authentication scopes

This error occurs when the authentication credentials don't have the necessary permissions to access the Gemini API.

**Possible causes and solutions**:

1. **Invalid or expired API key**:
   - Generate a new API key in Google AI Studio
   - Update the API key in your .env file or Secret Manager

2. **Insufficient service account permissions**:
   - Ensure the service account has the "Vertex AI User" role
   - Check if the service account has access to the Gemini API

3. **Cloud Run configuration issues**:
   - Verify the service account has access to the Secret Manager secret
   - Check that the secret is correctly mounted in the Cloud Run service

## Troubleshooting Tools

### Authentication Check Script

Use the provided authentication check script to diagnose issues:

```bash
# Run with environment variables
python tests/check_gemini_auth.py

# Run with a specific API key
python tests/check_gemini_auth.py --api-key YOUR_API_KEY

# Run with a specific service account file
python tests/check_gemini_auth.py --credentials-file path/to/credentials.json
```

### Cloud Run Authentication Fix Script

Use the provided script to fix Cloud Run authentication issues:

```bash
./scripts/fix_cloud_run_auth.sh
```

This script will:
1. Check if the Secret Manager API is enabled
2. Create or update the Gemini API key secret
3. Grant the service account access to the secret
4. Update the Cloud Run service to use the secret
5. Verify the deployment

## Checking Authentication Status

### Local Development

To check if your authentication is working locally:

```bash
# Set environment variables
export GEMINI_API_KEY='your-api-key'
# OR
export GOOGLE_APPLICATION_CREDENTIALS='path/to/credentials.json'

# Run the authentication check script
python tests/check_gemini_auth.py
```

### Cloud Run Deployment

To check if your authentication is working in Cloud Run:

```bash
# Check Cloud Run logs
gcloud logs read --project=your-project-id \
  --resource=cloud_run_revision \
  --service_name=linecoach \
  --region=asia-southeast1 \
  --filter="textPayload:\"Gemini API\""
```

## Additional Resources

- [Google AI Studio Documentation](https://ai.google.dev/docs)
- [Google Cloud Authentication Documentation](https://cloud.google.com/docs/authentication)
- [Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)