# LineCoach Deployment Guide

This guide provides detailed instructions for deploying the LineCoach application to Google Cloud Run.

## Prerequisites

1. **Google Cloud SDK**: Install and configure the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. **Docker**: Install [Docker](https://docs.docker.com/get-docker/) for building container images
3. **Google Cloud Project**: Create or use an existing Google Cloud project
4. **Permissions**: Ensure you have the necessary permissions to deploy to Cloud Run and create secrets

## Step 1: Set up your Google Cloud Project

```bash
# Set your project ID
export PROJECT_ID=your-project-id
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com \
  run.googleapis.com \
  containerregistry.googleapis.com \
  secretmanager.googleapis.com
```

## Step 2: Set up Secret Manager for API Keys

```bash
# Create a secret for the Gemini API key
echo -n "your-gemini-api-key" | gcloud secrets create gemini-api-key --data-file=-

# Grant access to the Cloud Run service account
gcloud secrets add-iam-policy-binding gemini-api-key \
  --member="serviceAccount:$PROJECT_ID-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## Step 3: Deploy using Cloud Build

This method uses the Cloud Build configuration file to automate the build and deployment process.

```bash
# For development environment
gcloud builds submit --config=cloudbuild.dev.yaml \
  --substitutions=_PROJECT_ID=$PROJECT_ID,_SERVICE_NAME=linecoach,_ENVIRONMENT=dev,_REGION=asia-southeast1
```

## Step 4: Manual Deployment (Alternative)

If you prefer to deploy manually or are troubleshooting the Cloud Build process:

```bash
# Build the container
docker build -t gcr.io/$PROJECT_ID/linecoach -f Dockerfile.cloud .

# Configure Docker to use gcloud as a credential helper
gcloud auth configure-docker

# Push to Container Registry
docker push gcr.io/$PROJECT_ID/linecoach

# Deploy to Cloud Run
gcloud run deploy linecoach \
  --image gcr.io/$PROJECT_ID/linecoach \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest \
  --env-vars-file deployment_values/env.dev.yaml
```

## Step 5: Verify Deployment

```bash
# Get the service URL
gcloud run services describe linecoach --region=asia-southeast1 --format='value(status.url)'

# Test the API
curl $(gcloud run services describe linecoach --region=asia-southeast1 --format='value(status.url)')/api/v1/health
```

## Environment Configuration

The application uses environment variables for configuration. These are set in the `deployment_values/env.dev.yaml` file.

Key configuration parameters:
- `ENVIRONMENT`: Set to "staging" or "production"
- `DEBUG`: Set to "False" for production deployments
- `LOG_LEVEL`: Set to appropriate logging level ("INFO", "WARNING", "ERROR")
- `GEMINI_API_KEY`: Set via Secret Manager
- `GEMINI_MODEL`: The Gemini model to use

## Troubleshooting

1. **Docker Build Issues**:
   - If you encounter PyAudio dependency issues, use `Dockerfile.cloud` which excludes PyAudio
   - Check that you're using Python 3.12 as specified in the pyproject.toml

2. **Deployment Issues**:
   - Verify that all required APIs are enabled
   - Check that the service account has the necessary permissions
   - Review Cloud Build logs for any errors

3. **Runtime Issues**:
   - Check Cloud Run logs for application errors
   - Verify that all environment variables are correctly set
   - Ensure the Gemini API key is correctly configured in Secret Manager

## Updating the Deployment

To update an existing deployment:

```bash
# Build and push a new image
docker build -t gcr.io/$PROJECT_ID/linecoach:v2 -f Dockerfile.cloud .
docker push gcr.io/$PROJECT_ID/linecoach:v2

# Update the Cloud Run service
gcloud run services update linecoach \
  --image gcr.io/$PROJECT_ID/linecoach:v2 \
  --region asia-southeast1
``` 