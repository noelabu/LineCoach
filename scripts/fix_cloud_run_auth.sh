#!/bin/bash
# Script to fix Gemini API authentication issues in Cloud Run deployment

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not found. Please install Google Cloud SDK.${NC}"
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is logged in to gcloud
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo -e "${YELLOW}You need to log in to Google Cloud first.${NC}"
    gcloud auth login
fi

# Get project ID
echo -e "${YELLOW}Fetching current project...${NC}"
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}No project selected. Please select a project:${NC}"
    gcloud projects list
    read -p "Enter project ID: " PROJECT_ID
    gcloud config set project $PROJECT_ID
fi

echo -e "${GREEN}Using project: ${PROJECT_ID}${NC}"

# Check if the service exists
echo -e "${YELLOW}Checking for LineCoach service...${NC}"
SERVICE_NAME="linecoach"
REGION="asia-southeast1"

if ! gcloud run services describe $SERVICE_NAME --region=$REGION &> /dev/null; then
    echo -e "${RED}Service '$SERVICE_NAME' not found in region '$REGION'.${NC}"
    read -p "Enter service name [linecoach]: " INPUT_SERVICE
    if [ ! -z "$INPUT_SERVICE" ]; then
        SERVICE_NAME=$INPUT_SERVICE
    fi
    
    read -p "Enter region [asia-southeast1]: " INPUT_REGION
    if [ ! -z "$INPUT_REGION" ]; then
        REGION=$INPUT_REGION
    fi
    
    if ! gcloud run services describe $SERVICE_NAME --region=$REGION &> /dev/null; then
        echo -e "${RED}Service '$SERVICE_NAME' not found in region '$REGION'. Exiting.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Found service: $SERVICE_NAME in region: $REGION${NC}"

# Check if Secret Manager API is enabled
echo -e "${YELLOW}Checking if Secret Manager API is enabled...${NC}"
if ! gcloud services list --enabled | grep -q secretmanager.googleapis.com; then
    echo -e "${YELLOW}Enabling Secret Manager API...${NC}"
    gcloud services enable secretmanager.googleapis.com
    echo -e "${GREEN}Secret Manager API enabled.${NC}"
else
    echo -e "${GREEN}Secret Manager API already enabled.${NC}"
fi

# Check if Gemini API key secret exists
echo -e "${YELLOW}Checking for Gemini API key secret...${NC}"
SECRET_NAME="gemini-api-key"

if ! gcloud secrets describe $SECRET_NAME &> /dev/null; then
    echo -e "${YELLOW}Secret '$SECRET_NAME' not found. Creating new secret...${NC}"
    read -p "Enter your Gemini API key: " GEMINI_API_KEY
    
    if [ -z "$GEMINI_API_KEY" ]; then
        echo -e "${RED}No API key provided. Exiting.${NC}"
        exit 1
    fi
    
    echo -n "$GEMINI_API_KEY" | gcloud secrets create $SECRET_NAME --data-file=-
    echo -e "${GREEN}Secret '$SECRET_NAME' created.${NC}"
else
    echo -e "${GREEN}Secret '$SECRET_NAME' exists.${NC}"
    read -p "Do you want to update the secret with a new API key? (y/n): " UPDATE_SECRET
    
    if [[ $UPDATE_SECRET == "y" || $UPDATE_SECRET == "Y" ]]; then
        read -p "Enter your new Gemini API key: " GEMINI_API_KEY
        
        if [ -z "$GEMINI_API_KEY" ]; then
            echo -e "${RED}No API key provided. Skipping update.${NC}"
        else
            echo -n "$GEMINI_API_KEY" | gcloud secrets versions add $SECRET_NAME --data-file=-
            echo -e "${GREEN}Secret '$SECRET_NAME' updated.${NC}"
        fi
    fi
fi

# Get service account
echo -e "${YELLOW}Getting service account for $SERVICE_NAME...${NC}"
SERVICE_ACCOUNT=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(spec.template.spec.serviceAccountName)")

if [ -z "$SERVICE_ACCOUNT" ]; then
    echo -e "${YELLOW}Service is using default compute service account.${NC}"
    SERVICE_ACCOUNT="$PROJECT_ID-compute@developer.gserviceaccount.com"
else
    echo -e "${GREEN}Service is using service account: $SERVICE_ACCOUNT${NC}"
fi

# Grant access to the secret
echo -e "${YELLOW}Granting secret access to service account...${NC}"
gcloud secrets add-iam-policy-binding $SECRET_NAME \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/secretmanager.secretAccessor"

echo -e "${GREEN}Secret access granted to $SERVICE_ACCOUNT.${NC}"

# Update the Cloud Run service
echo -e "${YELLOW}Updating Cloud Run service to use the secret...${NC}"
gcloud run services update $SERVICE_NAME \
  --region=$REGION \
  --set-secrets=GEMINI_API_KEY=$SECRET_NAME:latest

echo -e "${GREEN}Service updated to use the secret.${NC}"

# Verify deployment
echo -e "${YELLOW}Verifying deployment...${NC}"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')
echo -e "${GREEN}Service URL: $SERVICE_URL${NC}"

echo -e "${YELLOW}Testing root endpoint...${NC}"
curl -s $SERVICE_URL | grep -q "LineCoach API is running"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Root endpoint is working.${NC}"
else
    echo -e "${RED}Root endpoint is not working properly.${NC}"
fi

echo -e "\n${GREEN}=====================================${NC}"
echo -e "${GREEN}Authentication setup complete!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo -e "To test the API endpoints, run:"
echo -e "python tests/test_linecoach_api.py --url $SERVICE_URL"
echo -e "\nIf you still encounter authentication issues, check:"
echo -e "1. The Gemini API key is valid and has the necessary permissions"
echo -e "2. The service account has access to the secret"
echo -e "3. The Cloud Run service is correctly configured to use the secret"
echo -e "\nTo check the logs for the service, run:"
echo -e "gcloud logs read --project=$PROJECT_ID --resource=cloud_run_revision --service_name=$SERVICE_NAME --region=$REGION" 