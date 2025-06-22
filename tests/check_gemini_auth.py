#!/usr/bin/env python3
"""
Gemini API Authentication Check Script

This script tests the Gemini API authentication to help diagnose issues with the LineCoach API.
It attempts to authenticate with Gemini API using both API key and service account methods.

Usage:
    python check_gemini_auth.py [--api-key YOUR_API_KEY]
"""

import os
import sys
import argparse
import google.generativeai as genai
from google.auth import default
from google.oauth2 import service_account
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_api_key_auth(api_key):
    """Test authentication with Gemini API using API key."""
    try:
        logger.info("Testing Gemini API authentication with API key...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content("Hello, this is a test message to check authentication.")
        logger.info(f"✅ API key authentication successful! Response: {response.text[:50]}...")
        return True
    except Exception as e:
        logger.error(f"❌ API key authentication failed: {e}")
        return False


def check_service_account_auth(credentials_path=None):
    """Test authentication with Gemini API using service account."""
    try:
        logger.info("Testing Gemini API authentication with service account...")
        
        if credentials_path:
            if not os.path.exists(credentials_path):
                logger.error(f"❌ Service account credentials file not found: {credentials_path}")
                return False
                
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            genai.configure(credentials=credentials)
        else:
            # Try default credentials
            credentials, project = default()
            genai.configure(credentials=credentials)
            
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content("Hello, this is a test message to check authentication.")
        logger.info(f"✅ Service account authentication successful! Response: {response.text[:50]}...")
        return True
    except Exception as e:
        logger.error(f"❌ Service account authentication failed: {e}")
        return False


def check_environment():
    """Check environment for credentials."""
    logger.info("Checking environment for credentials...")
    
    # Check for API key in environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        logger.info("✅ GEMINI_API_KEY environment variable found")
    else:
        logger.warning("⚠️ GEMINI_API_KEY environment variable not found")
    
    # Check for service account credentials
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        if os.path.exists(credentials_path):
            logger.info(f"✅ GOOGLE_APPLICATION_CREDENTIALS points to existing file: {credentials_path}")
        else:
            logger.warning(f"⚠️ GOOGLE_APPLICATION_CREDENTIALS points to non-existent file: {credentials_path}")
    else:
        logger.warning("⚠️ GOOGLE_APPLICATION_CREDENTIALS environment variable not found")
    
    return api_key, credentials_path


def main():
    parser = argparse.ArgumentParser(description="Test Gemini API authentication")
    parser.add_argument("--api-key", help="Gemini API key to test")
    parser.add_argument("--credentials-file", help="Path to service account credentials JSON file")
    args = parser.parse_args()
    
    print("\n🔍 Starting Gemini API authentication check...\n")
    
    # Check environment
    env_api_key, env_credentials_path = check_environment()
    
    # Use provided API key or fall back to environment
    api_key = args.api_key or env_api_key
    credentials_path = args.credentials_file or env_credentials_path
    
    # Test API key authentication if available
    api_key_success = False
    if api_key:
        api_key_success = check_api_key_auth(api_key)
    else:
        logger.warning("⚠️ No API key available to test")
    
    # Test service account authentication if available
    service_account_success = False
    if credentials_path:
        service_account_success = check_service_account_auth(credentials_path)
    else:
        logger.warning("⚠️ No service account credentials path available")
        # Try default credentials anyway
        service_account_success = check_service_account_auth()
    
    # Summary
    print("\n📊 Authentication Check Summary:")
    print(f"  - API Key Authentication: {'✅ PASSED' if api_key_success else '❌ FAILED'}")
    print(f"  - Service Account Authentication: {'✅ PASSED' if service_account_success else '❌ FAILED'}")
    
    if not (api_key_success or service_account_success):
        print("\n❌ All authentication methods failed. Please check your credentials.")
        print("\nTroubleshooting steps:")
        print("1. Ensure you have a valid Gemini API key")
        print("2. Set the GEMINI_API_KEY environment variable")
        print("3. Or set up a service account with appropriate permissions")
        print("4. Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to point to your service account JSON file")
        print("\nFor Cloud Run deployments:")
        print("1. Check that the Secret Manager is properly configured")
        print("2. Verify the service account has access to the secrets")
        print("3. Redeploy the service after updating the secrets")
        sys.exit(1)
    else:
        print("\n✅ At least one authentication method succeeded!")
        sys.exit(0)


if __name__ == "__main__":
    main() 