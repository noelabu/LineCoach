import os
from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = Path(__file__).parents[2] / '.env'
load_dotenv(dotenv_path)

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    # Application settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LineCoach"
    
    # Authentication
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    PROJECT_ID: str = os.getenv("PROJECT_ID", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # CORS
    BACKEND_CORS_ORIGINS: list[str] = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings() 