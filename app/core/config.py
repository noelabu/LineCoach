from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    DOMAIN: str = "localhost"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"
    CORS_ENABLED: bool = True
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOWED_ORIGINS: frozenset[str] = frozenset({"*"})
    CORS_ALLOWED_METHODS: frozenset[str] = frozenset({"*"})
    CORS_ALLOWED_HEADERS: frozenset[str] = frozenset({"*"})
    
    # Gemini API settings
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-pro"
    
    # Application settings
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"


settings = Settings()
