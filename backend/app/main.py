from fastapi import FastAPI
from app.core.app import create_application
from app.api.main import api_router
from app.core.config import settings

app = create_application()

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "name": settings.PROJECT_NAME,
        "version": "1.0.0",
        "description": "LineCoach - Multi-agent Coaching and Real-time Escalation system"
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"} 