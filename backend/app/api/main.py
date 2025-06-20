from fastapi import APIRouter
from .routes import openai

api_router = APIRouter()

# Include routers from route modules
api_router.include_router(openai.router, prefix="/openai", tags=["openai"])

# Add additional routers here as needed
# api_router.include_router(gemini.router, prefix="/gemini", tags=["gemini"])