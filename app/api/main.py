from fastapi import APIRouter

api_router = APIRouter()

# Import and include routers from endpoints
from app.api.endpoints import linecoach

# Include the linecoach router with a prefix
api_router.include_router(linecoach.router, prefix="/linecoach", tags=["linecoach"]) 