from fastapi import APIRouter

api_router = APIRouter()

# Import and include routers from endpoints
from app.api.endpoints import linecoach, linecoach_gcp

# Include the linecoach router with a prefix
api_router.include_router(linecoach.router, prefix="/linecoach", tags=["linecoach"])

# Include the GCP linecoach router with a prefix
api_router.include_router(linecoach_gcp.router, prefix="/linecoach-gcp", tags=["linecoach-gcp"]) 