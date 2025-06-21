#!/usr/bin/env python3
"""
Simple API server for testing the LineCoach API.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="LineCoach API Test")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "LineCoach API Test Server"}

# Pydantic models
class AudioRequest(BaseModel):
    audio_data: str
    sample_rate: int = 16000
    channels: int = 1
    content_type: str = "audio/wav"

class TranscriptionResponse(BaseModel):
    transcript: str

class CoachingRequest(BaseModel):
    transcript: str
    conversation_history: List[str] = []

class AnalysisResponse(BaseModel):
    sentiment: float
    empathy: float
    resolution: float
    escalation: float

class RecommendationResponse(BaseModel):
    recommendations: str
    analysis: Optional[AnalysisResponse] = None

# API endpoints
@app.post("/api/v1/linecoach/transcribe", response_model=TranscriptionResponse)
async def transcribe(request: AudioRequest):
    # Simple test response
    return {"transcript": "This is a test transcript"}

@app.post("/api/v1/linecoach/analyze", response_model=AnalysisResponse)
async def analyze(request: CoachingRequest):
    # Simple test response
    return {
        "sentiment": 0.5,
        "empathy": 7.5,
        "resolution": 60.0,
        "escalation": 20.0
    }

@app.post("/api/v1/linecoach/coach", response_model=RecommendationResponse)
async def coach(request: CoachingRequest):
    # Simple test response
    return {"recommendations": "This is a test recommendation"}

@app.post("/api/v1/linecoach/full_coaching", response_model=RecommendationResponse)
async def full_coaching(request: CoachingRequest):
    # Simple test response
    return {
        "recommendations": "This is a test recommendation",
        "analysis": {
            "sentiment": 0.5,
            "empathy": 7.5,
            "resolution": 60.0,
            "escalation": 20.0
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 