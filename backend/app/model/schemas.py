from pydantic import BaseModel
from typing import List, Optional

class CoachingRequest(BaseModel):
    """Request model for coaching API"""
    transcript: str
    conversation_history: Optional[List[str]] = []
    user_id: Optional[str] = None

class CoachingResponse(BaseModel):
    """Response model for coaching API"""
    suggestions: List[str]
    tone_analysis: Optional[str] = None
    escalation_needed: bool = False
    escalation_reason: Optional[str] = None
