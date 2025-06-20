from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/coach", response_model=CoachingResponse)
async def get_coaching(request: CoachingRequest):
    """
    Process conversation transcript and provide coaching suggestions
    
    This endpoint analyzes conversation text and returns coaching suggestions,
    tone analysis, and escalation recommendations if needed.
    """
    try:
        # Mock response for now - in a real implementation, this would call
        # an OpenAI model or other coaching logic
        suggestions = [
            "Use more empathetic language when addressing customer concerns",
            "Offer specific solutions rather than general assurances",
            "Consider acknowledging the customer's frustration more directly"
        ]
        
        # Check for escalation keywords (simplified example)
        escalation_needed = False
        escalation_reason = None
        escalation_keywords = ["furious", "supervisor", "manager", "refund", "lawsuit"]
        
        if any(keyword in request.transcript.lower() for keyword in escalation_keywords):
            escalation_needed = True
            escalation_reason = "Customer showing signs of significant frustration"
        
        return CoachingResponse(
            suggestions=suggestions,
            tone_analysis="The conversation appears to have a neutral to slightly negative tone.",
            escalation_needed=escalation_needed,
            escalation_reason=escalation_reason
        )
        
    except Exception as e:
        logger.error(f"Error processing coaching request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing coaching request: {str(e)}") 