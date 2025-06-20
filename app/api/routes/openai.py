from fastapi import APIRouter, HTTPException
from app.model.schemas import CoachingRequest, CoachingResponse
from app.model.coach import process_coaching_request
from app.utils.logging import get_logger, log_error, log_request, log_response
from app.utils.error_handling import handle_exception

# Get logger for this module
logger = get_logger(__name__)
router = APIRouter()

@router.post("/coach", response_model=CoachingResponse)
async def get_coaching(request: CoachingRequest):
    """
    Process conversation transcript and provide coaching suggestions
    
    This endpoint analyzes conversation text and returns coaching suggestions,
    tone analysis, and escalation recommendations if needed.
    """
    try:
        # Log the incoming request
        log_request(logger, "coach", request.dict())
        
        # Process the coaching request using our model logic
        response = process_coaching_request(request)
        
        # Log the response
        log_response(logger, "coach", response.dict())
        
        return response
        
    except Exception as e:
        # Log the error with context
        log_error(
            logger, 
            "Error processing coaching request",
            e,
            {"transcript_length": len(request.transcript) if request.transcript else 0}
        )
        
        # Raise HTTP exception
        raise handle_exception(
            error=e,
            detail_message="Error processing coaching request",
            logger=logger
        ) 