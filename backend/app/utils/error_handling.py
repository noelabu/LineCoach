from fastapi import HTTPException
from typing import Any, Dict, Optional
import logging

def handle_exception(
    error: Exception, 
    status_code: int = 500, 
    detail_message: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> HTTPException:
    """
    Handle exceptions by creating an appropriate HTTPException
    
    Args:
        error: The exception that occurred
        status_code: HTTP status code to return
        detail_message: Custom message for the exception detail
        logger: Logger instance for logging the error
        
    Returns:
        HTTPException that can be raised
    """
    # Create detail message if not provided
    if detail_message is None:
        detail_message = f"Error processing request: {str(error)}"
    
    # Log the error if logger is provided
    if logger:
        logger.error(f"{detail_message}. Original error: {error}")
    
    # Return HTTPException
    return HTTPException(status_code=status_code, detail=detail_message)
