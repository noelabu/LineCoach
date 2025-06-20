import logging
from typing import Any, Dict, Optional

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name
    
    Args:
        name: The name of the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def log_error(logger: logging.Logger, message: str, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an error with context information
    
    Args:
        logger: Logger instance
        message: Error message
        error: Exception that occurred
        context: Additional context information
    """
    log_message = f"{message}: {str(error)}"
    if context:
        log_message += f" Context: {context}"
    logger.error(log_message)

def log_request(logger: logging.Logger, endpoint: str, request_data: Dict[str, Any]) -> None:
    """
    Log information about an API request
    
    Args:
        logger: Logger instance
        endpoint: The API endpoint being called
        request_data: The request data
    """
    logger.info(f"Request to {endpoint}: {request_data}")

def log_response(logger: logging.Logger, endpoint: str, response_data: Dict[str, Any]) -> None:
    """
    Log information about an API response
    
    Args:
        logger: Logger instance
        endpoint: The API endpoint being called
        response_data: The response data
    """
    logger.info(f"Response from {endpoint}: {response_data}")
