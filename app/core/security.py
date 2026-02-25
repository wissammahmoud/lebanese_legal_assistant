from fastapi import Header, HTTPException, Security
from app.core.config import settings

import structlog
log = structlog.get_logger()

async def verify_service_key(x_service_key: str = Header(..., alias="X-SERVICE-KEY")):
    """
    Validates the X-SERVICE-KEY header for service-to-service authentication.
    """
    if x_service_key != settings.SERVICE_API_KEY:
        log.warning("Invalid Service Key attempt", received=x_service_key, expected=settings.SERVICE_API_KEY)
        raise HTTPException(status_code=403, detail="Invalid Service Key")
    return x_service_key
