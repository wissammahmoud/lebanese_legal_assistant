from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_service import RAGService
from app.core.security import verify_service_key
import structlog

log = structlog.get_logger()

router = APIRouter()

def get_rag_service():
    return RAGService()

from fastapi.responses import StreamingResponse
import json

@router.post("/chat/stream", dependencies=[Depends(verify_service_key)])
async def chat_stream(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    async def event_generator():
        try:
            async for chunk in rag_service.stream_query(request):
                # Yield data in SSE format
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            log.error("Streaming error", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_service_key)])
async def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    try:
        return await rag_service.process_query(request)
    except Exception as e:
        log.error("Unhandled error in chat endpoint", error=str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")
