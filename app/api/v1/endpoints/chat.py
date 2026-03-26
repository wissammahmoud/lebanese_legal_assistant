from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_service import RAGService
from app.core.security import verify_service_key
from fastapi.responses import StreamingResponse
import structlog
import json

log = structlog.get_logger()

router = APIRouter()

def get_rag_service():
    return RAGService()

async def _stream_event_generator(rag_service: RAGService, request: ChatRequest):
    try:
        async for chunk in rag_service.stream_query(request):
            yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as e:
        log.error("Streaming error", error=str(e))
        yield f"data: {json.dumps({'type': 'error', 'content': 'An internal error occurred. Please try again later.'})}\n\n"

# --- Service-to-service endpoints (require X-SERVICE-KEY) ---

@router.post("/chat/stream", dependencies=[Depends(verify_service_key)])
async def chat_stream(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    return StreamingResponse(_stream_event_generator(rag_service, request), media_type="text/event-stream")

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

# --- Web frontend endpoint (no service key, protected by CORS origin restriction) ---

@router.post("/chat/web-stream")
async def chat_web_stream(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    return StreamingResponse(_stream_event_generator(rag_service, request), media_type="text/event-stream")
