from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's legal question")
    history: List[ChatMessage] = Field(default=[], description="Previous conversation history")
    user_context: Optional[Dict[str, Any]] = Field(default=None, description="Optional user metadata (e.g. language preference)")

class SourceDocument(BaseModel):
    id: Optional[int]
    score: float
    text: str
    source_type: str
    metadata: Optional[Dict[str, Any]]

class ChatResponse(BaseModel):
    response: str
    sources: List[SourceDocument] = []
    error: Optional[str] = None
