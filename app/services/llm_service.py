from openai import AsyncOpenAI
import structlog
from app.core.config import settings
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

log = structlog.get_logger()

class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    @traceable(run_type="llm", name="Final LLM Generation")
    async def generate_response(self, messages: list[dict], temperature: float = 0.2) -> str:
        log.info("Calling LLM", model="gpt-4o-mini", message_count=len(messages))
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            log.error("LLM generation failed", error=str(e))
            raise e

    @traceable(run_type="llm", name="Streaming LLM Generation")
    async def stream_response(self, messages: list[dict], temperature: float = 0.2):
        log.info("Calling Streaming LLM", model="gpt-4o-mini", message_count=len(messages))
        try:
            stream = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temperature,
                stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            log.error("LLM streaming failed", error=str(e))
            raise e
